"""VLM-as-Judge for 3D scene reconstruction quality, adapted from BlenderWorld.

Adapts the paper's "binary rubric + 8-dimension holistic quality" protocol to a
single-camera absolute-scoring judge that reuses our existing klink transport
(`v2s_integration.agent.api_transport.invoke`) and image assembly
(`v2s_integration.agent.api_context.image_content`).

Two model passes per judge call, matching the paper's split:

1. Rubric pass  -- each binary criterion scored exactly 0 or 1, weighted by
   `importance_weight` (default 1).  `rubric_score = 100 * sum(score*w)/sum(w)`.
2. Dimension pass -- 8 holistic dimensions scored 1-10.  `dimension_score =
   100 * sum(score) / (10 * 8)`.

Fused score (the paper's uncapped 1:1 formula):
    fused_score = 0.5 * rubric_score + 0.5 * dimension_score

This module is offline and Blender-free: it only needs the reference image(s),
the submission render(s), a rubric, and the task prompt. It does NOT touch the
training loop or the reward path; that integration is gated on calibration
results (see weekly_report/1104/doc/vlm_judge_design_20261102.md).
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from v2s_integration.agent.api_context import api_payload, image_content
from v2s_integration.agent.api_transport import invoke

# ---------------------------------------------------------------------------
# Judge identity. The model field is deliberately switchable (the paper uses
# gpt-5.6-terra; our klink channel serves qwen3.8-max by default). Keep the
# judge model isolated from the generation agent, exactly like JUDGE_MODEL in
# BlenderWorld.
# ---------------------------------------------------------------------------
KLINK_URL = 'http://klink.internal'
JUDGE_MODEL = 'gpt-5.6-terra'

# Low-temperature deterministic judging. Note: klink-served gpt-5.6-terra does
# NOT accept the Alibaba thinking params (enable_thinking/thinking_budget/
# preserve_thinking) that qwen3.8-max uses; it performs its own reasoning
# internally. Keep only provider-neutral sampling params so the same module
# works against either model family.
JUDGE_PARAMETERS = {
    'temperature': 0.0,
    'top_p': 1.0,
}

MAX_OUTPUT_TOKENS = 8192
TIMEOUT = 900

# ---------------------------------------------------------------------------
# 8 holistic quality dimensions (verbatim from BlenderWorld, ids kept stable).
# ---------------------------------------------------------------------------
DIMENSIONS = [
    {
        "id": "scene_fidelity",
        "name": "场景还原 / scene fidelity",
        "desc": "Overall, do the renders reproduce the target scene (from the "
                "reference image and/or query) as a whole: right kind of scene, "
                "the main objects present, the overall arrangement recognizable.",
    },
    {
        "id": "physics",
        "name": "物理合理性 / physical plausibility",
        "desc": "Support, connection and structure: objects rest on the floor or "
                "are properly attached, no floating/unsupported parts, no "
                "interpenetration, parts join into real structures, surfaces have thickness.",
    },
    {
        "id": "style_color",
        "name": "样式与颜色 / object style & color",
        "desc": "Each object's shape/silhouette actually reads as the intended "
                "object (not a featureless box), and its colors match the target.",
    },
    {
        "id": "realism",
        "name": "overall realism",
        "desc": "As one image, is it believable and coherent, with acceptable "
                "render quality? (Avoid washed-out, foggy, low-contrast issues.)",
    },
    {
        "id": "layout_spatial",
        "name": "布局与空间关系 / layout & spatial relations",
        "desc": "Relative positions, orientations and spacing of objects match "
                "the reference/query (what is next to / on / in front of what).",
    },
    {
        "id": "proportion_scale",
        "name": "比例与尺度 / proportion & scale",
        "desc": "Size ratios are sensible: object-to-object and object-to-room "
                "proportions look right, nothing miniature or oversized.",
    },
    {
        "id": "lighting_material",
        "name": "光照与材质质感 / lighting & material",
        "desc": "Lighting, shadows and material appearance look plausible for the "
                "intended surface. A clean matte / flat finish that matches the "
                "intended object is acceptable.",
    },
    {
        "id": "completeness",
        "name": "物件完整性 / completeness",
        "desc": "Every object the query/reference requires is actually present, "
                "in roughly the right count; nothing important is missing.",
    },
]

# ---------------------------------------------------------------------------
# Prompts, distilled from BlenderWorld's JUDGE_SYSTEM / DIMENSION_SYSTEM.
# We drop the multi-view Group A/B/C, animation-frame, and media-OSS rules
# because our probe is single-camera with a fixed reference. We keep the core
# discipline: score what the submission actually SHOWS, not what the prompt says.
# ---------------------------------------------------------------------------
RUBRIC_SYSTEM = """\
You are a strict visual judge for Blender scene reconstruction tasks.

You will receive:
1. The original task prompt.
2. One reference image of the target scene.
3. One submitted render of the artifact being judged.
4. A list of binary rubrics. Each rubric must be scored exactly 0 or 1.

Core principle:
Score what the submitted image actually SHOWS, not what the prompt says it
should contain. The prompt and reference define the TARGET; the submitted render
is the SUBMISSION being judged. Never credit the submission for something that
is only in the prompt/reference but not visible in the render.

Scoring rules:
- Score 1 only when the submitted image provides clear, unambiguous visual
  evidence that the criterion is satisfied.
- Score 0 when the condition is absent, contradicted, too ambiguous, or visible
  only as a vague/partial hint. When genuinely in doubt, score 0.
- Use the reference image only to understand the target's intent, layout,
  proportions, materials, and details. Do not score the reference itself.
- If the render is blank, clipped, noisy, or unusable, score 0 on the affected
  rubrics rather than guessing.
- An object that is indistinguishable from a plain box, slab, plane, or blob does
  NOT satisfy identity/proportion/quality rubrics. Do not give benefit of the
  doubt to featureless geometry.

Evidence discipline:
- Each rubric's "evidence" must be a concrete visual observation grounded in the
  submitted render (what is actually seen, and where). Do NOT restate or
  paraphrase the criterion as the evidence.

Output constraints:
- Every rubric id from the input must appear exactly once in the output.
- Each score must be integer 0 or 1.

Return ONLY a JSON object of this exact shape:
{
  "rubric_scores": [
    {"id": "R01", "score": 0, "evidence": "Concrete evidence grounded in the render.", "confidence": "high|medium|low"}
  ],
  "overall_notes": "Short summary of the artifact's main strengths and failures."
}
"""

RUBRIC_USER_TEMPLATE = """\
# Task Prompt
{prompt}

# Binary Rubrics To Score
{rubrics}

Reference image shows the target; the submitted render is the artifact being
judged. Score every rubric strictly as 0 or 1.
"""

DIMENSION_SYSTEM = """\
You are a strict holistic judge for Blender scene reconstruction tasks. Instead
of tiny binary checks, you rate the submission on a few broad dimensions.

You will receive:
1. The task prompt (the user's intent).
2. One reference image of the target scene.
3. One submitted render of the artifact being judged.

Core principle:
Score what the render actually SHOWS, not what the prompt says. The prompt and
reference define the TARGET; the render is the SUBMISSION being judged. Never
credit the submission for something that is only in the prompt/reference but not
visible in the render. If an object is only a featureless box/blob where a
specific object is expected, penalize style_color, scene_fidelity and realism.

Score EACH dimension as an integer from 1 to 10 using this scale:
- 9-10: essentially matches the target; only trivial flaws.
- 7-8:  good; recognizable and mostly correct with minor issues.
- 5-6:  acceptable but with clear, obvious problems.
- 3-4:  poor; major failures, only loosely resembles the target.
- 1-2:  absent, broken, or unrecognizable.

Rules:
- Be strict and calibrated; do not cluster everything at 7-8. Reserve 9-10 for
  genuinely faithful results and use low scores when warranted.
- Each dimension needs a concrete "reasons" grounded in the render, plus an
  "issues" list of specific problems you observed (empty if none).

Return ONLY a JSON object of this exact shape:
{
  "dimensions": [
    {"id": "scene_fidelity", "score": 7, "reasons": "concrete, view-grounded", "issues": ["..."]}
  ],
  "overall_notes": "Short summary of main strengths and failures."
}
Include every dimension id exactly once.
"""

DIMENSION_USER_TEMPLATE = """\
# Task Prompt / Query
{prompt}

# Dimensions To Score (integer 1-10 each)
{dimensions}

Score every dimension 1-10 based strictly on the submitted render. The reference
image and query define the target; the render is the submission being judged.
"""


# ---------------------------------------------------------------------------
# Robust JSON extraction (ported from BlenderWorld's parse_json_obj).
# ---------------------------------------------------------------------------
def _clean_json(raw):
    text = (raw or '').strip()
    if text.startswith('```'):
        text = text.strip('`')
        if text.lower().startswith('json'):
            text = text[4:]
    return text.strip()


def _extract_first_json_object(text):
    start = text.find('{')
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start: idx + 1]
    return text[start:] if start >= 0 else None


def parse_json_obj(raw):
    text = _clean_json(raw)
    candidates = [text]
    extracted = _extract_first_json_object(text)
    if extracted and extracted not in candidates:
        candidates.append(extracted)
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


# ---------------------------------------------------------------------------
# Normalization + scoring (ported from BlenderWorld's normalize_scores /
# build_score_summary / score_dimensions).
# ---------------------------------------------------------------------------
def normalize_scores(obj, rubrics):
    """Return (ok, normalized_scores, error)."""
    if not obj:
        return False, [], 'model output was not valid JSON'
    raw_scores = obj.get('rubric_scores')
    if not isinstance(raw_scores, list):
        return False, [], 'missing rubric_scores list'

    by_id = {}
    for item in raw_scores:
        if not isinstance(item, dict):
            continue
        rid = str(item.get('id', '')).strip()
        if not rid:
            continue
        score = item.get('score')
        if score in ('0', '0.0'):
            score = 0
        elif score in ('1', '1.0'):
            score = 1
        if score not in (0, 1):
            continue
        by_id[rid] = {
            'id': rid,
            'score': int(score),
            'evidence': str(item.get('evidence', '')).strip(),
            'confidence': str(item.get('confidence', 'medium')).strip() or 'medium',
        }

    normalized, missing = [], []
    for rubric in rubrics:
        rid = str(rubric.get('id', '')).strip()
        if rid in by_id:
            normalized.append(by_id[rid])
        else:
            missing.append(rid)
    if missing:
        return False, normalized, f'missing scores for ids: {missing[:10]}'
    return True, normalized, ''


def build_score_summary(rubrics, scores):
    score_by_id = {str(item.get('id')): int(item.get('score', 0)) for item in scores}
    raw_pass = raw_total = 0
    weighted_pass = weighted_total = 0.0
    missing_scores = []
    for rubric in rubrics:
        rid = str(rubric.get('id', '')).strip()
        raw_total += 1
        score = 1 if score_by_id.get(rid) else 0
        if rid not in score_by_id:
            missing_scores.append(rid)
        raw_pass += score
        weight = float(rubric.get('importance_weight') or rubric.get('weight') or 1)
        weighted_pass += score * weight
        weighted_total += weight
    return {
        'raw_pass': raw_pass,
        'raw_total': raw_total,
        'raw_score_0_100': (100.0 * raw_pass / raw_total) if raw_total else 0.0,
        'weighted_pass': weighted_pass,
        'weighted_total': weighted_total,
        'weighted_score_0_100': (100.0 * weighted_pass / weighted_total) if weighted_total else 0.0,
        'missing_scores': missing_scores,
    }


def score_dimensions(obj, dimensions=DIMENSIONS):
    """Return a dict with ok / error / dimension_scores / dimension_score_0_100."""
    result = {
        'ok': False, 'error': '', 'dimension_scores': [],
        'dimension_score_0_100': 0.0,
    }
    if not obj:
        result['error'] = 'dimension output was not valid JSON'
        return result
    raw = obj.get('dimensions')
    if not isinstance(raw, list):
        result['error'] = 'missing dimensions list'
        return result

    by_id = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        did = str(item.get('id', '')).strip()
        if not did:
            continue
        try:
            score = int(round(float(item.get('score'))))
        except (TypeError, ValueError):
            continue
        score = max(1, min(10, score))
        by_id[did] = {
            'id': did, 'score': score,
            'reasons': str(item.get('reasons', '')).strip(),
            'issues': item.get('issues') if isinstance(item.get('issues'), list) else [],
        }

    scores, missing, total = [], [], 0
    for dim in dimensions:
        did = str(dim.get('id', '')).strip()
        row = by_id.get(did)
        if row is None:
            missing.append(did)
            scores.append({'id': did, 'score': 0, 'reasons': '', 'issues': [], 'missing': True})
        else:
            row['name'] = dim.get('name')
            scores.append(row)
            total += row['score']

    denom = 10 * len(dimensions)
    result.update({
        'ok': not missing,
        'error': 'missing dimensions: ' + ', '.join(missing) if missing else '',
        'dimension_scores': scores,
        'dimension_score_0_100': (100.0 * total / denom) if denom else 0.0,
        'missing_dimensions': missing,
    })
    return result


# ---------------------------------------------------------------------------
# Transport helper: one chat call through klink, reusing invoke + api_payload.
# ---------------------------------------------------------------------------
def _call_model(key, system, user_text, reference_images, submission_images,
                folder, *, model=JUDGE_MODEL, cache_key, parameters=None,
                max_output_tokens=MAX_OUTPUT_TOKENS):
    refs = [{'path': str(p), 'label': 'Reference target'} for p in reference_images]
    subs = [{'path': str(p), 'label': 'Submission render'} for p in submission_images]
    content = image_content(user_text, refs + subs)
    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': content},
    ]
    params = dict(JUDGE_PARAMETERS)
    if parameters:
        params.update(parameters)
    payload = api_payload(messages, [], model=model, protocol='chat',
                          max_output_tokens=max_output_tokens, cache_key=cache_key,
                          parameters=params)
    for name in ['tools', 'tool_choice', 'parallel_tool_calls']:
        payload.pop(name, None)
    payload = _adapt_gpt_payload(payload, model)
    message, record = invoke(KLINK_URL, key, payload, folder, timeout=TIMEOUT)
    return message, record


def _adapt_gpt_payload(payload, model):
    """Adapt the chat payload for gpt-* reasoning models served by klink.

    gpt-5.6-terra rejects `max_tokens` (wants `max_completion_tokens`) and any
    explicit temperature/top_p other than their defaults. qwen3.8-max accepts
    `max_tokens` + temperature. Keep the two families working through the same
    module by rewriting the fields here instead of touching the shared
    api_payload that the probe/rollout path depends on.
    """
    if not str(model).startswith('gpt-'):
        return payload
    if 'max_tokens' in payload and 'max_completion_tokens' not in payload:
        payload['max_completion_tokens'] = payload.pop('max_tokens')
    for field in ('temperature', 'top_p', 'reasoning_effort'):
        payload.pop(field, None)
    return payload


def _format_rubric_block(rubrics):
    lines = []
    for r in rubrics:
        rid = r.get('id', '')
        weight = float(r.get('importance_weight') or r.get('weight') or 1)
        text = r.get('text', r.get('pass_condition', ''))
        lines.append(f"- {rid} (weight={weight:g}): {text}")
    return '\n'.join(lines)


def _format_dimension_block(dimensions):
    lines = []
    for d in dimensions:
        lines.append(f"- {d['id']} ({d['name']}): {d['desc']}")
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------
def judge(*, key, reference_images, submission_images, rubrics, task_prompt,
          folder, model=JUDGE_MODEL, dimensions=None):
    """Score one submission against a reference.

    Args:
        key: klink API key string.
        reference_images: list of Path/str to the target reference image(s).
        submission_images: list of Path/str to the submission render(s).
        rubrics: list of {'id', 'text'/'pass_condition', 'importance_weight'/'weight'}.
        task_prompt: the task instruction/query text.
        folder: Path where request/response artifacts are dumped (mkdir'd).
        model: judge model id (default qwen3.8-max).
        dimensions: optional override of the 8 holistic dimensions.

    Returns a dict with rubric_scores, rubric_score_0_100, dimension_scores,
    dimension_score_0_100, fused_score_0_100, model, usage, and per-pass notes.
    """
    folder = Path(folder)
    dims = dimensions if dimensions is not None else DIMENSIONS

    # Pass 1: binary rubrics.
    rubric_user = RUBRIC_USER_TEMPLATE.format(
        prompt=task_prompt, rubrics=_format_rubric_block(rubrics))
    rubric_msg, rubric_rec = _call_model(
        key, RUBRIC_SYSTEM, rubric_user, reference_images, submission_images,
        folder / 'rubric', model=model, cache_key='vlm-judge-rubric')
    rubric_obj = parse_json_obj(rubric_msg.get('content'))
    ok, rubric_scores, rubric_error = normalize_scores(rubric_obj, rubrics)
    if not ok:
        raise RuntimeError('Rubric pass invalid: ' + rubric_error)
    rubric_summary = build_score_summary(rubrics, rubric_scores)
    rubric_notes = (rubric_obj or {}).get('overall_notes', '')

    # Pass 2: holistic dimensions.
    dim_user = DIMENSION_USER_TEMPLATE.format(
        prompt=task_prompt, dimensions=_format_dimension_block(dims))
    dim_msg, dim_rec = _call_model(
        key, DIMENSION_SYSTEM, dim_user, reference_images, submission_images,
        folder / 'dimension', model=model, cache_key='vlm-judge-dimension')
    dim_obj = parse_json_obj(dim_msg.get('content'))
    dim_result = score_dimensions(dim_obj, dims)
    if not dim_result['ok']:
        raise RuntimeError('Dimension pass invalid: ' + dim_result['error'])
    dim_notes = (dim_obj or {}).get('overall_notes', '')

    rubric_score = rubric_summary['weighted_score_0_100']
    dimension_score = dim_result['dimension_score_0_100']
    fused = 0.5 * (rubric_score + dimension_score)

    usage = {
        'rubric': {k: rubric_rec.get(k) for k in ('input_tokens', 'cached_tokens', 'output_tokens')},
        'dimension': {k: dim_rec.get(k) for k in ('input_tokens', 'cached_tokens', 'output_tokens')},
    }

    return {
        'ok': True,
        'rubric_scores': rubric_scores,
        'rubric_score_0_100': rubric_score,
        'rubric_raw_score_0_100': rubric_summary['raw_score_0_100'],
        'rubric_notes': rubric_notes,
        'dimension_scores': dim_result['dimension_scores'],
        'dimension_score_0_100': dimension_score,
        'dimension_notes': dim_notes,
        'fused_score_0_100': fused,
        'model': model,
        'usage': usage,
    }
