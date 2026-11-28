"""Validate the reviewed derivatives and issue new, source-bound receipts."""
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path('/data/zgw/short_cot_sft_20261124')
REPO = Path('/m2v_intern/xuboshen/zgw/ROLL')
BASE = REPO / 'artifacts/raw_cot_sft_preflight_20261121'
ANNOT = Path('/data/zgw/roll3d_short_cot_full_20261124')
EXPECTED = 'f190f4b90b64f44c6a533cb85c192d3e30e0428ec36faad1ba39fc425f21fdfa'


def read(p):
    with (gzip.open(p, 'rt') if str(p).endswith('.gz') else open(p)) as f:
        return json.load(f)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    t.replace(p)


def data_receipts(directory, rows, original_index_sha):
    directory.mkdir(parents=True, exist_ok=True)
    index = directory / 'lengths.jsonl'
    index.write_text(''.join(json.dumps(x, ensure_ascii=False, separators=(',', ':'))+'\n' for x in rows))
    policy = read(BASE / 'all_body_clean_context_v2/context_policy_manifest.json')
    shutil.copy2(BASE / 'all_body_clean_context_v2/context_profile.json', directory / 'context_profile.json')
    dump(directory / 'context_policy_manifest.json', policy)
    provenance = dict(source_reviewed_index_sha256=original_index_sha,
                      reasoning_only_replaced=True, candidate_decisions=len(rows),
                      lengths_sha256=sha(index), passed=True)
    dump(directory / 'target_cleanup_manifest.json', dict(**provenance,
         policy='drop_protocol_shaped_body_v1', records=[dict(sample_id=r['sample_id'],
         record_sha256=r['record_sha256'], body_removed=r['body_removed']) for r in rows]))
    dump(directory / 'observation_cleanup_manifest.json', dict(**provenance,
         policy_version='raw_cot_context_v2', context_profile_sha256=policy['context_profile_sha256'],
         observation_contract_sha256=policy['observation_contract_sha256'],
         records=[dict(sample_id=r['sample_id'], record_sha256=r['record_sha256'],
                       target_ids_sha256=r['target_ids_sha256']) for r in rows]))
    config = read(BASE / 'final_train_config_context_v2.json')
    config.update(lengths_file=str(index), checkpoint_interval=25,
                  annotation='v31_short_cot_reviewed_20261124', reviewed_index_sha256=original_index_sha)
    return config


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    index = ANNOT / 'reviewed/reviewed_eligible_index.jsonl'
    assert sha(index) == EXPECTED
    rows = [json.loads(x) for x in index.read_text().splitlines()]
    assert len(rows) == len({x['sample_id'] for x in rows}) == 3231
    assert len({x['scene_id'] for x in rows}) == 199
    status = ROOT / 'prepare_status.json'
    sys.path.insert(0, str(ROOT / 'code/training_source'))
    from pack_raw_cot import setup
    processor = setup(BASE / 'runtime_context_v2', 'raw_cot_context_v2')
    for i, row in enumerate(rows):
        p = Path(row['record_path'])
        assert sha(p) == row['record_sha256'], row['sample_id']
        oldpath = Path(row['source_record_path'])
        assert sha(oldpath) == row['source_record_sha256']
        old, new = read(oldpath), read(p)
        expected = copy.deepcopy(old)
        expected['target']['reasoning_content'] = new['target']['reasoning_content']
        assert expected == new, 'NON_REASONING_CHANGE:' + row['sample_id']
        validation = read(p.parent / 'packing_validation.json')
        assert validation['passed'] and validation['prefix_images_action_body_equal']
        assert validation['loss_masks_recomputed']
        assert validation['source_record_sha256'] == row['source_record_sha256']
        for k in ['total_tokens','target_tokens','prompt_tokens','prompt_ids_sha256','image_hashes']:
            assert validation['metrics'][k] == row[k], (row['sample_id'],k)
        # The annotation index inherited the original formatter's target hash.
        # Recompute the changed target suffix, without retokenizing past images.
        prompt = processor.apply_chat_template(new['messages'], tools=new['tools'], tokenize=False,
                     add_generation_prompt=True, enable_thinking=True)
        full = processor.apply_chat_template(new['messages']+[new['target']], tools=new['tools'],
                     tokenize=False, add_generation_prompt=False, enable_thinking=True)
        assert full.startswith(prompt)
        ids = processor.tokenizer(full[len(prompt):], add_special_tokens=False)['input_ids']
        assert len(ids) == row['target_tokens']
        row['source_target_ids_sha256'] = row['target_ids_sha256']
        row['target_ids_sha256'] = hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
        assert row['total_tokens'] <= 98304 and row['cot_tokens'] <= 600
        if i % 100 == 0:
            dump(status, dict(stage='validating_derivatives', finished=i, total=len(rows), at=time.time()))
    config = data_receipts(ROOT / 'train_data', rows, EXPECTED)
    dump(ROOT / 'train_config.json', config)
    longest = sorted(rows, key=lambda r:r['total_tokens'])[-2:]
    smoke = data_receipts(ROOT / 'smoke_data', longest, EXPECTED)
    smoke.update(effective_batch_size=2, epochs=1)
    dump(ROOT / 'smoke_config.json', smoke)
    from train_raw_cot import prepare
    rr, excluded, manifest = prepare(config, verify_images=True)
    assert len(rr) == 3231 and not excluded and manifest['total_updates'] == 303
    assert manifest['warmup_steps'] == 16 and manifest['tail_actual_batch'] == 31
    dump(ROOT / 'training_manifest.json', manifest)
    dump(ROOT / 'prepare_receipt.json', dict(passed=True, reviewed_index_sha256=EXPECTED,
         decisions=len(rr), scenes=199, total_updates=303, warmup_steps=16,
         max_sequence_length=manifest['max_eligible_length'], full_image_hashes_verified=True,
         source_targets_unchanged_except_reasoning=True, manifest_hash=manifest['manifest_hash'],
         smoke_ids=[r['sample_id'] for r in longest], smoke_lengths=[r['total_tokens'] for r in longest]))
    dump(status, dict(stage='complete', passed=True, at=time.time()))
    print('SHORT_COT_DATA_PREPARED 3231 decisions /303 updates', flush=True)


if __name__ == '__main__':
    main()
