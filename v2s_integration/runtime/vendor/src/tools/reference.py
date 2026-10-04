"""Inspect a crop of public evidence; register exactly those pixels for generation."""

import base64
import io
import json
import math
import uuid

from PIL import Image

from environment.artifacts import ArtifactManifest
from environment.budget import Usage
from tools.base import ToolResult, ToolSpec


class ReferenceProvider:
    VERSION = "reference-crop-v1"

    def __init__(self, store, references, segmenter=None):
        self.store, self.references = store, references
        self.segmenter = segmenter

    def register(self, registry):
        registry.register(ToolSpec(
            "reference.crop", "Observe one object in a supplied reference image. Return the crop image "
            "and a new reference_image ID usable by asset.generate_3d_from_image. "
            "bbox is [left, top, right, bottom] in normalized 0..1 image coordinates. "
            "Keep the whole visible object and a small margin; this crops, not segments or completes it.",
            {"type": "object", "properties": {
                "reference_image": {"type": "string"},
                "bbox": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1},
                         "minItems": 4, "maxItems": 4}},
             "required": ["reference_image", "bbox"], "additionalProperties": False},
            "reference", self.VERSION, category="observation", side_effects=("creates_artifact",)), self.crop)
        if self.segmenter is not None:
            registry.register(ToolSpec(
                "reference.segment", "Segment one visible object selected by a normalized bbox. "
                "Returns an RGBA cutout usable for generation, then a full-image mask overlay for inspection. "
                "Select the complete object; segmentation cannot recover occluded or missing parts.",
                {"type": "object", "properties": {
                    "reference_image": {"type": "string"},
                    "bbox": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1},
                             "minItems": 4, "maxItems": 4}},
                 "required": ["reference_image", "bbox"], "additionalProperties": False},
                "reference", self.segmenter.version, category="observation",
                side_effects=("creates_artifact",)), self.segment)

    def crop(self, args):
        ref, box = args.get("reference_image"), args.get("bbox")
        if (args.keys() - {"reference_image", "bbox"} or not isinstance(ref, str)
                or ref not in self.references or not isinstance(box, list) or len(box) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in box)
                or box[0] >= box[2] or box[1] >= box[3]):
            return ToolResult("ERROR: use a registered image ID and ordered normalized bbox.",
                              usage=Usage.zero(), status="rejected", error_type="invalid_arguments")
        with Image.open(self.references[ref]) as source:
            w, h = source.size
            pixels = [math.floor(box[0]*w), math.floor(box[1]*h),
                      math.ceil(box[2]*w), math.ceil(box[3]*h)]
            crop = source.convert("RGBA").crop(pixels)
        if min(crop.size) < 8:
            return ToolResult("ERROR: crop must be at least 8 pixels on each side.",
                              usage=Usage.zero(), status="rejected", error_type="invalid_arguments")
        buffer = io.BytesIO()
        crop.save(buffer, format="PNG")
        data = buffer.getvalue()
        artifact_id = "crop_" + uuid.uuid4().hex
        target = self.store.root / (artifact_id + ".png")
        target.write_bytes(data)
        provenance = {"parent_reference": ref, "bbox": box, "bbox_pixels": pixels,
                      "source_size": [w, h], "size": list(crop.size)}
        manifest = self.store.register(ArtifactManifest(
            artifact_id, "evidence", target.name, "png", "reference.crop", self.VERSION,
            "image_pixels", "pixel", evidence_refs=(ref,), source="observed", normalization=provenance))
        self.references[artifact_id] = str(target)
        result = {"reference_image": artifact_id, "sha256": manifest["sha256"], **provenance}
        return ToolResult(json.dumps(result), images=[{"mime": "image/png", "data": base64.b64encode(data).decode()}],
                          artifact_ids=[artifact_id], structured=result)

    def segment(self, args):
        # Use the same public-reference and bbox boundary as crop; never accept paths.
        ref, box = args.get("reference_image"), args.get("bbox")
        if (args.keys() - {"reference_image", "bbox"} or not isinstance(ref, str)
                or ref not in self.references or not isinstance(box, list) or len(box) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in box)
                or box[0] >= box[2] or box[1] >= box[3]):
            return ToolResult("ERROR: use a registered image ID and ordered normalized bbox.",
                              usage=Usage.zero(), status="rejected", error_type="invalid_arguments")
        import numpy as np
        try:
            with Image.open(self.references[ref]) as image:
                source = image.convert("RGBA")
            w, h = source.size
            mask, score = self.segmenter.predict(source.convert("RGB"), [box[0]*w, box[1]*h, box[2]*w, box[3]*h])
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (h, w): raise ValueError("mask shape mismatch")
            alpha = np.asarray(source.getchannel("A"))
            mask &= alpha > 0
            if not mask.any() or not math.isfinite(float(score)): raise ValueError("empty or invalid mask")
            ys, xs = np.where(mask)
            pad = max(2, math.ceil(max(xs.max()-xs.min()+1, ys.max()-ys.min()+1)*.03))
            pixels = [max(0, int(xs.min())-pad), max(0, int(ys.min())-pad),
                      min(w, int(xs.max())+1+pad), min(h, int(ys.max())+1+pad)]
            rgba = source.copy()
            rgba.putalpha(Image.fromarray(np.where(mask, alpha, 0).astype("uint8")))
            overlay = np.asarray(source.convert("RGB")).copy()
            overlay[mask] = (.55*overlay[mask] + .45*np.array([0, 220, 120])).astype("uint8")
            artifact_id = "segment_" + uuid.uuid4().hex
            metadata = {"parent_reference": ref, "prompt_bbox": box, "bbox_pixels": pixels,
                        "source_size": [w, h], "mask_score": float(score), "foreground_pixels": int(mask.sum()),
                        "mask_is_model_prediction": True, "backend": self.segmenter.version,
                        "touches_source_border": bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())}
            images, ids = [], []
            for suffix, output in [("", rgba.crop(pixels)), ("_overlay", Image.fromarray(overlay)),
                                   ("_mask", Image.fromarray(mask.astype("uint8")*255))]:
                name = artifact_id + suffix
                target = self.store.root / (name + ".png")
                output.save(target)
                manifest = self.store.register(ArtifactManifest(name, "evidence", target.name, "png",
                    "reference.segment", self.segmenter.version, "image_pixels", "pixel",
                    evidence_refs=(ref,), source="observed" if not suffix else "generated", normalization=metadata))
                ids.append(name)
                if not suffix:
                    metadata["sha256"] = manifest["sha256"]
                    self.references[name] = str(target)
                if suffix != "_mask":
                    images.append({"mime": "image/png", "data": base64.b64encode(target.read_bytes()).decode()})
            result = {"reference_image": artifact_id, "mask_artifact_id": artifact_id+"_mask", **metadata}
            return ToolResult(json.dumps(result), images=images, artifact_ids=ids, structured=result)
        except Exception as exc:
            return ToolResult("ERROR: segmentation failed ("+type(exc).__name__+")", status="failed",
                              error_type="segmentation_failed")
