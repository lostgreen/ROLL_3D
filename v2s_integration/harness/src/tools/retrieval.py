"""Provider-neutral local asset retrieval tools for adaptive runs.

The initial provider consumes the normalized SceneActBench asset index. It is
deliberately limited to search, inspect, and import; scene placement remains a
separate Blender operation so retrieval does not silently mutate the scene.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
from pathlib import Path

from environment.artifacts import ArtifactManifest
from environment.budget import Usage
from tools.base import ToolResult, ToolSpec


class RetrievalProvider:
    VERSION = "phase0-v3-preview"
    MAX_PREVIEWS = 4
    MAX_PREVIEW_BYTES = 2 * 1024 * 1024
    MAX_PREVIEW_PIXELS = 4_000_000

    def __init__(self, index_path, artifact_store, default_license=None):
        self.index_path = Path(index_path)
        self.store = artifact_store
        payload = self._load_index()
        self.index_metadata = payload if isinstance(payload, dict) else {}
        raw_items = payload.get("assets", payload.get("items", [])) if isinstance(payload, dict) else payload
        self.items = [item for item in raw_items if isinstance(item, dict)]
        self.default_license = self._resolve_default_license(default_license)

    def _load_index(self):
        if not self.index_path.is_file():
            return []
        with self.index_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, (dict, list)):
            raise ValueError("asset index must be a JSON object or array")
        return payload

    def _resolve_default_license(self, explicit):
        if explicit:
            return dict(explicit)
        pack_id = str(self.index_metadata.get("pack_id", "")).lower()
        if pack_id.startswith("quaternius"):
            return {
                "name": "Creative Commons Zero v1.0 Universal",
                "spdx": "CC0-1.0",
                "source": "Quaternius asset pack manifest",
                "verified": True,
            }
        return {
            "name": "unverified",
            "source": "asset index does not contain license metadata",
            "verified": False,
        }

    def register(self, registry):
        registry.register(
            ToolSpec(
                "asset.search",
                "Search the local asset collection without importing or changing the scene.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    },
                    "required": ["query"],
                },
                "retrieval",
                self.VERSION,
                usage_upper_bound=Usage.zero(),
            ),
            self.search,
        )
        registry.register(
            ToolSpec(
                "asset.inspect",
                "Inspect candidate geometry using labeled preview images and dimensions. "
                "Check preview_status: unavailable previews are insufficient visual evidence, "
                "not evidence that the asset matches or fails to match. Does not change the scene.",
                {
                    "type": "object",
                    "properties": {"asset_id": {"type": "string"}},
                    "required": ["asset_id"],
                },
                "retrieval",
                self.VERSION,
                usage_upper_bound=Usage.zero(),
            ),
            self.inspect,
        )
        registry.register(
            ToolSpec(
                "asset.import",
                "Copy a licensed candidate into the run artifact store as an editable mesh artifact.",
                {
                    "type": "object",
                    "properties": {
                        "asset_id": {"type": "string"},
                        "normalization": {"type": "object"},
                    },
                    "required": ["asset_id"],
                },
                "retrieval",
                self.VERSION,
                usage_upper_bound=Usage(imported_assets=1),
            ),
            self.import_asset,
        )

    @staticmethod
    def _asset_id(item):
        return str(item.get("asset_id", item.get("id", "")))

    def _resolve_path(self, value):
        path = Path(str(value))
        return path if path.is_absolute() else self.index_path.parent / path

    def _source_path(self, item):
        for key in ("glb_path", "path", "file"):
            value = item.get(key)
            if value:
                return self._resolve_path(value)
        return None

    @staticmethod
    def _display_name(item):
        explicit = item.get("name") or item.get("display_name")
        if explicit:
            return str(explicit)
        relative = item.get("source_relative_path")
        if relative:
            return Path(str(relative)).stem
        return RetrievalProvider._asset_id(item)

    def _find(self, asset_id):
        return next((item for item in self.items if self._asset_id(item) == asset_id), None)

    def _license(self, item):
        value = item.get("license")
        if isinstance(value, dict) and value.get("name"):
            return dict(value)
        if isinstance(value, str) and value:
            return {"name": value, "verified": False, "source": "asset index entry"}
        return dict(self.default_license)

    def _metadata(self, item):
        value = item.get("metadata")
        if isinstance(value, dict):
            return value
        metadata_path = item.get("metadata_path")
        if not metadata_path:
            return {}
        path = self._resolve_path(metadata_path)
        if not path.is_file():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    def _public(self, item):
        source = self._source_path(item)
        metadata = self._metadata(item)
        dimensions = item.get("dimensions", metadata.get("dimensions"))
        clips = item.get("animation_clips", metadata.get("animation_clips", []))
        return {
            "asset_id": self._asset_id(item),
            "name": self._display_name(item),
            "source_relative_path": item.get("source_relative_path", metadata.get("source_relative_path")),
            "format": source.suffix.lstrip(".").lower() if source else None,
            "payload_available": bool(source and source.is_file()),
            "metadata_available": bool(metadata),
            "dimensions": dimensions,
            "animation_clips": clips,
            "license": self._license(item),
        }

    def search(self, args):
        query = str(args.get("query", "")).strip().lower()
        if not query:
            return ToolResult("ERROR: query is required", status="failed", error_type="invalid_arguments")
        try:
            limit = max(1, min(50, int(args.get("limit", 10))))
        except (TypeError, ValueError):
            return ToolResult("ERROR: limit must be an integer", status="failed", error_type="invalid_arguments")
        scored = []
        for item in self.items:
            public = self._public(item)
            fields = [str(public.get(key, "")).lower()
                      for key in ("asset_id", "name", "source_relative_path", "dimensions", "animation_clips")]
            haystack = " ".join(fields)
            # Match token overlap for natural-language queries. Requiring the
            # whole query as a contiguous substring makes useful requests such
            # as "platformer crate" return an empty list even when both tokens
            # are present in the asset metadata.
            tokens = [token for token in re.split(r"[^a-z0-9]+", query) if token]
            token_hits = sum(token in haystack for token in tokens)
            if not token_hits:
                continue
            # Stable, provider-neutral ranking: exact ID/name, token coverage,
            # then original index as a deterministic tie-breaker.
            exact = int(query in fields[0] or query == fields[1])
            phrase = int(query in haystack)
            scored.append((exact, phrase, token_hits / len(tokens), token_hits,
                           -len(haystack), -self.items.index(item), public))
        scored.sort(key=lambda row: row[:-1], reverse=True)
        hits = [row[-1] for row in scored[:limit]]
        return ToolResult(json.dumps({"assets": hits}, ensure_ascii=False))

    def inspect(self, args):
        asset_id = str(args.get("asset_id", ""))
        item = self._find(asset_id)
        if not item:
            return ToolResult("ERROR: asset not found", status="failed", error_type="asset_not_found")
        public = self._public(item)
        metadata = self._metadata(item)
        preview = item.get("preview", metadata.get("preview", {}))
        preview = preview if isinstance(preview, dict) else {}
        images, views, errors = [], [], []
        records = preview.get("views", [])
        source = self._source_path(item)
        valid_source = False
        if not source or not source.is_file():
            errors.append("asset_payload_missing")
        elif not preview.get("source_sha256"):
            errors.append("preview_not_prepared")
        else:
            try:
                valid_source = self._file_sha256(source) == preview["source_sha256"]
            except OSError:
                errors.append("asset_payload_unreadable")
            if not valid_source:
                errors.append("preview_source_mismatch")
                # A precomputed size tied to another payload is not trustworthy.
                public["dimensions"] = None
        if preview.get("preparation_error"):
            errors.append("preview_preparation_failed: " + str(preview["preparation_error"])[:160])
        if valid_source:
            if not isinstance(records, list) or not records:
                records = []
                errors.append("preview_views_missing")
            for record in records[:self.MAX_PREVIEWS]:
                try:
                    from PIL import Image
                    if not isinstance(record, dict):
                        raise ValueError("invalid_preview_record")
                    path = self._resolve_path(record["path"])
                    if path.stat().st_size > self.MAX_PREVIEW_BYTES:
                        raise ValueError("preview_too_large")
                    data = path.read_bytes()
                    digest = hashlib.sha256(data).hexdigest()
                    if not record.get("sha256") or digest != record["sha256"]:
                        raise ValueError("preview_hash_mismatch")
                    try:
                        image = Image.open(path)
                    except Image.DecompressionBombError as exc:
                        raise ValueError("preview_pixel_limit_exceeded") from exc
                    with image:
                        width, height = image.size
                        fmt = image.format
                        if fmt not in ("PNG", "JPEG") or width * height > self.MAX_PREVIEW_PIXELS:
                            raise ValueError("preview_format_or_size_invalid")
                        image.verify()
                    label = f"{asset_id}: {str(record.get('label', 'view'))[:80]}"
                    mime = "image/png" if fmt == "PNG" else "image/jpeg"
                    images.append({"data": base64.b64encode(data).decode("ascii"),
                                   "mime": mime, "label": label})
                    views.append({"label": label, "sha256": digest,
                                  "width": width, "height": height})
                except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
                    errors.append(type(exc).__name__ + ": " + str(exc)[:160])
        state = "available" if images and not errors else "partial" if images else "unavailable"
        payload = {"asset_id": asset_id, "preview_status": state,
                   "visual_evidence": "provided" if images else "insufficient",
                   "preview_errors": errors, "preview_views": views,
                   "preview_style": preview.get("style"),
                   "preview_note": "Views are individually framed; compare dimensions separately. "
                                   "Only judge attributes visible in the supplied images.",
                   **public,
                   "dimensions_info": metadata.get("dimensions_info", item.get("dimensions_info"))}
        # Metadata lookup can succeed while visual evidence is unavailable. Keep these separate.
        return ToolResult(json.dumps(payload, ensure_ascii=False), images=images)

    @staticmethod
    def _file_sha256(path):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _safe_artifact_id(asset_id):
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", asset_id).strip("_")
        if not safe:
            safe = hashlib.sha256(asset_id.encode("utf-8")).hexdigest()[:12]
        return f"retrieved_{safe}"

    def import_asset(self, args):
        asset_id = str(args.get("asset_id", ""))
        item = self._find(asset_id)
        if not item:
            return ToolResult("ERROR: asset not found", status="failed", error_type="asset_not_found")
        source = self._source_path(item)
        if not source or not source.is_file():
            return ToolResult("ERROR: asset payload unavailable", status="failed", error_type="asset_missing")
        if source.suffix.lower() not in {".glb", ".gltf", ".obj", ".ply"}:
            return ToolResult("ERROR: unsupported mesh payload format", status="failed", error_type="asset_format")

        normalization = args.get("normalization") or {}
        if not isinstance(normalization, dict):
            return ToolResult("ERROR: normalization must be an object", status="failed", error_type="invalid_arguments")
        coordinate_frame = str(normalization.get("coordinate_frame", "right_handed_y_up"))
        units = str(normalization.get("units", "meter"))
        normalized_meta = {
            "coordinate_frame": coordinate_frame,
            "units": units,
            "scale": normalization.get("scale", 1.0),
            "orientation": normalization.get("orientation", "preserve"),
        }
        artifact_id = self._safe_artifact_id(asset_id)
        destination = self.store.root / f"{artifact_id}{source.suffix.lower()}"
        manifest_path = self.store.root / f"{artifact_id}.manifest.json"
        if not destination.exists():
            shutil.copyfile(source, destination)
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
            return ToolResult(
                json.dumps({"artifact_id": artifact_id, "sha256": record["sha256"],
                            "normalization": record.get("normalization", {}), "reused": True}),
                artifact_ids=[artifact_id],
                usage=Usage.zero(),
            )

        metadata = self._metadata(item)
        manifest = ArtifactManifest(
            artifact_id=artifact_id,
            artifact_type="mesh",
            path=destination.name,
            format=source.suffix.lstrip(".").lower(),
            tool_name="asset.import",
            tool_version=self.VERSION,
            coordinate_frame=coordinate_frame,
            units=units,
            license=self._license(item),
            source="retrieval",
            editable=True,
            renderable=True,
            normalization=normalized_meta,
        )
        record = self.store.register(manifest)
        response = {
            "artifact_id": artifact_id,
            "asset_id": asset_id,
            "sha256": record["sha256"],
            "format": manifest.format,
            "dimensions": item.get("dimensions", metadata.get("dimensions")),
            "license": manifest.license,
            "normalization": normalized_meta,
            "reused": False,
        }
        return ToolResult(json.dumps(response, ensure_ascii=False), artifact_ids=[artifact_id],
                          usage=Usage(imported_assets=1))
