"""Scene-side consumers for artifacts created by retrieval or generation tools."""

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path

from environment.artifacts import ArtifactStore
from environment.budget import Usage
from tools.base import ToolResult, ToolSpec


class SceneProvider:
    VERSION = "scene-v3-root-transform"

    def __init__(self, store: ArtifactStore, mcp):
        self.store = store
        self.mcp = mcp

    def register(self, registry):
        registry.register(
            ToolSpec(
                "scene.import_artifact",
                "Import a mesh into Blender under one root Empty, pivot at bottom center. "
                "The returned root_name moves/scales the whole asset; preserve it as the semantic object root.",
                {
                    "type": "object",
                    "properties": {
                        "artifact_id": {"type": "string"},
                        "collection": {"type": "string"},
                    },
                    "required": ["artifact_id"],
                },
                "scene",
                self.VERSION,
                usage_upper_bound=Usage(),
            ),
            self.import_artifact,
        )
        registry.register(
            ToolSpec(
                "scene.set_transform",
                "Set specified local location, Euler XYZ rotation (radians), or scale. Omitted fields stay unchanged. "
                "Use the imported root_name for the entire asset; an unparented root uses world coordinates.",
                {
                    "type": "object",
                    "properties": {
                        "object_name": {"type": "string"},
                        "location": {"type": "array", "minItems": 3, "maxItems": 3},
                        "rotation": {"type": "array", "minItems": 3, "maxItems": 3},
                        "scale": {"type": "array", "minItems": 3, "maxItems": 3},
                    },
                    "required": ["object_name"],
                },
                "scene",
                self.VERSION,
                usage_upper_bound=Usage(),
            ),
            self.set_transform,
        )
        registry.register(
            ToolSpec(
                "scene.list_objects",
                "List geometry and Empty roots, including parent, world position and descendant world bounds.",
                {"type": "object", "properties": {}},
                "scene",
                self.VERSION,
                usage_upper_bound=Usage(),
            ),
            self.list_objects,
        )

    def _manifest(self, artifact_id):
        if not artifact_id or any(ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-" for ch in artifact_id):
            return None, ToolResult("ERROR: invalid artifact_id", status="failed", error_type="invalid_arguments")
        path = self.store.root / f"{artifact_id}.manifest.json"
        if not path.is_file():
            return None, ToolResult("ERROR: artifact not found", status="failed", error_type="artifact_not_found")
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None, ToolResult("ERROR: invalid artifact manifest", status="failed", error_type="artifact_manifest")
        if not isinstance(manifest, dict) or manifest.get("artifact_type") != "mesh":
            return None, ToolResult("ERROR: artifact is not an editable mesh", status="failed", error_type="artifact_type")
        payload = (self.store.root / str(manifest.get("path", ""))).resolve()
        if not payload.is_file() or not payload.is_relative_to(self.store.root):
            return None, ToolResult("ERROR: artifact payload unavailable", status="failed", error_type="artifact_missing")
        if str(manifest.get("format", "")).lower() not in {"glb", "gltf"}:
            return None, ToolResult("ERROR: only GLB/GLTF scene import is supported", status="failed", error_type="artifact_format")
        return (manifest, payload), None

    def import_artifact(self, args):
        artifact_id = str(args.get("artifact_id", "")).strip()
        resolved, error = self._manifest(artifact_id)
        if error:
            return error
        manifest, payload = resolved
        collection = str(args.get("collection", "")).strip()
        root_name = "asset_" + uuid.uuid4().hex
        code = f"""
import bpy, json
from mathutils import Vector
filepath = {json.dumps(str(payload))}
before = set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=filepath)
new_objects = [o for o in bpy.data.objects if o not in before]
if not new_objects: raise ValueError("artifact imported no objects")
for obj in new_objects: obj["v2s_artifact_id"] = {json.dumps(artifact_id)}
bpy.context.view_layer.update()
points = [o.matrix_world @ Vector(c) for o in new_objects if o.type == 'MESH' for c in o.bound_box]
root = bpy.data.objects.new({json.dumps(root_name)}, None)
bpy.context.scene.collection.objects.link(root)
root["v2s_artifact_id"] = {json.dumps(artifact_id)}
if points:
    lo = Vector(tuple(min(p[i] for p in points) for i in range(3)))
    hi = Vector(tuple(max(p[i] for p in points) for i in range(3)))
    root.location = ((lo.x+hi.x)/2, (lo.y+hi.y)/2, lo.z)
bpy.context.view_layer.update()
for obj in new_objects:
    if obj.parent not in new_objects:
        world = obj.matrix_world.copy()
        obj.parent = root
        obj.matrix_world = world
new_objects.append(root)
collection_name = {json.dumps(collection)}
if collection_name:
    target = bpy.data.collections.get(collection_name) or bpy.data.collections.new(collection_name)
    if target.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(target)
    for obj in new_objects:
        for owner in list(obj.users_collection):
            owner.objects.unlink(obj)
        target.objects.link(obj)
print(json.dumps({{"artifact_id": {json.dumps(artifact_id)}, "objects": [o.name for o in new_objects]}}))
"""
        try:
            result = self.mcp.call_tool_rich("execute_blender_code", {"code": code})
        except Exception as exc:
            return ToolResult(f"ERROR importing artifact: {exc}", status="failed", error_type="scene_import")
        text = result.get("text", "") if isinstance(result, dict) else str(result)
        if isinstance(result, dict) and result.get("isError"):
            return ToolResult(text, status="failed", error_type="scene_import")
        return ToolResult(json.dumps({
            "artifact_id": artifact_id,
            "format": manifest.get("format"),
            "objects": text,
            "collection": collection or None,
            "root_name": root_name,
        }, ensure_ascii=False), artifact_ids=[artifact_id], structured={"root_name": root_name, "artifact_id": artifact_id})

    @staticmethod
    def _vector(args, key, default):
        value = args.get(key, default)
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError(f"{key} must be an array of three numbers")
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
            raise ValueError(f"{key} must contain finite numbers")
        return [float(x) for x in value]

    def set_transform(self, args):
        object_name = str(args.get("object_name", "")).strip()
        if not object_name:
            return ToolResult("ERROR: object_name is required", status="failed", error_type="invalid_arguments")
        try:
            values = {key: self._vector(args, key, None)
                      for key in ("location", "rotation", "scale") if key in args}
            if not values or args.keys() - {"object_name", "location", "rotation", "scale"}:
                raise ValueError("provide at least one transform field; unknown fields are not allowed")
        except (TypeError, ValueError) as exc:
            return ToolResult(f"ERROR: {exc}", status="failed", error_type="invalid_arguments")
        code = """import bpy
obj = bpy.data.objects.get(%s)
if obj is None:
    raise ValueError("object not found")
for key, value in %r.items():
    setattr(obj, 'rotation_euler' if key == 'rotation' else key, value)
print("TRANSFORM_OK")
""" % (object_name.__repr__(), values)
        try:
            result = self.mcp.call_tool_rich("execute_blender_code", {"code": code})
        except Exception as exc:
            return ToolResult(f"ERROR setting transform: {exc}", status="failed", error_type="scene_transform")
        text = result.get("text", "") if isinstance(result, dict) else str(result)
        if isinstance(result, dict) and result.get("isError"):
            return ToolResult(text, status="failed", error_type="scene_transform")
        return ToolResult(json.dumps({"object_name": object_name, **values}, ensure_ascii=False))

    def list_objects(self, _args):
        code = """import bpy, json
from mathutils import Vector
bpy.context.view_layer.update()
items=[]
for obj in bpy.context.scene.objects:
    if obj.type not in {'MESH', 'CURVE', 'SURFACE', 'FONT', 'META', 'EMPTY'}: continue
    geometry = [o for o in [obj, *obj.children_recursive] if o.type in {'MESH','CURVE','SURFACE','FONT','META'}]
    points = [o.matrix_world @ Vector(c) for o in geometry for c in o.bound_box]
    bounds = [[min(p[i] for p in points) for i in range(3)],
              [max(p[i] for p in points) for i in range(3)]] if points else None
    items.append({'name': obj.name, 'type': obj.type, 'parent': obj.parent.name if obj.parent else None,
                  'dimensions': list(obj.dimensions), 'location': list(obj.location),
                  'world_location': list(obj.matrix_world.translation), 'world_bounds': bounds,
                  'artifact_id': obj.get('v2s_artifact_id')})
print(json.dumps({'objects': items}))
"""
        try:
            result = self.mcp.call_tool_rich("execute_blender_code", {"code": code})
        except Exception as exc:
            return ToolResult(f"ERROR listing scene objects: {exc}", status="failed", error_type="scene_list")
        text = result.get("text", "") if isinstance(result, dict) else str(result)
        if isinstance(result, dict) and result.get("isError"):
            return ToolResult(text, status="failed", error_type="scene_list")
        return ToolResult(text)
