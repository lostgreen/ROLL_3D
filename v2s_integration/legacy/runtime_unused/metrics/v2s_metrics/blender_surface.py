"""Optional Blender surface adapter, including rendered beveled curves.

Run inside Blender. Mesh-only selection silently loses CURVE/FONT/SURFACE
geometry that appears in RGB renders. Unsupported representations are reported.
"""
import numpy as np

SURFACE_TYPES = frozenset({'MESH', 'CURVE', 'SURFACE', 'FONT', 'META'})
NON_GEOMETRY_TYPES = frozenset({'EMPTY', 'CAMERA', 'LIGHT', 'ARMATURE', 'LATTICE', 'SPEAKER', 'LIGHT_PROBE'})


def surface_objects(scene):
    visible = [o for o in scene.objects if not o.hide_render]
    objects = [o for o in visible if o.type in SURFACE_TYPES]
    unsupported = [{'name': o.name, 'type': o.type} for o in visible
                   if o.type not in SURFACE_TYPES | NON_GEOMETRY_TYPES]
    return objects, unsupported


def mesh_arrays(objects, depsgraph):
    """Triangulate evaluated geometry in world coordinates, clearing temp meshes."""
    vertices, faces = [], []
    offset = 0
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            if mesh is None:
                continue
            mesh.calc_loop_triangles()
            world = evaluated.matrix_world
            v = np.array([world @ x.co for x in mesh.vertices], dtype=float).reshape(-1, 3)
            f = np.array([list(t.vertices) for t in mesh.loop_triangles], dtype=int).reshape(-1, 3)
            if len(v) and len(f):
                vertices.append(v)
                faces.append(f + offset)
                offset += len(v)
        finally:
            evaluated.to_mesh_clear()
    return (np.concatenate(vertices) if vertices else np.empty((0, 3)),
            np.concatenate(faces) if faces else np.empty((0, 3), dtype=int))
