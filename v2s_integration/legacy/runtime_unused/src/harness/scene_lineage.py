"""Read-only observations after calls; separate from model feedback and rendering."""
import json,time

CODE = '''import bpy,json
from mathutils import Vector
records=[]
for obj in bpy.data.objects:
    aid=obj.get("v2s_artifact_id")
    if not aid: continue
    corners=[obj.matrix_world @ Vector(x) for x in obj.bound_box]
    dims=[max(v[i] for v in corners)-min(v[i] for v in corners) for i in range(3)]
    area=0.0
    if obj.type=="MESH":
        for face in obj.data.polygons:
            verts=[obj.matrix_world @ obj.data.vertices[i].co for i in face.vertices]
            for i in range(1,len(verts)-1): area+=(verts[i]-verts[0]).cross(verts[i+1]-verts[0]).length/2
    records.append(dict(name=obj.name,artifact_id=aid,matrix_world=[list(r) for r in obj.matrix_world],dimensions=dims,
        hide_render=obj.hide_render,hide_viewport=obj.hide_viewport,parent=obj.parent.name if obj.parent else None,
        collections=[c.name for c in obj.users_collection],modifiers=[dict(name=m.name,type=m.type) for m in obj.modifiers],
        vertices=len(obj.data.vertices) if obj.type=="MESH" else 0,faces=len(obj.data.polygons) if obj.type=="MESH" else 0,surface_area=area))
print("V2S_LINEAGE="+json.dumps(records))
'''


def diff(before,after):
    changes=[];old={o['name']:o for o in before};new={o['name']:o for o in after}
    for name in old.keys()-new.keys():changes.append(dict(name=name,type='deleted_or_renamed_or_joined',evidence='derived'))
    for name,n in new.items():
        o=old.get(name)
        if o is None:changes.append(dict(name=name,type='appeared',evidence='derived'));continue
        scales=lambda obj:[sum(obj['matrix_world'][row][col]**2 for row in range(3))**0.5 for col in range(3)]
        scale_ratio=[b/a if abs(a)>1e-10 else None for a,b in zip(scales(o),scales(n))]
        if any(r is not None and abs(r-1)>1e-5 for r in scale_ratio):changes.append(dict(name=name,type='scaled',scale_ratio=scale_ratio,evidence='derived'))
        if any(abs(o['matrix_world'][i][3]-n['matrix_world'][i][3])>1e-6 for i in range(3)):changes.append(dict(name=name,type='moved',evidence='derived'))
        ratio=[b/a if abs(a)>1e-10 else None for a,b in zip(o['dimensions'],n['dimensions'])]
        if any(r is not None and abs(r-1)>1e-5 for r in ratio):changes.append(dict(name=name,type='bounds_changed',dimension_ratio=ratio,evidence='derived'))
        for field,kind in [('matrix_world','transform_changed'),('parent','reparented'),('hide_render','render_visibility_changed'),('hide_viewport','viewport_visibility_changed'),('vertices','mesh_edited'),('faces','mesh_edited'),('surface_area','surface_area_changed')]:
            if o[field]!=n[field]:changes.append(dict(name=name,type=kind,evidence='derived'))
    return changes


class LineageRecorder:
    def __init__(self,recorder): self.recorder=recorder;self.previous=[]
    def after_call(self,mcp,name,args,call_id):
        if name!='execute_blender_code' and not name.startswith('scene.'):return
        start=time.monotonic()
        try:
            text=mcp.call_tool('execute_blender_code',{'code':CODE})
            raw=str(text).split('V2S_LINEAGE=',1)[1].splitlines()[0]
            objects=json.loads(raw);changes=diff(self.previous,objects)
            code=args.get('code','')
            touched=[o['name'] for o in self.previous if o['name'] in code]
            self.recorder.event('scene.lineage',call_id=call_id,status='ok',evidence='observed',objects=objects,changes=changes,
                  static_touch={'evidence':'heuristic','names':touched,'tag_lookup':'v2s_artifact_id' in code},elapsed_sec=time.monotonic()-start)
            self.previous=objects
        except Exception as exc:
            self.recorder.event('scene.lineage',call_id=call_id,status='failed',error_type=type(exc).__name__,elapsed_sec=time.monotonic()-start)
