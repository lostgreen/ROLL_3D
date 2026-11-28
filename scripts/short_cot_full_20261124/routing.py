import ast
import json
import re

def action_intent(code):
    """Conservative reachable-function summary, never execute teacher Python."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return dict(status='needs_review', reason='syntax_error', calls=[])
    functions = {n.name:n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    calls, assignments, pending, visited = [], [], [tree], set()
    class Walk(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            pass
        visit_AsyncFunctionDef = visit_FunctionDef
        def visit_Call(self, node):
            name = ast.unparse(node.func)
            calls.append(name)
            if isinstance(node.func, ast.Name) and node.func.id in functions and node.func.id not in visited:
                visited.add(node.func.id); pending.extend(functions[node.func.id].body)
            self.generic_visit(node)
        def visit_Assign(self, node):
            assignments.extend(ast.unparse(t) for t in node.targets)
            self.generic_visit(node)
        def visit_AugAssign(self, node):
            assignments.append(ast.unparse(node.target)); self.generic_visit(node)
    while pending:
        Walk().visit(pending.pop())
    creates = any('primitive_' in c or c.endswith(('from_pydata','bmesh.ops.create_cube','bmesh.ops.create_cone')) for c in calls)
    creates |= any(c.endswith(('meshes.new','curves.new')) for c in calls)
    mutations = creates or any('.ops.' in c or c.endswith(('.new','.remove','.clear','.link','.unlink','.append','.update','.transform_apply')) for c in calls)
    mutations |= any('.' in a or '[' in a for a in assignments)
    # Query-only support is intentionally narrow; unsupported calls remain reviewable.
    allowed = ('print','len','list','tuple','dict','str','float','int','round','range','enumerate','sorted','getattr','hasattr','isinstance','min','max','abs','sum',
        'dir','set','any','all','zip','repr','type','bool','Vector','Matrix','world_to_camera_view')
    imported_queries=tuple(alias.asname or alias.name for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)
        and n.module=='bpy_extras.object_utils' for alias in n.names if alias.name=='world_to_camera_view')
    allowed+=imported_queries
    readonly_methods=('.get','.keys','.values','.items','.to_list','.copy','.startswith','.endswith','.lower','.upper',
        '.inverted','.inverted_safe','.to_3x3','.to_4x4','.to_translation','.to_euler','.to_quaternion','.to_scale',
        '.normalized','.lerp','.dot','.cross','.view_frame','.ray_cast','.evaluated_depsgraph_get','.depsgraph_get','.visible_get')
    query = not mutations and all(c in allowed or c in functions or c.endswith(readonly_methods) for c in calls)
    # __import__('mathutils').Vector is an explicit query constructor, not arbitrary dynamic execution.
    if not mutations and not query:
        try:
            imports=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='__import__']
            safe_imports=bool(imports) and all(len(n.args)==1 and isinstance(n.args[0],ast.Constant) and n.args[0].value=='mathutils' and not n.keywords for n in imports)
            query=safe_imports and all(c in allowed or c in functions or c.endswith(readonly_methods) or c in ['__import__',"__import__('mathutils').Vector"] for c in calls)
        except (ValueError,TypeError):pass
    # Reviewed local set accumulation used by scene_127:011; .add elsewhere stays unsupported.
    if not mutations and not query and 'ns.add' in calls:
        local_set=any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ns' for t in n.targets)
            and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='set' for n in ast.walk(tree))
        query=local_set and all(c in allowed or c in functions or c.endswith(readonly_methods) or c=='ns.add' for c in calls)
    return dict(status='supported' if mutations or query else 'needs_review', creates=bool(creates),
                mutates=bool(mutations), readonly=bool(query), calls=sorted(set(calls)),
                writes=sorted(set(assignments)), reachable_helpers=sorted(visited))

def state_of(messages):
    text = messages[-1]['content']
    assert isinstance(text,str) and text.startswith('Current public scene state: ')
    return json.loads(text.split(': ',1)[1])['public_scene_state']


def route_record(record):
    actions=record['target']['tool_calls'];assert len(actions)==1
    fn=actions[0]['function'];code=fn['arguments'].get('code','')
    state=state_of(record['messages'])
    geometry=[x for x in state.get('objects',[]) if x.get('type') in ['MESH','CURVE','SURFACE','META','FONT','VOLUME']]
    intent=action_intent(code) if fn['name']=='execute_blender_code' else {}
    errors=[]
    for i,m in enumerate(record['messages']):
        if m['role']!='tool':continue
        try:feedback=json.loads(m['content'])
        except (ValueError,TypeError):continue
        if feedback.get('error_type') or feedback.get('status') in ['error','failed']:
            errors.append(dict(message_index=i,error_type=feedback.get('error_type'),feedback=feedback))
    linked=None
    if errors and fn['name']=='execute_blender_code' and intent.get('mutates'):
        last=errors[-1];i=last['message_index']
        if not any(m['role']=='assistant' for m in record['messages'][i+1:]):
            symbols=re.findall(r"['\"]([A-Za-z_][A-Za-z_0-9.]{2,})['\"]",json.dumps(last['feedback']))
            hits=sorted(set(x for x in symbols if x not in ['text','status','error','failed','error_type','AttributeError','NameError','TypeError','KeyError'] and x in code))
            if hits:linked=dict(evidence_id='msg:%03d'%i,symbols=hits,error_type=last['error_type'],rule='immediately prior hard error, current mutating action names error symbol')
    if fn['name']=='finish':route='D2'
    elif fn['name'] in ['reference.crop','scene.list_objects','render_scene_view']:route='D1'
    elif intent.get('readonly'):route='D1'
    elif linked:route='C'
    elif intent.get('status')=='needs_review':route=None
    elif intent.get('creates') and not geometry:route='A'
    elif intent.get('mutates'):route='B'
    else:route=None
    return dict(type=route,status='routed' if route else 'needs_review',pre_geometry=len(geometry),intent=intent,
        prior_hard_errors=[{k:v for k,v in x.items() if k!='feedback'} for x in errors],recovery_link=linked)
