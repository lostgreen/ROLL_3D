import copy
import json
import unittest
from presentation import sections
from routing import route_record
from validate import validate
from export import sampling

class Tokenizer:
    def __call__(self,text,**kw):return {'input_ids':text.split()}

def record(code,objects=None):
    return dict(step=3,messages=[dict(role='user',content='Current public scene state: '+json.dumps({'public_scene_state':{'objects':objects or []}}))],
        target={'tool_calls':[{'function':{'name':'execute_blender_code','arguments':{'code':code}}}]})

class FullTests(unittest.TestCase):
    def test_review_sampling_handles_exhausted_null_attempt(self):
        rows=[dict(sample_id='s:%03d'%i,scene_id='s%d'%(i%12),route_type=['A','B','C','D1','D2'][i%5],state='ok',content_attempt=1) for i in range(80)]
        failed=dict(sample_id='failed:001',scene_id='failed',route_type='B',state='failed',content_attempt=None)
        result=sampling(rows,rows+[failed])
        self.assertEqual(len(result['random_ids']),55)
        self.assertEqual(len(result['risk_ids']),5)
        self.assertIn('failed:001',result['risk_ids'])
        self.assertFalse(set(result['random_ids']) & set(result['risk_ids']))
    def test_unused_helper_not_first_build(self):
        r=record('def unused():\n bpy.ops.mesh.primitive_cube_add()\nprint(1)')
        self.assertEqual(route_record(r)['type'],'D1')
    def test_called_helper_and_probe(self):
        r=record('def box():\n bpy.ops.mesh.primitive_cube_add()\nbox()')
        self.assertEqual(route_record(r)['type'],'A')
        r=record('obj.hide_render = True',[dict(name='Cube',type='MESH')])
        self.assertEqual(route_record(r)['type'],'B')
    def test_projection_alias_readonly(self):
        r=record('from bpy_extras.object_utils import world_to_camera_view as w2c\nprint(w2c(sc,cam,p))')
        self.assertEqual(route_record(r)['type'],'D1')
    def test_presentation_roundtrip_and_late_metadata(self):
        p=dict(schema_version='v',student_tools=[{'tool':'x'}],camera_state={'camera':'unknown'},student_messages=[dict(role='system',content='same'),dict(role='user',content='state')],
            images=[dict(path='/not_read',image_id='image:0',message_index=0)],sample_id='a:001',step=1,route_type='A',current_action=[1],teacher_current_raw_cot='first')
        a,h=sections(p);other=copy.deepcopy(p);other.update(sample_id='a:002',step=2,teacher_current_raw_cot='next')
        b,j=sections(other)
        self.assertEqual(a[:-1],b[:-1]);self.assertNotEqual(h,j)
        self.assertEqual(p['images'][0]['path'],'/not_read')
        self.assertNotIn('path',a[-1]['value']['images'][0])
    def test_CJK_and_controls(self):
        p=dict(sample_id='s:001',route_type='B',images=[],evidence_catalog=[dict(id='action:current')],
            student_messages=record('')['messages'],current_action=[dict(function=dict(arguments=dict(code="obj = bpy.data.objects.get('床头柜')")))])
        def result(text):
            obj=dict(schema_version='short_cot_rewrite_v1',sample_id='s:001',status='ok',route_type='B',short_cot=text,
                audit_sidecar=dict(claims=[],dropped_ungrounded_raw_cot=[],flag_reason=None))
            return validate(obj,p,Tokenizer())['passed']
        self.assertTrue(result("Move '床头柜' slightly."));self.assertFalse(result('现在移动床头柜。'))
        self.assertFalse(result('Move `未知柜` slightly.'));self.assertFalse(result('Move\u200b the cabinet.'))

if __name__=='__main__':unittest.main()
