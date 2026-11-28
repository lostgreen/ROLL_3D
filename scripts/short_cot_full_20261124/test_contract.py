import unittest
from validate import private_aliases,narrow_cleanup,validate

class Tokenizer:
    def __call__(self,text,**kwargs):return {'input_ids':text.split()}

def packet():return dict(sample_id='s:001',route_type='B',images=[{'label':'Reference ref:0'}],
    evidence_catalog=[{'id':x} for x in ['image:3','msg:001','custom:abc','action:current']])

class GateTests(unittest.TestCase):
    def test_all_catalog_and_namespaces(self):
        p=packet();text='image:3 msg:999 state:current action:L0002 tools:public raw_cot:current custom:abc'
        self.assertEqual(len(private_aliases(text,p)),7)
    def test_real_visible_ids_and_words_allowed(self):
        self.assertEqual(private_aliases('ref:0 PublicCamera1 state action Image3 Target_image:3suffix',packet()),[])
        self.assertEqual(private_aliases('custom:abcd msg:0012x',packet()),[])
    def test_only_claim_level_null_cleanup(self):
        x={'derivation_note':None,'audit_sidecar':{'claims':[{'derivation_note':None},{'derivation_note':'real content'},{'derivation':{'derivation_note':None}}]}}
        y,c=narrow_cleanup(x);self.assertEqual(len(c),1);self.assertIn('derivation_note',x['audit_sidecar']['claims'][0])
        self.assertIn('derivation_note',y);self.assertIn('derivation_note',y['audit_sidecar']['claims'][1]);self.assertIn('derivation_note',y['audit_sidecar']['claims'][2]['derivation'])
    def test_flag_does_not_need_student_body(self):
        x={'schema_version':'short_cot_rewrite_v1','sample_id':'s:001','route_type':'B','status':'needs_human_review',
            'short_cot':None,'audit_sidecar':{'claims':[],'dropped_ungrounded_raw_cot':[],'flag_reason':'Severe source defect'}}
        r=validate(x,packet(),Tokenizer());self.assertTrue(r['schema_valid_review_flag']);self.assertFalse(r['passed'])
    def test_sidecar_aliases_allowed_body_aliases_fail(self):
        x={'schema_version':'short_cot_rewrite_v1','sample_id':'s:001','route_type':'B','status':'ok',
           'short_cot':'Inspect the view.','audit_sidecar':{'claims':[{'span':'Inspect the view.','epistemic_type':'planned','evidence_ids':['image:3'],'derivation':None}], 'dropped_ungrounded_raw_cot':[],'flag_reason':None}}
        self.assertTrue(validate(x,packet(),Tokenizer())['passed'])
        x['short_cot']+=' image:3';self.assertFalse(validate(x,packet(),Tokenizer())['passed'])

if __name__=='__main__':unittest.main()
