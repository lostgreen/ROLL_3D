"""ROLL-compatible environment using an immutable snapshot of the real V2S harness."""
import atexit
import json
import os
from pathlib import Path
import socket
import sys
import uuid
import threading
import numpy as np
from PIL import Image

WORK=Path(__file__).resolve().parent
HARNESS=WORK/'harness'
for p in (HARNESS/'src',HARNESS/'src/harness',HARNESS/'headless',HARNESS/'metrics'):
    sys.path.insert(0,str(p))
from environment.session import RunSession
from environment.task_spec import TaskSpec
from blender_process import HeadlessBlender
from mcp_client import BlenderMCPClient
from local_vlm import parse_tool_calls
from action_response import final_action_text
from v2s_metrics.image import rgb

RENDER='''import bpy
s=bpy.context.scene
s.camera=bpy.data.objects['EvaluationCamera']
s.render.engine='CYCLES';s.cycles.device='CPU';s.cycles.samples=16
# Keep the rendered observation at the GT resolution. The frozen image metric
# intentionally rejects implicit resize, so prediction and reference must
# enter it with identical HxW.
s.render.resolution_x=512;s.render.resolution_y=512;s.render.resolution_percentage=100
s.render.image_settings.file_format='PNG';s.render.image_settings.color_mode='RGB'
s.render.filepath=OUTPUT
bpy.ops.render.render(write_still=True)
'''

class Video2SceneEnv:
    """reset/step contract; no nested LLM calls, rewards only on terminal actions."""
    def __init__(self,task_manifest=None,output_root=None,max_steps=3,**kwargs):
        self.task=json.loads(Path(task_manifest or WORK/'task/task.json').read_text())
        self.output_root=Path(output_root or WORK/'episodes');self.output_root.mkdir(parents=True,exist_ok=True)
        self.max_steps=max_steps;self.blender=None;self.mcp=None;self.session=None
        atexit.register(self.close)

    def _execute(self,code):
        result=self.mcp.call_tool_rich('execute_blender_code',{'code':code})
        if result.get('isError') or result.get('text','').startswith('Error'):
            raise RuntimeError('Blender execution failed: '+result.get('text','')[:240])
        return result

    def _render(self,label):
        path=self.episode/f'{label}.png'
        self._execute(RENDER.replace('OUTPUT',repr(str(path))))
        if not path.is_file():raise RuntimeError('render image missing')
        return path

    def _obs(self,text,image):
        return {'prompt':[{'type':'text','text':text},{'type':'image'}], 'image':[Image.open(image).convert('RGB')]}

    def reset(self,seed=None):
        self.close();self.step_count=0
        self.episode=self.output_root/f'{self.task["id"]}_{seed}_{uuid.uuid4().hex[:10]}'
        self.episode.mkdir();self.events=[]
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        self.blender=HeadlessBlender(port=port,log_path=self.episode/'blender.log',blender_bin=os.getenv('BLENDER_BIN','/home/xuboshen/zgw/tools/blender/blender')).start()
        self.mcp=BlenderMCPClient(command=sys.executable,args=['-c','from blender_mcp.server import main; main()'],cwd=str(HARNESS/'blender-mcp'),env={'BLENDER_PORT':str(port),'PYTHONPATH':str(HARNESS/'blender-mcp/src'),'PYTHONDONTWRITEBYTECODE':'1'})
        self.mcp.start()
        self._execute('import bpy\nbpy.ops.wm.open_mainfile(filepath='+repr(self.task['initial_blend'])+')')
        task={'id':self.task['id'],'allowed_tools':['execute_blender_code'],'budget':{'max_agent_steps':self.max_steps,'max_tool_calls':self.max_steps,'max_wall_seconds':600}}
        self.session=RunSession(TaskSpec.from_task(task),self.mcp,artifact_root=self.episode/'artifacts')
        image=self._render('initial')
        self.initial_score=self._score(image)
        instruction=self.task['prompt']+'\nOutput exactly one plain JSON object, without XML tags, markdown or explanations: {"name":"execute_blender_code","arguments":{"code":"..."}} or {"name":"finish","arguments":{}}. bpy is available. Only set existing Base Color inputs; do not create nodes or objects. Use at most two short assignments per action. Example syntax (grey is only an example; choose RGB from the reference): bpy.data.objects[\'Cube\'].data.materials[0].node_tree.nodes[\'Principled BSDF\'].inputs[\'Base Color\'].default_value=(0.5,0.5,0.5,1). The objects are Cube and Sphere. The first image is the reference; the second is the current scene.'
        obs={'prompt':[{'type':'text','text':instruction},{'type':'image'},{'type':'image'}], 'image':[Image.open(self.task['reference_image']).convert('RGB'),Image.open(image).convert('RGB')]}
        return obs,{'env_instruction':'You are a visual Blender editing agent. Use only the documented tool protocol.'}

    def _score(self,image):
        # Diagnostic reward on the OBSERVED image, not novel-view or benchmark quality.
        a=rgb(image);b=rgb(self.task['reference_image'])
        mse=float(np.square(a-b).mean())
        return {'mse':mse,'reward':float(np.exp(-30*mse))}

    def step(self,action):
        self.step_count+=1;done=False;error=None;log=None
        raw=action.replace('<|im_end|>','').replace('<|endoftext|>','').strip()
        try:
            calls=parse_tool_calls(final_action_text(raw),self.step_count)
            if len(calls)!=1:raise ValueError('Exactly one explicit tool call required')
            call=calls[0]
            if call.name=='finish':done=True;text='Episode finished.'
            else:
                result,log=self.session.executor.execute(call.name,call.arguments)
                text=result.text[:2500]
                if result.status not in ('succeeded','success','ok'):error=result.error_type
        except ValueError as exc:
            error='invalid_action';text=str(exc)
        truncated=self.step_count>=self.max_steps and not done
        done=done or truncated
        image=self._render(f'step_{self.step_count}')
        score=self._score(image)
        if done:self._execute('import bpy\nbpy.ops.wm.save_as_mainfile(filepath='+repr(str(self.episode/'scene.blend'))+')')
        event={'step':self.step_count,'action':raw,'tool':log,'error':error,'error_detail':text if error else None,'score':score,'terminal':done}
        self.events.append(event)
        (self.episode/'events.json').write_text(json.dumps(self.events,indent=2))
        reward=score['reward'] if done else 0.
        info={'metrics':{'quality':score['reward'],'invalid_action':float(error=='invalid_action')},'success':done and score['mse']<.005}
        if done:
            (self.episode/'result.json').write_text(json.dumps({'task_id':self.task['id'],'initial':self.initial_score,'final':score,'steps':self.step_count,'reward':reward,'truncated':truncated,'reward_version':self.task['reward_version']},indent=2))
        observation=self._obs(text+'\nCurrent rendered scene:',image)
        if done:self.close()
        return observation,reward,done,truncated,info

    def close(self):
        if self.mcp:
            try:self.mcp.close()
            finally:self.mcp=None
        if self.blender:
            self.blender.stop();self.blender=None


_REGISTERED=False
_REGISTER_LOCK=threading.Lock()

def register():
    global _REGISTERED
    import gem
    with _REGISTER_LOCK:
        if not _REGISTERED:
            gem.register('video2scene',entry_point='v2s_env:Video2SceneEnv')
            _REGISTERED=True

if __name__=='__main__':
    env=Video2SceneEnv(max_steps=2)
    try:
        obs,_=env.reset(seed=42)
        action='<tool_call>'+json.dumps({'name':'execute_blender_code','arguments':{'code':"import bpy\nfor name,color in [('Cube',(0.9,0.12,0.025,1)),('Sphere',(0.02,0.18,0.85,1))]:\n bpy.data.objects[name].data.materials[0].node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value=color"}})+'</tool_call>'
        env.step(action)
        _,reward,done,_,_=env.step('<tool_call>{"name":"finish","arguments":{}}</tool_call>')
        assert done and reward>.999 and reward>env.initial_score['reward']
        print('DONE contract_smoke '+json.dumps({'initial':env.initial_score,'terminal_reward':reward,'reference_images':len(obs['image']),'episode':str(env.episode)}),flush=True)
    finally:env.close()
