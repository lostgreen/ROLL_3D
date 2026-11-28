"""Loopback file-protocol adapter for upstream vLLM; no model patches."""
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT = Path('/data/zgw/short_cot_sft_20261124')
sys.path.insert(0, str(Path(__file__).parent))
from hf_server import Worker, Server, Handler, RequestError, inside, file_hash, write_json


class NativeWorker(Worker):
    def load(self):
        started = time.monotonic()
        try:
            os.environ.update(VLLM_WORKER_MULTIPROC_METHOD='spawn', HF_HUB_OFFLINE='1',
                              TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
            import torch
            import transformers
            import vllm
            from transformers import AutoProcessor
            from vllm import LLM
            torch.set_num_threads(2)
            profile = Path(self.config['context_profile'])
            policy = json.loads((profile.parent/'context_policy_manifest.json').read_text())
            expected = self.config.get('context_profile_sha256', policy['context_profile_sha256'])
            assert expected == policy['context_profile_sha256'], 'CONTEXT_PROFILE_MISMATCH'
            assert policy['version'] == 'raw_cot_context_v2'
            runtime = Path(self.config['runtime_root'])
            for name, digest in policy['context_sources'].items():
                assert file_hash(runtime/'v2s_integration/agent'/name) == digest, 'CONTEXT_SOURCE_MISMATCH'
            self.processor = AutoProcessor.from_pretrained(Path(self.config['serving_model']), local_files_only=True)
            self.decode = dict(temperature=.7, top_p=.9, top_k=-1, max_tokens=32768,
                               stop_token_ids=[248044,248046], skip_special_tokens=False)
            params = dict(dtype='bfloat16', tensor_parallel_size=1, gpu_memory_utilization=float(self.config.get('gpu_memory_utilization', .70)),
                          max_model_len=131072, enforce_eager=False, enable_prefix_caching=True,
                          max_num_seqs=1, max_num_batched_tokens=4096, limit_mm_per_prompt={'image':5},
                          seed=0, compilation_config=dict(mode=0,cudagraph_mode='FULL_DECODE_ONLY',
                                                         cudagraph_capture_sizes=[1]))
            self.model = LLM(model=str(Path(self.config['serving_model'])), **params)
            receipt = dict(backend='vllm_native', patches=[], vllm=vllm.__version__,
                           torch=torch.__version__, transformers=transformers.__version__,
                           model=str(Path(self.config['serving_model'])), params=params, decode=self.decode,
                           context_profile_sha256=expected, profile_file_sha256=file_hash(profile),
                           source_integrity_receipt=str(ROOT/'native_upstream_integrity.json'),
                           seconds=time.monotonic()-started)
            write_json(self.output/'load_receipt.json', receipt)
            self.update(stage='ready',ready=True,context_profile_sha256=expected,
                        adapter_sha256=self.config['adapter_sha256'],
                        backend='vllm_native',load_seconds=receipt['seconds'])
        except BaseException as exc:
            (self.output/'load_traceback.log').write_text(traceback.format_exc())
            self.update(stage='load_failed',ready=False,error_type=type(exc).__name__,message=str(exc)[:240])

    def generate(self, payload):
        if set(payload) != {'request_dir','seed'}:
            raise RequestError('request_requires_only_request_dir_and_seed')
        if not isinstance(payload['request_dir'],str) or not Path(payload['request_dir']).is_absolute():
            raise RequestError('request_dir_must_be_absolute')
        if type(payload['seed']) is not int or not 0 <= payload['seed'] < 2**32:
            raise RequestError('seed_must_be_uint32')
        request = inside(payload['request_dir'],self.request_root)
        if not request.is_dir():
            raise RequestError('request_directory_missing')
        with self.serial():
            if not self.state['ready']:
                raise RequestError('model_not_ready',503)
            if (request/'output.json').exists():
                raise RequestError('request_already_has_output',409)
            started = time.monotonic()
            self.update(stage='generating',busy=True,request_dir=str(request))
            images = []
            try:
                import torch
                from PIL import Image
                from vllm import SamplingParams
                write_json(request/'generation_status.json',dict(stage='running',backend='vllm_native',pid=os.getpid()))
                prompt_path = inside(request/'prompt.txt',request)
                paths = json.loads(inside(request/'images.json',request).read_text())
                if not isinstance(paths,list) or any(not isinstance(p,str) or not Path(p).is_absolute() for p in paths):
                    raise RequestError('images_must_be_absolute_path_list')
                receipts = []
                for name in paths:
                    path = inside(name,request)
                    with Image.open(path) as source:
                        image = source.convert('RGB')
                        image.load()
                    images.append(image)
                    receipts.append(dict(path=str(path),sha256=file_hash(path),size=list(image.size)))
                materialize_started = time.monotonic()
                prompt = prompt_path.read_text()
                encoded = self.processor(text=[prompt],images=images or None,return_tensors='pt',padding=False,truncation=False)
                ids = encoded['input_ids'].cpu()
                n = ids.shape[1]
                if ids.ndim != 2 or ids.shape[0] != 1 or not 0 < n <= 98304:
                    raise RequestError('context_budget_exceeded',422,input_tokens=n)
                torch.save(ids,request/'input_ids.pt')
                materialize_seconds = time.monotonic()-materialize_started
                del encoded
                generation_started = time.monotonic()
                result = self.model.generate([dict(prompt=prompt,multi_modal_data={'image':images})],
                    SamplingParams(**self.decode,seed=payload['seed']),use_tqdm=False)[0]
                seconds = time.monotonic()-generation_started
                assert result.prompt_token_ids == ids.flatten().tolist(), 'VLLM_EXPANDED_PREFIX_IDS_MISMATCH'
                output = result.outputs[0]
                generated = torch.tensor([list(output.token_ids)],dtype=torch.long)
                count = generated.shape[1]
                assert 0 < count <= 32768, 'GENERATED_LENGTH_INVALID'
                torch.save(generated,request/'generated_ids.pt')
                (request/'generated.txt').write_text(output.text)
                ended = int(generated[0,-1]) in {248044,248046}
                receipt = dict(passed=True,backend='vllm_native',inference_patches=[],variant=self.args.variant,
                    seed=payload['seed'],request_dir=str(request),response_path=str(request/'generated.txt'),
                    input_ids_path=str(request/'input_ids.pt'),generated_ids_path=str(request/'generated_ids.pt'),
                    input_ids_sha256=file_hash(request/'input_ids.pt'),generated_ids_sha256=file_hash(request/'generated_ids.pt'),
                    prompt_sha256=file_hash(prompt_path),images=receipts,input_tokens=n,input_ids_shape=list(ids.shape),
                    generated_tokens=count,generated_ids_shape=list(generated.shape),generation_seconds=seconds,
                    materialize_seconds=materialize_seconds,wall_seconds=time.monotonic()-started,
                    peak_allocated_bytes=None,peak_reserved_bytes=None,ended_on_eos=ended,
                    hit_output_limit=output.finish_reason=='length',finish_reason=output.finish_reason,
                    cached_tokens=getattr(result,'num_cached_tokens',None),decode=self.decode,
                    context_profile_sha256=self.state['context_profile_sha256'],adapter_sha256=self.state['adapter_sha256'])
                write_json(request/'output.json',receipt)
                write_json(request/'generation_status.json',dict(stage='complete',passed=True))
                self.update(stage='ready',busy=False,request_dir=None,completed=self.state['completed']+1,
                            latest_output=str(request/'output.json'))
                return {k:receipt[k] for k in ['passed','response_path','input_tokens','generated_tokens',
                                              'generation_seconds','wall_seconds','hit_output_limit']}
            except BaseException as exc:
                (request/'generation_traceback.log').write_text(traceback.format_exc())
                failure = dict(passed=False,error_type=type(exc).__name__,error=str(exc)[:240],wall_seconds=time.monotonic()-started)
                if isinstance(exc,RequestError):
                    failure.update(error=exc.code,**exc.details)
                write_json(request/'output.json',failure)
                write_json(request/'generation_status.json',dict(stage='failed',**failure))
                self.update(stage='ready',busy=False,request_dir=None,failed=self.state['failed']+1,latest_error=failure['error'])
                if isinstance(exc,RequestError):
                    exc.details.update(output_path=str(request/'output.json'))
                    raise
                raise RequestError('generation_failed',500,output_path=str(request/'output.json'))
            finally:
                for image in images:
                    image.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int,choices=[0,1],required=True)
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--variant',choices=['sft'],default='sft')
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    args = parser.parse_args()
    assert args.output_root.is_absolute() and 1024 <= args.port <= 65535
    worker = NativeWorker(args)
    server = Server(('127.0.0.1',args.port),Handler)
    server.worker = worker
    threading.Thread(target=worker.load,name='load-native-model',daemon=True).start()
    try:
        server.serve_forever(poll_interval=.5)
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
