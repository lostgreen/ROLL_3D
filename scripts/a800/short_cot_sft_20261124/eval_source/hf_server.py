"""Loopback-only, single-model HF generation worker for the frozen v2 comparison.

The HTTP boundary carries paths and compact receipts, never model inputs or text
responses. One FIFO gate serializes processor/seed/generation for all episodes.
"""
import argparse
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import uuid


REPO = Path('/m2v_intern/xuboshen/zgw/ROLL')
ROOT = REPO / 'artifacts/raw_cot_sft_preflight_20261121'
MODEL = Path('/m2v_intern/xuboshen/models/Qwen3.5-9B')
ADAPTER = ROOT / 'runs/raw_cot_r64_context_v2_3ep_resume103_001/checkpoint_step_00000309/adapter'
EXPECTED_ADAPTER_HASH = '9a5fea2ba8dbff1b281f4ee3a0b4c804025fdde01613f42dc1d7823ae37fb857'
INPUT_LIMIT, OUTPUT_LIMIT, CONTEXT_LIMIT = 98304, 32768, 131072
DECODE = {'do_sample': True, 'temperature': .7, 'top_p': .9, 'top_k': 0,
          'max_new_tokens': OUTPUT_LIMIT, 'use_cache': True, 'logits_to_keep': 1,
          'return_dict_in_generate': False, 'output_scores': False}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise RequestError('path_outside_request_root', 400)
    return path


class RequestError(ValueError):
    def __init__(self, code, status=400, **details):
        super().__init__(code)
        self.code, self.status, self.details = code, status, details


class Worker:
    def __init__(self, args):
        self.args = args
        self.config = json.loads(args.config.read_text())
        self.output = args.output_root.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.local = Path('/tmp') / ('agentic3d_hf_' + args.variant + '_' + str(args.port))
        self.local.mkdir(parents=True, exist_ok=True)
        request_root = Path(self.config.get('request_root', '/tmp/agentic3d_eval30_20261123/runs'))
        if not request_root.is_absolute():
            raise ValueError('request_root must be absolute')
        self.request_root = request_root.resolve()
        self._state_lock = threading.RLock()
        self._gate = threading.Condition()
        self._issued = self._serving = 0
        self._queued = 0
        self.state = {'stage': 'loading', 'ready': False, 'busy': False, 'pid': os.getpid(),
                      'gpu': args.gpu, 'variant': args.variant, 'port': args.port,
                      'completed': 0, 'failed': 0, 'output_root': str(self.output)}
        self.model = self.processor = None
        self.update()

    def update(self, **fields):
        with self._state_lock:
            self.state.update(fields, updated_at=time.time())
            snapshot = dict(self.state)
            write_json(self.output / 'status.json', snapshot)
            write_json(self.local / 'status.json', snapshot)

    def snapshot(self):
        with self._state_lock:
            return dict(self.state, queued=self._queued)

    @contextmanager
    def serial(self):
        with self._gate:
            ticket = self._issued
            self._issued += 1
            self._queued += 1
            while ticket != self._serving:
                self._gate.wait()
            self._queued -= 1
        try:
            yield
        finally:
            with self._gate:
                self._serving += 1
                self._gate.notify_all()

    def load(self):
        started = time.monotonic()
        try:
            repo = Path(self.config.get('repo', REPO)).resolve()
            root = repo / 'artifacts/raw_cot_sft_preflight_20261121'
            runtime = Path(self.config.get('runtime_root', root / 'runtime_context_v2')).resolve()
            profile_path = Path(self.config.get('context_profile', root / 'all_body_clean_context_v2/context_profile.json')).resolve()
            model_path = Path(self.config.get('model', MODEL)).resolve()
            source = root / 'source_context_v2'
            sys.path.insert(0, str(source))
            os.environ.update(CUDA_VISIBLE_DEVICES=str(self.args.gpu), RAW_COT_REQUIRE_FAST_KERNELS='1',
                              RAW_COT_CONTEXT_RUNTIME=str(runtime), HF_HUB_OFFLINE='1',
                              TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                              TRITON_CACHE_DIR=str(self.local / 'triton_cache'))
            from gpu_probe import bootstrap
            bootstrap(repo)
            # Bootstrap must not redirect this worker to a legacy packer.
            sys.path.insert(0, str(source))
            import torch
            import transformers
            from transformers import AutoModelForImageTextToText, GenerationConfig
            from pack_raw_cot import setup
            self.processor = setup(runtime, expected_version='raw_cot_context_v2')
            from v2s_integration.agent.context_policy import load_context_profile, context_profile_sha256
            profile = load_context_profile(profile_path)
            if profile['policy_version'] != 'raw_cot_context_v2':
                raise ValueError('Expected frozen context v2 profile')
            policy_path = profile_path.parent / 'context_policy_manifest.json'
            policy = json.loads(policy_path.read_text())
            if context_profile_sha256(profile) != policy['context_profile_sha256']:
                raise ValueError('Frozen profile hash mismatch')
            if self.config.get('context_profile_sha256', policy['context_profile_sha256']) != policy['context_profile_sha256']:
                raise ValueError('Configured profile hash mismatch')
            if policy['version'] != 'raw_cot_context_v2':
                raise ValueError('Expected context v2 source manifest')
            for name, expected in policy['context_sources'].items():
                if file_hash(runtime / 'v2s_integration/agent' / name) != expected:
                    raise ValueError('Frozen context source mismatch: ' + name)
            import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
            if not qwen.is_fast_path_available:
                raise RuntimeError('Qwen3.5 fast linear-attention kernels are unavailable')
            # Validate HF sampling values before allocating the model.
            GenerationConfig(do_sample=True, temperature=.7, top_p=.9, top_k=0).validate()
            torch.set_num_threads(2)
            torch.cuda.set_device(0)
            if torch.cuda.device_count() != 1:
                raise RuntimeError('Each HF server must see exactly one CUDA device')
            self.update(stage='loading_model', device_name=torch.cuda.get_device_name(0))
            self.model = AutoModelForImageTextToText.from_pretrained(
                str(model_path), dtype=torch.bfloat16, attn_implementation='sdpa',
                device_map={'': 'cuda:0'}, local_files_only=True)
            adapter_receipt = {'adapter': None, 'adapter_loaded': False, 'merged': False}
            if self.args.variant == 'sft':
                from peft import PeftModel
                adapter = Path(self.config.get('adapter', ADAPTER)).resolve()
                saved_config = json.loads((adapter / 'adapter_config.json').read_text())
                if saved_config.get('r') != 64 or saved_config.get('lora_alpha') != 128:
                    raise ValueError('Unexpected adapter rank/alpha')
                self.model = PeftModel.from_pretrained(self.model, str(adapter), is_trainable=False)
                digest, tensors = hashlib.sha256(), 0
                for name, parameter in sorted(self.model.named_parameters()):
                    if 'lora_' in name:
                        digest.update(name.encode())
                        digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
                        tensors += 1
                actual = digest.hexdigest()
                if tensors != 496 or actual != EXPECTED_ADAPTER_HASH:
                    raise RuntimeError('Final adapter tensor count/actual float32 LoRA hash mismatch')
                adapter_receipt = {'adapter': str(adapter), 'adapter_loaded': True, 'merged': True,
                                   'adapter_sha256': actual, 'adapter_tensors': tensors,
                                   'adapter_config_sha256': file_hash(adapter / 'adapter_config.json'),
                                   'adapter_file_sha256': file_hash(adapter / 'adapter_model.safetensors'),
                                   'hash_method': 'sorted actual loaded named LoRA parameters + contiguous float32 bytes'}
                self.model = self.model.merge_and_unload(safe_merge=True)
                if any('lora_' in name for name, _ in self.model.named_parameters()):
                    raise RuntimeError('Adapter merge left active LoRA parameters')
            self.model.eval()
            self.model.config.use_cache = True
            # A base config's EOS can differ from the chat tokenizer's EOS when
            # generation_config.json is absent. Native turns end at im_end.
            tokenizer = self.processor.tokenizer
            special_ids = {token: tokenizer.convert_tokens_to_ids(token)
                           for token in ('<|endoftext|>', '<|im_end|>')}
            for token, ident in special_ids.items():
                if tokenizer.convert_ids_to_tokens(ident) != token:
                    raise ValueError('Missing native stop token: ' + token)
            original_eos = self.model.generation_config.eos_token_id
            stop_ids = sorted(set(special_ids.values()) | {tokenizer.eos_token_id})
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else special_ids['<|im_end|>']
            self.model.generation_config.eos_token_id = stop_ids
            self.model.generation_config.pad_token_id = pad_id
            self.decode = dict(DECODE, eos_token_id=stop_ids, pad_token_id=pad_id)
            torch.cuda.synchronize()
            versions = {}
            for name in ['torch', 'transformers', 'fla', 'causal_conv1d', 'peft']:
                module = importlib.import_module(name)
                versions[name] = {'version': getattr(module, '__version__', None), 'file': getattr(module, '__file__', None)}
            receipt = {'passed': True, 'variant': self.args.variant, 'model': str(model_path),
                       'model_config_sha256': file_hash(model_path / 'config.json'),
                       'profile': str(profile_path), 'context_profile_sha256': context_profile_sha256(profile),
                       'runtime_root': str(runtime), 'context_sources': policy['context_sources'],
                       'qwen_module_sha256': file_hash(qwen.__file__), 'versions': versions,
                       'fast_path_available': True, 'dtype': 'bfloat16', 'attention': 'sdpa',
                       'use_cache': True, 'decode': self.decode,
                       'stop_contract': {'original_model_eos': original_eos,
                                         'tokenizer_eos': tokenizer.eos_token_id,
                                         'special_ids': special_ids, 'effective_eos': stop_ids},
                       'model_generation_config': self.model.generation_config.to_dict(),
                       'limits': {'input': INPUT_LIMIT, 'output': OUTPUT_LIMIT, 'total': CONTEXT_LIMIT},
                       'load_seconds': time.monotonic() - started, 'script_sha256': file_hash(__file__),
                       'adapter_receipt': adapter_receipt,
                       'vision_rotary_buffers': {name: str(buffer.dtype) for name, buffer in self.model.named_buffers()
                                                if 'visual' in name and ('rotary' in name or 'inv_freq' in name)}}
            write_json(self.output / 'load_receipt.json', receipt)
            self.update(stage='ready', ready=True, load_seconds=receipt['load_seconds'],
                        context_profile_sha256=receipt['context_profile_sha256'],
                        adapter_sha256=adapter_receipt.get('adapter_sha256'))
        except BaseException as error:
            (self.local / 'load_traceback.log').write_text(traceback.format_exc())
            write_json(self.output / 'load_error.json', {'error_type': type(error).__name__,
                       'message': str(error)[:300], 'traceback_path': str(self.local / 'load_traceback.log')})
            self.update(stage='failed', ready=False, busy=False, error_type=type(error).__name__,
                        message=str(error)[:240], error_path=str(self.output / 'load_error.json'))

    def generate(self, payload):
        if set(payload) != {'request_dir', 'seed'}:
            raise RequestError('request_requires_only_request_dir_and_seed')
        if not isinstance(payload['request_dir'], str) or not Path(payload['request_dir']).is_absolute():
            raise RequestError('request_dir_must_be_absolute')
        if type(payload['seed']) is not int or not 0 <= payload['seed'] < 2 ** 32:
            raise RequestError('seed_must_be_uint32')
        request = inside(payload['request_dir'], self.request_root)
        if not request.is_dir():
            raise RequestError('request_directory_missing')
        with self.serial():
            if not self.state['ready']:
                raise RequestError('model_not_ready', 503, stage=self.state['stage'])
            if (request / 'output.json').exists():
                raise RequestError('request_already_has_output', 409)
            started = time.monotonic()
            self.update(busy=True, stage='generating', request_dir=str(request))
            images = []
            try:
                write_json(request / 'generation_status.json', {'stage': 'running', 'variant': self.args.variant, 'pid': os.getpid()})
                import torch
                from PIL import Image
                from transformers import set_seed
                prompt_path = inside(request / 'prompt.txt', request)
                images_path = inside(request / 'images.json', request)
                prompt = prompt_path.read_text()
                paths = json.loads(images_path.read_text())
                if not isinstance(paths, list) or any(not isinstance(x, str) or not Path(x).is_absolute() for x in paths):
                    raise RequestError('images_must_be_absolute_path_list')
                image_receipts = []
                for name in paths:
                    path = inside(name, request)
                    with Image.open(path) as source:
                        image = source.convert('RGB')
                        image.load()
                    images.append(image)
                    image_receipts.append({'path': str(path), 'sha256': file_hash(path), 'size': list(image.size)})
                torch.cuda.reset_peak_memory_stats()
                materialize_started = time.monotonic()
                encoded = self.processor(text=[prompt], images=images or None,
                                         return_tensors='pt', padding=False, truncation=False)
                ids = encoded['input_ids']
                if ids.ndim != 2 or ids.shape[0] != 1:
                    raise RequestError('one_generation_per_request_required')
                n = int(ids.shape[1])
                if not n or n > INPUT_LIMIT or n + OUTPUT_LIMIT > CONTEXT_LIMIT:
                    raise RequestError('context_budget_exceeded', 422, input_tokens=n,
                                       input_limit=INPUT_LIMIT, reserved_output_tokens=OUTPUT_LIMIT, total_limit=CONTEXT_LIMIT)
                torch.save(ids.cpu(), request / 'input_ids.pt')
                model_inputs = {key: value.to('cuda:0') if torch.is_tensor(value) else value
                                for key, value in encoded.items()}
                materialize_seconds = time.monotonic() - materialize_started
                set_seed(payload['seed'])
                generation_started = time.monotonic()
                worker_variant = self.args.variant
                class Progress:
                    def __init__(self):
                        self.first, self.count, self.last_saved = True, 0, 0
                    def put(self, value):
                        if self.first:
                            self.first = False
                            return
                        self.count += value.numel()
                        if self.count - self.last_saved >= 1024:
                            self.save('decoding')
                    def save(self, stage):
                        self.last_saved = self.count
                        write_json(request / 'generation_status.json',
                                   {'stage': stage, 'variant': worker_variant,
                                    'input_tokens': n, 'generated_tokens_so_far': self.count,
                                    'elapsed_seconds': time.monotonic() - generation_started})
                    def end(self):
                        self.save('decoded')
                progress = Progress()
                with torch.inference_mode():
                    sequences = self.model.generate(**model_inputs, **self.decode, streamer=progress)
                torch.cuda.synchronize()
                generation_seconds = time.monotonic() - generation_started
                generated = sequences[:, n:].detach().cpu()
                raw = self.processor.tokenizer.decode(generated[0], skip_special_tokens=False)
                torch.save(generated, request / 'generated_ids.pt')
                (request / 'generated.txt').write_text(raw)
                count = int(generated.shape[1])
                if not 0 < count <= OUTPUT_LIMIT:
                    raise RuntimeError('Model returned an empty or excessive generation')
                eos = self.model.generation_config.eos_token_id
                eos = set(eos if isinstance(eos, list) else [eos])
                ended_eos = count > 0 and int(generated[0, -1]) in eos
                result = {'passed': True, 'variant': self.args.variant, 'seed': payload['seed'],
                          'request_dir': str(request), 'response_path': str(request / 'generated.txt'),
                          'input_ids_path': str(request / 'input_ids.pt'), 'generated_ids_path': str(request / 'generated_ids.pt'),
                          'input_ids_sha256': file_hash(request / 'input_ids.pt'),
                          'generated_ids_sha256': file_hash(request / 'generated_ids.pt'),
                          'prompt_sha256': file_hash(prompt_path), 'images': image_receipts,
                          'input_tokens': n, 'input_ids_shape': list(ids.shape),
                          'generated_tokens': count, 'generated_ids_shape': list(generated.shape),
                          'generation_seconds': generation_seconds, 'materialize_seconds': materialize_seconds,
                          'wall_seconds': time.monotonic() - started,
                          'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                          'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                          'ended_on_eos': ended_eos, 'hit_output_limit': count == OUTPUT_LIMIT and not ended_eos,
                          'decode': self.decode, 'context_profile_sha256': self.state['context_profile_sha256'],
                          'adapter_sha256': self.state.get('adapter_sha256')}
                write_json(request / 'output.json', result)
                write_json(request / 'generation_status.json', {'stage': 'complete', 'passed': True,
                           'output_path': str(request / 'output.json')})
                self.update(stage='ready', busy=False, completed=self.state['completed'] + 1,
                            request_dir=None, latest_output=str(request / 'output.json'))
                del model_inputs, sequences, encoded
                return {'output_path': str(request / 'output.json'), 'response_path': result['response_path'],
                        **{key: result[key] for key in ['passed', 'input_tokens', 'generated_tokens', 'generation_seconds', 'wall_seconds',
                                                        'peak_allocated_bytes', 'peak_reserved_bytes', 'hit_output_limit']}}
            except BaseException as error:
                code = error.code if isinstance(error, RequestError) else type(error).__name__
                (request / 'generation_traceback.log').write_text(traceback.format_exc())
                failure = {'passed': False, 'variant': self.args.variant, 'error_type': type(error).__name__,
                           'error': code, 'message': str(error)[:300], 'wall_seconds': time.monotonic() - started,
                           'traceback_path': str(request / 'generation_traceback.log'),
                           **(error.details if isinstance(error, RequestError) else {})}
                write_json(request / 'output.json', failure)
                write_json(request / 'generation_status.json', {'stage': 'failed', **failure})
                self.update(stage='ready', busy=False, failed=self.state['failed'] + 1, request_dir=None,
                            latest_error=code, latest_output=str(request / 'output.json'))
                if isinstance(error, RequestError):
                    error.details.update(output_path=str(request / 'output.json'))
                    raise
                raise RequestError('generation_failed', 500, output_path=str(request / 'output.json'), error_type=type(error).__name__)
            finally:
                for image in images:
                    image.close()


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    server_version = 'Agentic3DHF/1'

    def log_message(self, *_):
        pass

    def respond(self, status, value):
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        try:
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Generation receipts already exist if the caller disconnected.

    def do_GET(self):
        if self.path.rstrip('/') not in ['/health', '/status']:
            return self.respond(404, {'error': 'not_found'})
        state = self.server.worker.snapshot()
        self.respond(200, state)

    def do_POST(self):
        if self.path.rstrip('/') != '/generate':
            return self.respond(404, {'error': 'not_found'})
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 16384:
                raise RequestError('invalid_request_size', 413)
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise RequestError('request_must_be_object')
            self.respond(200, self.server.worker.generate(payload))
        except RequestError as error:
            self.respond(error.status, {'passed': False, 'error': error.code, **error.details})
        except (ValueError, UnicodeError):
            self.respond(400, {'passed': False, 'error': 'invalid_json'})
        except Exception as error:
            self.respond(500, {'passed': False, 'error': type(error).__name__})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, choices=[0, 1], required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--variant', choices=['base', 'sft'], required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    if not args.output_root.is_absolute() or not 1024 <= args.port <= 65535:
        parser.error('output-root must be absolute and port must be 1024..65535')
    worker = Worker(args)
    server = Server(('127.0.0.1', args.port), Handler)
    server.worker = worker
    threading.Thread(target=worker.load, name='load-model', daemon=True).start()
    try:
        server.serve_forever(poll_interval=.5)
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
