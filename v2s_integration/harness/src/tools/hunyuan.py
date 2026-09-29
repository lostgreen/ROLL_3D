"""Opt-in Hunyuan3D generation through explicit local or HTTP backends.

No Blender mutation, hidden polling or retry.

The upstream synchronous /generate endpoint is used because /status only treats
textured files as complete in the inspected API revision. All input image IDs
must be explicitly registered by the environment, never model-provided paths.
"""
import base64
import hashlib
import json
import struct
import time
import uuid
from pathlib import Path
from environment.artifacts import ArtifactManifest
from environment.budget import Usage
from tools.base import ToolResult, ToolSpec

PRESETS = {'draft': {'octree_resolution':128, 'num_inference_steps':5},
           'balanced': {'octree_resolution':256, 'num_inference_steps':10},
           'high': {'octree_resolution':512, 'num_inference_steps':20}}
V21_PRESETS = {'draft': {'octree_resolution':256, 'num_inference_steps':20},
               'balanced': {'octree_resolution':384, 'num_inference_steps':30},
               'high': {'octree_resolution':512, 'num_inference_steps':50}}


def validate_glb(data):
    if len(data) < 20 or data[:4] != b'glTF': raise ValueError('invalid_glb_magic')
    version, length = struct.unpack_from('<II', data, 4)
    if version != 2 or length != len(data): raise ValueError('invalid_glb_length')
    size, kind = struct.unpack_from('<II', data, 12)
    if kind != 0x4E4F534A or 20 + size > len(data): raise ValueError('invalid_glb_json_chunk')
    document = json.loads(data[20:20+size])
    if not document.get('meshes'): raise ValueError('empty_generated_mesh')
    # External resources cannot be resolved by the artifact-only import contract.
    for item in document.get('buffers', []) + document.get('images', []):
        if item.get('uri') and not item['uri'].startswith('data:'):
            raise ValueError('external_glb_resource')
    return document


def texture_summary(document):
    """Count material-linked textures on UV-mapped primitives, not loose images."""
    images, textures = document.get('images', []), document.get('textures', [])
    materials = document.get('materials', [])
    counts = {'base_color_primitives': 0, 'pbr_mr_primitives': 0, 'embedded_images': len(images)}
    for mesh in document.get('meshes', []):
        for primitive in mesh.get('primitives', []):
            index = primitive.get('material', -1)
            if not isinstance(index, int) or not 0 <= index < len(materials): continue
            pbr = materials[index].get('pbrMetallicRoughness', {})
            for field, key in [('baseColorTexture', 'base_color_primitives'),
                               ('metallicRoughnessTexture', 'pbr_mr_primitives')]:
                info = pbr.get(field, {})
                idx = info.get('index', -1)
                if not isinstance(idx, int) or not 0 <= idx < len(textures): continue
                source = textures[idx].get('source', -1)
                if (isinstance(source, int) and 0 <= source < len(images)
                        and ('bufferView' in images[source] or images[source].get('uri', '').startswith('data:'))
                        and 'TEXCOORD_'+str(info.get('texCoord', 0)) in primitive.get('attributes', {})):
                    counts[key] += 1
    return counts


class HunyuanHTTPBackend:
    version = 'hunyuan3d-2.1-generate.v1'

    def __init__(self, endpoint, timeout=180, max_bytes=100 * 1024 * 1024):
        from urllib.parse import urlparse
        parsed = urlparse(endpoint)
        if parsed.scheme not in ('http', 'https') or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError('explicit HTTP service endpoint required')
        self.endpoint, self.timeout, self.max_bytes = endpoint.rstrip('/'), timeout, max_bytes

    def generate(self, payload, timeout=None):
        import httpx
        # Never follow redirects or resubmit uncertain generation requests.
        with httpx.stream('POST', self.endpoint + '/generate', json=payload,
                          timeout=min(timeout, self.timeout) if timeout is not None else self.timeout,
                          follow_redirects=False) as response:
            response.raise_for_status()
            data = bytearray()
            for chunk in response.iter_bytes():
                data.extend(chunk)
                if len(data) > self.max_bytes: raise ValueError('generated_asset_too_large')
            return bytes(data)


class HunyuanProvider:
    VERSION = 'hunyuan-provider.v2-backend-presets'

    def __init__(self, store, backend, references, recorder=None, budget=None):
        self.store, self.backend = store, backend
        self.references, self.recorder, self.budget = references, recorder, budget
        self.presets = getattr(backend, 'presets', PRESETS)

    def register(self, registry):
        texture_schema = {'type':'boolean', 'default':getattr(self.backend, 'supports_texture', True)}
        if not getattr(self.backend, 'supports_texture', True): texture_schema['enum'] = [False]
        registry.register(ToolSpec('asset.generate_3d_from_image',
            'Generate one reusable 3D asset from an available reference image. '
            'The returned artifact is not automatically added to the Blender scene.',
            {'type':'object', 'properties': {
                'reference_image':{'type':'string'}, 'texture':texture_schema,
                'seed':{'type':'integer','minimum':0,'maximum':2**32-1},
                'quality_preset':{'type':'string','enum':list(self.presets),'default':'balanced'}},
             'required':['reference_image'],'additionalProperties':False},
            'hunyuan3d', self.VERSION + ':' + self.backend.version,
            usage_upper_bound=Usage(None, None, 1, 0), category='generation',
            side_effects=('creates_artifact',), async_mode='environment_managed',
            ui={'display_name':'图像生成 3D 资产','result_renderer':'artifact_generation'}), self.generate)

    def event(self, kind, **payload):
        if self.recorder: self.recorder.event(kind, **payload)

    def generate(self, args):
        allowed = {'reference_image','texture','seed','quality_preset'}
        ref, texture = args.get('reference_image'), args.get('texture', getattr(self.backend, 'supports_texture', True))
        preset, seed = args.get('quality_preset', 'balanced'), args.get('seed', 1234)
        if (args.keys() - allowed or not isinstance(ref, str) or ref not in self.references or
            not isinstance(texture, bool) or (texture and not getattr(self.backend, 'supports_texture', True)) or not isinstance(seed, int) or isinstance(seed, bool) or
            not 0 <= seed <= 2**32-1 or not isinstance(preset, str) or preset not in self.presets):
            return ToolResult('ERROR: invalid generation arguments or unregistered image reference',
                              usage=Usage.zero(), status='rejected', error_type='invalid_arguments')
        generation_id = uuid.uuid4().hex
        started = time.monotonic()
        submitted = False
        try:
            image_path = Path(self.references[ref])
            if image_path.stat().st_size > 20 * 1024 * 1024: raise ValueError('invalid_image_size')
            image = image_path.read_bytes()
            if not image or len(image) > 20 * 1024 * 1024: raise ValueError('invalid_image_size')
            timeout = None
            if self.budget:
                self.budget.check_wall()
                cap = self.budget.limits.max_wall_seconds
                if cap is not None: timeout = max(.001, cap - (self.budget.clock() - self.budget.started))
            payload = dict(self.presets[preset], image=base64.b64encode(image).decode(), texture=texture, seed=seed)
            self.event('generation.start', generation_id=generation_id, reference_image=ref,
                       image_sha256=hashlib.sha256(image).hexdigest(), preset=preset, seed=seed,
                       texture=texture, provider_version=self.VERSION, backend_version=self.backend.version,
                       input_image=(self.recorder.blob(image, '.media') if self.recorder else None),
                       input_mime='image/png' if image.startswith(b'\x89PNG') else 'image/jpeg',
                       backend_parameters={k:v for k,v in payload.items() if k != 'image'},
                       text_prompt=None, input_mode='image_to_3d',
                       evidence_scope='provider adapter input before backend preprocessing')
            prepare = getattr(self.backend, 'prepare_image', None)
            if callable(prepare):
                prepared, preprocessing = prepare(image)
                payload['image'] = base64.b64encode(prepared).decode()
                self.event('generation.preprocessed', generation_id=generation_id,
                           preprocessing=preprocessing, input_mime='image/png',
                           input_image=(self.recorder.blob(prepared, '.media') if self.recorder else None),
                           image_sha256=hashlib.sha256(prepared).hexdigest(),
                           evidence_scope='shape pipeline image before internal resize and normalization')
            submitted = True
            data = self.backend.generate(payload, timeout=timeout)
            textures = texture_summary(validate_glb(data))
            if texture and not textures['base_color_primitives']:
                raise ValueError('requested texture but GLB has no material-linked base-color image and UVs')
            artifact_id = 'hunyuan_' + generation_id
            target = self.store.root / (artifact_id + '.glb')
            target.write_bytes(data)
            manifest = self.store.register(ArtifactManifest(artifact_id, 'mesh', target.name, 'glb',
                'asset.generate_3d_from_image', self.VERSION, 'provider_native', 'unknown',
                evidence_refs=(ref,), editable=True, renderable=True,
                normalization={'generation_id':generation_id, 'reference_sha256':hashlib.sha256(image).hexdigest(),
                               'seed':seed, 'texture':texture, 'quality_preset':preset,
                               'backend_version':self.backend.version, 'texture_summary': textures}))
            structured = {'generation_id':generation_id, 'artifact_id':artifact_id,
                          'sha256':manifest['sha256'], 'bytes':len(data), 'texture':texture,
                          'quality_preset':preset, 'seed':seed, 'duration_sec':time.monotonic()-started,
                          'coordinate_frame':'provider_native','units':'unknown', 'texture_summary': textures,
                          'backend_details': getattr(self.backend, 'last_metadata', {})}
            self.event('artifact.created', artifact=manifest)
            self.event('generation.end', status='succeeded', **structured)
            return ToolResult(json.dumps(structured), usage=Usage(None, None, 1, 0),
                              artifact_ids=[artifact_id], structured=structured)
        except Exception as exc:
            error = type(exc).__name__
            # An uncertain timeout never triggers duplicate GPU work; costs remain unknown.
            self.event('generation.end', generation_id=generation_id, status='failed', error_type=error,
                       duration_sec=time.monotonic()-started, submitted=submitted,
                       backend_details=getattr(self.backend, 'last_metadata', {}))
            return ToolResult('ERROR: generation failed (' + error + ')',
                              usage=Usage(None, None, None, 0) if submitted else Usage.zero(),
                              status='failed', error_type='generation_failed',
                              structured={'generation_id':generation_id, 'submitted':submitted})


class HunyuanLocalBackend:
    """Local 2.x shape pipeline; textures require a separate backend.

    Loading is lazy so constructing a session never allocates a GPU. The caller
    chooses an existing device and local checkpoint; no implicit model download.
    """
    supports_texture = False

    def __init__(self, model_path, subfolder, device='cuda:0'):
        model_path = Path(model_path).resolve()
        model_dir = model_path / subfolder
        self.use_safetensors = (model_dir / 'model.fp16.safetensors').is_file()
        if not (model_dir / 'config.yaml').is_file() or not (self.use_safetensors or (model_dir / 'model.fp16.ckpt').is_file()):
            raise ValueError('local Hunyuan config and fp16 safetensors/ckpt are required')
        self.model_path, self.subfolder, self.device = model_path, subfolder, device
        self.version = 'hunyuan3d-local-shape.v2-rembg:' + subfolder
        self.presets = V21_PRESETS if subfolder == 'hunyuan3d-dit-v2-1' else PRESETS
        self.pipeline = None
        self.background_remover = None

    def prepare_image(self, data):
        import io
        from PIL import Image
        source = Image.open(io.BytesIO(data))
        image = source.convert('RGBA')
        before = image.getchannel('A').getextrema()
        method = 'preserve_existing_alpha'
        if before == (255, 255):
            if self.background_remover is None:
                if self.subfolder == 'hunyuan3d-dit-v2-1':
                    from hy3dshape.rembg import BackgroundRemover
                else:
                    from hy3dgen.rembg import BackgroundRemover
                self.background_remover = BackgroundRemover()
            image = self.background_remover(source.convert('RGB')).convert('RGBA')
            method = 'official_background_remover'
        after = image.getchannel('A').getextrema()
        if after[1] == 0 or after[0] == 255:
            raise ValueError('foreground alpha mask must contain foreground and background')
        output = io.BytesIO()
        image.save(output, format='PNG')
        return output.getvalue(), {'method':method, 'source_mode':source.mode,
                                  'alpha_before':list(before), 'alpha_after':list(after),
                                  'size':list(image.size)}

    def load(self):
        if self.pipeline is None:
            import torch
            if self.subfolder == 'hunyuan3d-dit-v2-1':
                from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
            else:
                from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
            self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                str(self.model_path), subfolder=self.subfolder, use_safetensors=self.use_safetensors,
                variant='fp16', device=self.device, dtype=torch.float16)
        return self.pipeline

    def generate(self, payload, timeout=None):
        import io
        import torch
        from PIL import Image
        if payload.get('texture'): raise ValueError('local backend supports shape generation only')
        prepared, _ = self.prepare_image(base64.b64decode(payload['image'], validate=True))
        image = Image.open(io.BytesIO(prepared)).convert('RGBA')
        pipeline = self.load()
        mesh = pipeline(image=image, num_inference_steps=payload['num_inference_steps'],
                        octree_resolution=payload['octree_resolution'],
                        generator=torch.Generator(device=self.device).manual_seed(payload['seed']))[0]
        return mesh.export(file_type='glb')
