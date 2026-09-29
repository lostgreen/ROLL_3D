"""Optional official Hunyuan 2.1 Paint worker, isolated from the agent's imports."""
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


class HunyuanTexturedBackend:
    supports_texture = True

    def __init__(self, shape, code_root, work_root, python=None):
        self.shape, self.code_root = shape, Path(code_root).resolve()
        if not (self.code_root / "hy3dpaint/textureGenPipeline.py").is_file():
            raise ValueError("official Hunyuan 2.1 Paint code not found")
        self.work_root, self.python = Path(work_root), python or sys.executable
        self.presets, self.prepare_image = shape.presets, shape.prepare_image
        self.version = shape.version + ":paint2.1-v1-6views-512-1024tex"
        self.last_metadata = {}

    def generate(self, payload, timeout=None):
        started = time.monotonic()
        self.last_metadata = {}
        mesh = self.shape.generate({**payload, "texture": False}, timeout=timeout)
        if not payload.get("texture"): return mesh
        folder = self.work_root / uuid.uuid4().hex
        folder.mkdir(parents=True)
        (folder / "shape.glb").write_bytes(mesh)
        (folder / "input.png").write_bytes(base64.b64decode(payload["image"], validate=True))
        remaining = None if timeout is None else max(.001, timeout-(time.monotonic()-started))
        self.last_metadata = {"intermediate_dir": str(folder), "paint_input": str(folder/"input.png"),
                              "shape_glb": str(folder/"shape.glb"), "texture_size": 1024, "paint_seed": 0}
        # The official worker uses top-level utils and bpy; subprocess isolation avoids
        # polluting the harness namespace. Never retry uncertain GPU work automatically.
        with (folder / "paint.log").open("w") as log:
            subprocess.run([self.python, str(Path(__file__).resolve()), str(self.code_root), str(folder)],
                           stdout=log, stderr=subprocess.STDOUT, timeout=remaining, check=True)
        output = folder / "textured.glb"
        self.last_metadata["textured_glb"] = str(output)
        return output.read_bytes()


def paint(code_root, folder):
    import torch
    code_root, folder = Path(code_root), Path(folder)
    sys.path[:0] = [str(code_root), str(code_root / "hy3dpaint")]
    # Official compatibility shim for BasicSR on current torchvision.
    from torchvision_fix import apply_fix
    apply_fix()
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
    from huggingface_hub import snapshot_download
    config = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
    config.multiview_cfg_path = str(code_root / "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml")
    config.dino_ckpt_path = snapshot_download("facebook/dinov2-giant", local_files_only=True)
    config.realesrgan_ckpt_path = os.environ["V2S_REALESRGAN_CHECKPOINT"]
    config.render_size = config.texture_size = 1024
    torch.manual_seed(0)  # Official multiview implementation also fixes its seed to zero.
    pipeline = Hunyuan3DPaintPipeline(config)
    pipeline(mesh_path=str(folder/"shape.glb"), image_path=str(folder/"input.png"),
             output_mesh_path=str(folder/"textured.obj"), save_glb=True)
    (folder/"paint_settings.json").write_text(json.dumps({"views":6,"resolution":512,
        "texture_size":1024,"paint_seed":0,"model":"tencent/Hunyuan3D-2.1"}, indent=2))


if __name__ == "__main__":
    paint(*sys.argv[1:])
