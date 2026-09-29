"""Optional SAM 2.1 bbox predictor; installation and checkpoints are explicit."""
from pathlib import Path


class SAM2Segmenter:
    def __init__(self, checkpoint, config="configs/sam2.1/sam2.1_hiera_l.yaml", device="cuda:0"):
        self.checkpoint = str(Path(checkpoint).resolve())
        if not Path(self.checkpoint).is_file(): raise ValueError("SAM checkpoint not found")
        self.config, self.device, self.predictor = config, device, None
        self.version = "sam2.1-bbox-v1:" + Path(checkpoint).name

    def predict(self, image, bbox):
        import numpy as np
        import torch
        if self.predictor is None:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            self.predictor = SAM2ImagePredictor(build_sam2(self.config, self.checkpoint, device=self.device))
        with torch.inference_mode():
            self.predictor.set_image(np.asarray(image))
            masks, scores, _ = self.predictor.predict(box=np.asarray(bbox, dtype=np.float32), multimask_output=False)
        return masks[0], float(scores[0])
