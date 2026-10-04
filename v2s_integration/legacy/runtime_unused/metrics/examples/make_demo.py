"""Create a reproducible miniature evaluation fixture, not benchmark results."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    root = parser.parse_args().output
    rng = np.random.default_rng(12)
    cloud = rng.random((2000, 3)) * [0.5, 0.5, 1.0]
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[16:48, 20:44] = [100, 150, 200]
    for side in ("pred", "gt"):
        folder = root / side
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "surface.npy", cloud)
        np.save(folder / "chair.npy", cloud)
        Image.fromarray(image).save(folder / "input-01.png")
        hidden = image.copy()
        if side == "pred":
            hidden = np.roll(hidden, 4, axis=1)
        Image.fromarray(hidden).save(folder / "test-01.png")
    manifest = json.loads(Path(__file__).with_name("manifest.json").read_text())
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(root / "manifest.json")


if __name__ == "__main__":
    main()
