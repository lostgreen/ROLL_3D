"""Small explicit configuration; unknown keys and invalid units fail early."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class MetricConfig:
    image_metrics: tuple = ("mse", "psnr_db", "ssim")
    psnr_cap_db: float = 100.0
    ssim_window: int = 7
    worst_fraction: float = 0.2
    distance_thresholds_m: tuple = (0.02, 0.05, 0.1)
    surface_samples: int = 20000
    seed: int = 0
    match_max_distance_m: float = 1.0
    match_category: bool = True
    lpips_allow_download: bool = False

    def __post_init__(self):
        names = tuple(self.image_metrics)
        if not names or len(set(names)) != len(names) or set(names) - {"mse", "psnr_db", "ssim", "lpips"}:
            raise ValueError("image_metrics must contain unique supported names")
        thresholds = tuple(float(t) for t in self.distance_thresholds_m)
        if not thresholds or len(set(thresholds)) != len(thresholds):
            raise ValueError("distance_thresholds_m must be nonempty and unique")
        for value in (*thresholds, self.psnr_cap_db, self.match_max_distance_m):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("thresholds, PSNR cap and match distance must be positive and finite")
        if not math.isfinite(self.worst_fraction) or not 0 < self.worst_fraction <= 1:
            raise ValueError("worst_fraction must be in (0, 1]")
        for name, minimum in (("surface_samples", 1), ("seed", 0), ("ssim_window", 3)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.ssim_window % 2 == 0:
            raise ValueError("ssim_window must be odd")
        for name in ("match_category", "lpips_allow_download"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        object.__setattr__(self, "image_metrics", names)
        object.__setattr__(self, "distance_thresholds_m", tuple(sorted(thresholds)))

    def to_dict(self):
        return asdict(self)
