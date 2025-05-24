from pathlib import Path
import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

__all__ = ["SamCropper"]

class SamCropper:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device = "cuda:0",
        generator_kwargs: dict | None = None,
    ):
        self.device = torch.device(device)
        model = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
        model.to(self.device).eval()

        default_kwargs = dict(
            points_per_side=32,
            pred_iou_thresh=0.98,
            stability_score_thresh=0.96,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
        default_kwargs.update(generator_kwargs or {})
        self.mask_gen = SamAutomaticMaskGenerator(model=model, **default_kwargs)

    def crop_square_regions(self, bgr: np.ndarray) -> list[np.ndarray]:
        """Return square crops (BGR) for every mask bbox."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        masks = self.mask_gen.generate(rgb)
        return [self._crop_to_square(bgr, m["bbox"]) for m in masks]

    @staticmethod
    def _crop_to_square(img: np.ndarray, bbox) -> np.ndarray:
        x, y, w, h = bbox
        H, W, _ = img.shape
        side = int(max(w, h))
        cx, cy = x + w // 2, y + h // 2
        l, t = max(0, cx - side // 2), max(0, cy - side // 2)
        r, b = min(W, l + side), min(H, t + side)
        crop = img[t:b, l:r]

        # pad if crop touches an image border
        canvas = np.zeros((side, side, 3), dtype=np.uint8)
        dy, dx = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
        canvas[dy:dy + crop.shape[0], dx:dx + crop.shape[1]] = crop
        return canvas
