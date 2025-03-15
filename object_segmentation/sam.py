import numpy
import torch
import matplotlib.pyplot as plt
import cv2
import supervision as sv
import json

from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator

IMAGE_PATH = "cropped-banner4.jpg"

image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# plt.imshow(image_rgb)
# plt.title("Annotated Image")
# plt.axis("off")
# plt.show()

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sam.to(device=DEVICE)

# parameter
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side = 32,
    pred_iou_thresh = 0.980,
    stability_score_thresh = 0.96,
    crop_n_layers = 1,
    crop_n_points_downscale_factor = 2,
    min_mask_region_area = 100
)

result = mask_generator.generate(image_rgb)

mask_annotator = sv.MaskAnnotator(
    color_lookup=sv.ColorLookup.INDEX
)
detections = sv.Detections.from_sam(result)
annotated_image = mask_annotator.annotate(image_bgr, detections)

# print(result)
# with open("result.json", "w") as f:
#     json.dump(result, f)
saveable_result = []
for mask_info in result:
    # Omit the "segmentation" field (which is a numpy array) 
    # and just store metadata that is JSON-friendly
    item = {
        "area": mask_info["area"],
        "bbox": mask_info["bbox"],
        "predicted_iou": mask_info["predicted_iou"],
        "point_coords": mask_info["point_coords"],
        "stability_score": mask_info["stability_score"],
        "crop_box": mask_info["crop_box"]
    }
    saveable_result.append(item)

# Now dump to JSON
with open("result.json", "w") as f:
    json.dump(saveable_result, f, indent=2)


annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

plt.imshow(annotated_image_rgb)
plt.title("Annotated Image")
plt.axis("off")
plt.show()
