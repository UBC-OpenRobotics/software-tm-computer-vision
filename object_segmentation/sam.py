# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import cv2

# # def show_mask(mask, ax, random_color=False):
# #     for mask in masks:
# #         if random_color:
# #             color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
# #         else:
# #             color = np.array([30/255, 144/255, 255/255, 0.6])
# #         h, w = mask.shape[-2:]
# #         mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
# #         ax.imshow(mask_image)

# def show_mask(masks, ax, random_color=False):
#     for mask in masks:
#         seg = mask["segmentation"]  # <-- get the actual mask array
#         if random_color:
#             color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#         else:
#             color = np.array([30/255, 144/255, 255/255, 0.6])

#         # seg should be shape (H, W). Now you can do:
#         h, w = seg.shape[-2:]
#         mask_image = seg.reshape(h, w, 1) * color.reshape(1, 1, -1)
#         ax.imshow(mask_image)
    
# def show_points(coords, labels, ax, marker_size=375):
#     pos_points = coords[labels==1]
#     neg_points = coords[labels==0]
#     ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
#     ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
# def show_box(box, ax, edgecolor='green'):
#     x0, y0 = box[0], box[1]
#     w, h = box[2] - box[0], box[3] - box[1]
#     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))  


# # image = cv2.imread("ycb/026_sponge/NP1_0.jpg")
# image = cv2.imread("cropped-banner4.jpg")

# import sys
# sys.path.append("..")
# from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# sam_checkpoint = "sam_vit_h_4b8939.pth"
# model_type = "vit_h"

# # device = "cuda"
# device = "cpu"

# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# mask_generator = SamAutomaticMaskGenerator(sam)
# masks = mask_generator.generate(image)

# plt.figure(figsize=(8,8))
# plt.imshow(image[..., ::-1])  # if you want BGR->RGB correction
# show_mask(masks, plt.gca())
# plt.show()

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
