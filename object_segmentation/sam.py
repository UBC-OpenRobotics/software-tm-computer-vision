import numpy
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import supervision as sv
import json
# import rospy

from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image


def crop_to_square(image, bbox):
    """
    Given an image (H x W x 3) and a bounding box [x, y, w, h],
    return a centered square crop. Any empty areas are padded with black.
    """
    (x, y, w, h) = bbox
    H, W, _ = image.shape

    # Compute the side length of the square
    side = int(max(w, h))

    # Determine the center of the bounding box
    center_x = x + w // 2
    center_y = y + h // 2

    # Compute top-left corner of the square
    # so that center of bounding box remains at the center of the square
    left = int(center_x - side // 2)
    top  = int(center_y - side // 2)
    right = int(left + side)
    bottom = int(top + side)

    # Clamp to image boundaries in case the bounding box is near edges
    left   = max(0, left)
    top    = max(0, top)
    right  = min(W, right)
    bottom = min(H, bottom)

    # Crop the region from the original image
    cropped = image[top:bottom, left:right]

    # If the cropped region is already square-shaped (side x side) and 
    # fully inside the image, `cropped` will be the correct shape. 
    # However, if bounding box is near the boundary, the crop might be smaller. 
    # We'll pad it to ensure it is `side x side`.

    crop_height, crop_width, _ = cropped.shape
    # Create a black canvas for the final square
    final_crop = np.zeros((side, side, 3), dtype=np.uint8)

    # Compute how to place the cropped image in the square canvas
    offset_y = (side - crop_height) // 2
    offset_x = (side - crop_width) // 2

    final_crop[offset_y:offset_y+crop_height, offset_x:offset_x+crop_width] = cropped

    return final_crop

def main(IMAGE_PATH):
    rospy.init_node("sam_square_cropper", anonymous=True)

    # cropped images publisher
    pub = rospy.Publisher("/cropped_square_images", Image, queue_size=10)
    bridge = CvBridge()

    image_bgr = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

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
    annotated_image = mask_annotator.annotate(image_bgr.copy(), detections)

    ## If you want to save image as a json file
    # saveable_result = []
    # for mask_info in result:
    #     # Omit the "segmentation" field (which is a numpy array) 
    #     # and just store metadata that is JSON-friendly
    #     item = {
    #         "area": mask_info["area"],
    #         "bbox": mask_info["bbox"],
    #         "predicted_iou": mask_info["predicted_iou"],
    #         "point_coords": mask_info["point_coords"],
    #         "stability_score": mask_info["stability_score"],
    #         "crop_box": mask_info["crop_box"]
    #     }
    #     saveable_result.append(item)

    # # Now dump to JSON
    # with open("result.json", "w") as f:
    #     json.dump(saveable_result, f, indent=2)

    for i, mask_info in enumerate(result):
        bbox = mask_info["bbox"]  # [x, y, w, h]
        cropped_square = crop_to_square(image_bgr, bbox)
        # cropped_square_rgb = cv2.cvtColor(cropped_square, cv2.COLOR_BGR2RGB)
        ros_image = bridge.cv2_to_imgmsg(cropped_square, encoding="bgr8")
        pub.publish(ros_image)
        rospy.loginfo(f"Published cropped square image {i}")


    # annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # plt.imshow(annotated_image_rgb)
    # plt.title("Annotated Image")
    # plt.axis("off")
    # plt.show()

if __name__ == "__main__":
    IMAGE_PATH = "ycb/026_sponge/NP1_0.jpg"
    main(IMAGE_PATH)
