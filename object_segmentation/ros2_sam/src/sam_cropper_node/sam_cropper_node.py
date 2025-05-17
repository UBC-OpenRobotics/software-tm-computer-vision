import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from sam_cropper_py import SamCropper   # ← import the library

class SamCropperNode(Node):
    def __init__(self):
        super().__init__("sam_cropper_node")
        self.bridge = CvBridge()

        # Declare parameters so they can be set via a launch file
        self.declare_parameter("checkpoint", "sam_vit_h_4b8939.pth")
        ckpt = self.get_parameter("checkpoint").get_parameter_value().string_value

        # Heavy model can take a moment; do it once here
        self.cropper = SamCropper(ckpt)

        self.pub = self.create_publisher(Image, "/cropped_images", 10)
        self.sub = self.create_subscription(
            Image, "/image_raw", self.image_cb, 10
        )

    def image_cb(self, msg: Image):
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        crops = self.cropper.crop_square_regions(bgr)
        for idx, crop in enumerate(crops):
            ros_img = self.bridge.cv2_to_imgmsg(crop, encoding="bgr8")
            ros_img.header.stamp = self.get_clock().now().to_msg()
            ros_img.header.frame_id = f"crop_{idx}"
            self.pub.publish(ros_img)

def main():
    rclpy.init()
    node = SamCropperNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
