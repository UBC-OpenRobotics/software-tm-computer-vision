import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

class WebcamPublisher(Node):
    def __init__(self):
        super().__init__('webcam_publisher')
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        self.publisher_= self.create_publisher(Image, "/image/rgb", 10)
        timer_period = 0.2
        self.timer = self.create_timer(timer_period, self.timer_callback)
    
    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret == True:
            self.publisher_.publish(self.bridge.cv2_to_imgmsg(frame))
            self.get_logger().info("Publishing image frame")
        else:
            self.get_logger().warning("Failed to capture frame")
            self.cap.release()
            cv2.destroyAllWindows()

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)

    webcam_publisher = WebcamPublisher()

    try:
        rclpy.spin(webcam_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        webcam_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()