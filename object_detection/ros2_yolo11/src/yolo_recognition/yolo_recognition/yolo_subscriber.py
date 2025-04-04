import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image

from ultralytics import YOLO
from cv_bridge import CvBridge



class YoloSubscriber(Node):

    def __init__(self):
        super().__init__('yolo_subscriber')
        self.bridge = CvBridge()
        self.model = YOLO('yolo11n.pt')
        self.get_logger().info("model created")
        self.subscription = self.create_subscription(
            Image,
            '/image/rgb',
            self.camera_callback,
            10)
        self.subscription 

    def camera_callback(self, msg):
        self.get_logger().info('Receiving video frame')
        img = self.bridge.imgmsg_to_cv2(msg)
        results = self.model(img)

        for result in results:
            boxes = result.boxes
            for box in boxes:

                b = box.xyxy[0].to('cpu').detach().numpy().copy()  
                c = box.cls
                self.get_logger().info(f'{self.model.names[int(c)]} at {b[0], b[1], b[2], b[3]}')
                #self.get_logger().info(str(int(b[0])))


def main(args=None):
    rclpy.init(args=args)

    yolo_subscriber = YoloSubscriber()

    rclpy.spin(yolo_subscriber)


    yolo_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()