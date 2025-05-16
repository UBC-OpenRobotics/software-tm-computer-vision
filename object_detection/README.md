### Object Dection Module

Currently a ROS2 package utilizing an untrained YOLO v11 model. There is a webcam publisher that publishes webcam data to the topic "/image/rgb", and a subscriber that is subscribed to that same topic. The subscriber runs object detection on the published image and returns the detected object and a corresponding bounding box. 

### Future Considerations

The current YOLO model is untrained, and good for general object detection. For this to work better with Robocup @ home, the YOLO model should be trained on a dataset comprised of objects from the competition items. 