#!/usr/bin/env python

# An example of TurtleBot 3 subscribe to camera topic, mask colours, find and display contours, and move robot to center the object in image frame
# Written for humble
# cv2 image types - http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2
import numpy as np

class ColourChaser(Node):
    def __init__(self):
        super().__init__('colour_chaser')
        
        self.turn_vel = 0.0
        self.forward_vel = 0.0  # Initialize forward velocity

        # publish cmd_vel topic to move the robot
        self.pub_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 10)

        # create timer to publish cmd_vel topic
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # subscribe to the camera topic
        self.create_subscription(Image, '/limo/depth_camera_link/image_raw', self.camera_callback, 10)

        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()

    def camera_callback(self, data):
        # Convert ROS Image message to OpenCV image
        current_frame = self.br.imgmsg_to_cv2(data, desired_encoding='bgr8')

        # Convert image to HSV
        current_frame_hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
        # Create mask for range of colours (HSV low values, HSV high values)
        current_frame_mask = cv2.inRange(current_frame_hsv, (0, 150, 50), (255, 255, 255))  # orange

        contours, _ = cv2.findContours(current_frame_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

        # Default velocities
        self.turn_vel = 0.0
        self.forward_vel = 0.0  

        if len(contours) > 0:
            M = cv2.moments(contours[0])
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(current_frame, (cx, cy), 5, (0, 255, 0), -1)
                
                # Centering logic
                if cx < data.width / 3:
                    self.turn_vel = 0.3  # Turn left
                elif cx >= 2 * data.width / 3:
                    self.turn_vel = -0.3  # Turn right
                else:
                    self.turn_vel = 0.0  # No turn
                    self.forward_vel = 0.1

                # # Moving forward logic
                # area = cv2.contourArea(contours[0])
                # target_area = 50000  # Adjust based on object size and camera distance
                # if area < target_area:
                #     self.forward_vel = 0.1  # Move forward until object is large enough
                # else:
                #     self.forward_vel = 0.0  # Stop when object is "close enough"

        else:
            # No object detected: rotate to search
            self.turn_vel = 0.3
            self.forward_vel = 0.0  

        # Show image with contours
        current_frame_contours = cv2.drawContours(current_frame, contours, 0, (255, 255, 0), 3)
        cv2.imshow("Image window", cv2.resize(current_frame_contours, (0, 0), fx=0.4, fy=0.4))
        cv2.waitKey(1)

    def timer_callback(self):
        self.tw = Twist()
        self.tw.linear.x = self.forward_vel  # Set forward velocity
        self.tw.angular.z = self.turn_vel  # Set angular velocity
        self.pub_cmd_vel.publish(self.tw)

def main(args=None):
    print('Starting colour_chaser.py.')
    cv2.startWindowThread()
    rclpy.init(args=args)

    colour_chaser = ColourChaser()

    rclpy.spin(colour_chaser)

    # Destroy the node explicitly
    colour_chaser.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
