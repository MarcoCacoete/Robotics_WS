#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from threading import Lock

class ColourChaser(Node):
    def __init__(self):
        super().__init__('colour_chaser')
        
        self.turn_vel = 0.0
        self.forward_vel = 0.0
        self.pub_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 10)
        
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.create_subscription(Image, '/limo/depth_camera_link/image_raw', self.camera_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.laser_scan_callback, 10)
        
        self.br = CvBridge()
        self.forward_range_value = None
        self.initial_distance = None
        self.laser_scan = None
        self.middleValue = None
        self.initial_distance = None
        self.key = True
        self.lock = Lock()
        self.horFlip = False    

    def camera_callback(self, data):
        current_frame = self.br.imgmsg_to_cv2(data, desired_encoding='bgr8')
        current_frame_hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
        current_frame_mask = cv2.inRange(current_frame_hsv, (0, 150, 50), (255, 255, 255))
        contours, _ = cv2.findContours(current_frame_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        top_contours = []
        bottom_contours = []

        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cy = int(M['m01'] / M['m00'])
                if cy < data.height / 2:
                    top_contours.append(contour)
                else:
                    bottom_contours.append(contour)

        top_contours = sorted(top_contours, key=cv2.contourArea, reverse=True)[:1]
        bottom_contours = sorted(bottom_contours, key=cv2.contourArea, reverse=True)[:1]

        self.turn_vel = 0.0
        self.forward_vel = 0.0

        for contour in top_contours:
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(current_frame, (cx, cy), 5, (255, 0, 0), -1)

        if len(bottom_contours) > 0:
            M = cv2.moments(bottom_contours[0])
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(current_frame, (cx, cy), 5, (0, 255, 0), -1)
                
                if cy >= data.height / 2:
                    if cx < data.width / 3:
                        self.turn_vel = 0.3
                    elif cx >= 2 * data.width / 3:
                        self.turn_vel = -0.3
                    else:
                        self.turn_vel = 0.0
                        self.forward_vel = 0.1
        else:
            with self.lock:
                if self.key == True:
                    self.initial_distance = self.middleValue
                    print("in")
                    self.key = False
                print("out")
                cubeDistance = self.initial_distance - self.middleValue
                print(cubeDistance)
                if cubeDistance < 0.2:
                    self.forward_vel = 0.1  
                else:
                    self.forward_vel = -0.2
                self.key = True

        current_frame_contours = cv2.drawContours(current_frame, contours, 0, (255, 255, 0), 3)
        cv2.imshow("Image window", cv2.resize(current_frame_contours, (0, 0), fx=0.4, fy=0.4))
        cv2.waitKey(1)

    def laser_scan_callback(self, data):
        self.laser_scan = data
        if self.laser_scan is not None:
            rangeMin = self.laser_scan.angle_min
            rangeMax = self.laser_scan.angle_max
            increment = self.laser_scan.angle_increment
            middle = (rangeMin + rangeMax) / 2
            forward_index = int((middle - rangeMin) / increment)
            self.middleValue = self.laser_scan.ranges[forward_index]

    def timer_callback(self):
        self.tw = Twist()
        self.tw.linear.x = self.forward_vel
        self.tw.angular.z = self.turn_vel
        self.pub_cmd_vel.publish(self.tw)

def main(args=None):
    print('Starting colour_chaser.py.')
    cv2.startWindowThread()
    rclpy.init(args=args)
    colour_chaser = ColourChaser()
    rclpy.spin(colour_chaser)
    colour_chaser.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()