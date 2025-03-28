#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from threading import Lock

class ColourChaser(Node):
    def __init__(self):
        super().__init__('colour_chaser')
        
        self.turn_vel = 0.0
        self.forward_vel = 0.0
        self.pub_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 10)
        timer_period = 0.1  # 0.1 seconds per tick
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.create_subscription(Image, '/limo/depth_camera_link/image_raw', self.camera_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.laser_scan_callback, 10)
        self.br = CvBridge()
        self.frontAvoidRange = None
        self.avoid_distance = 0.30  # Stop when obstacle is within 0.3 meters

        # Simplified state variables
        self.state = "SEARCHING"  # or "PUSHING"
        self.target_in_view = False
        self.target_centered = False
        self.laser_scan = None
        self.top_color = None
        self.bottom_color = None

    def camera_callback(self, data):
        current_frame = self.br.imgmsg_to_cv2(data, desired_encoding='bgr8')
        current_frame_hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
        current_frame_mask = cv2.inRange(current_frame_hsv, (0, 150, 50), (255, 255, 255))
        contours, _ = cv2.findContours(current_frame_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        image_center_y = current_frame_hsv.shape[0] // 2
        offset_y = current_frame_hsv.shape[0]//3
        bottom_contours = []
        top_contours = []


        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cy = int(M['m01'] / M['m00'])
                if cy > image_center_y and cy<image_center_y+offset_y and cv2.contourArea(contour)> 40:
                    # print(cv2.contourArea(contour))
                    bottom_contours.append(contour)
                else:
                    top_contours.append(contour)
        
              

        if len(top_contours) > 0:
            T = cv2.moments(top_contours[0])
            if T['m00'] > 0:
                tx = int(T['m10'] / T['m00'])
                ty = int(T['m01'] / T['m00'])
                self.top_color = np.array(current_frame[ty, tx])  # Force NumPy array
                # print(f"Upper color (BGR): {self.top_color}")


        if len(bottom_contours) > 0:
            self.target_in_view = True
            M = cv2.moments(bottom_contours[0])
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])  # y-coordinate of centroid
                self.bottom_color = np.array(current_frame[cy, cx])  # Force NumPy array
                # print(f"Lower color (BGR): {self.bottom_color}")
                # Check if target is centered (within middle 1/3 of image)
                self.target_centered = (data.width / 3 <= cx <= 2 * data.width / 3)
                # print("Area",cv2.contourArea(bottom_contours[0]))
                self.targetArea =cv2.contourArea(bottom_contours[0])
                print("Area", self.targetArea)

                # Get bounding rectangle to find width/height
                x, y, w, h = cv2.boundingRect(bottom_contours[0])
                print(f"Contour width: {w}, height: {h}")
                

        else:
            self.target_in_view = False
            self.target_centered = False

        # Draw contours
        if len(bottom_contours) > 0:
            current_frame_contours = cv2.drawContours(current_frame, bottom_contours, 0, (255, 255, 0), 3)
        else:
            current_frame_contours = current_frame

        cv2.imshow("Image window", cv2.resize(current_frame_contours, (0, 0), fx=0.4, fy=0.4))
        cv2.waitKey(1)

    def timer_callback(self):     
        if self.laser_scan is None:  # Check if laser data is available
            self.forward_vel = 0.0
            self.turn_vel = 0.0
            print("Waiting for laser data...")
            return

        # Obstacle detection with better front range
        obstacle_detected = False
        if self.frontAvoidRange is not None:
            min_distance = min(self.frontAvoidRange)
            if min_distance < self.avoid_distance:
                obstacle_detected = True
                print(f"Obstacle detected! Min distance: {min_distance:.2f}m")
        
        if obstacle_detected:
            # Stop forward motion and turn away
            self.forward_vel = 0.0  # Stop moving forward         
            self.turn_vel = 0.3  # Increased turn speed
            self.state = "SEARCHING"
            # print(self.state)
        elif self.state == "SEARCHING":
            if self.target_in_view and self.target_centered and self.targetArea > 300:
            #     and self.top_color is not None 
            #     and self.bottom_color is not None 
            #     and (
            #         (self.top_color[1] > 100 and self.bottom_color[1] > 100) 
            #         or (self.top_color[2] > 200 and self.bottom_color[2] > 200)
            #     )
            # ):
                self.state = "PUSHING"
            else:
                self.forward_vel = 0.0
                self.turn_vel = 0.3
        elif self.state == "PUSHING":
            # Check for obstacles even while pushing
            if min(self.frontAvoidRange) < self.avoid_distance:  # Closer threshold while pushing
                self.state = "SEARCHING"  # Switch back to searching if too close
                self.forward_vel = 0.0
                # print(self.state)
            else:
                self.forward_vel = 0.2
                self.turn_vel = 0.0
            # print(self.state)

        self.tw = Twist()
        self.tw.linear.x = self.forward_vel
        self.tw.angular.z = self.turn_vel
        self.pub_cmd_vel.publish(self.tw)

    def laser_scan_callback(self, data):
        self.laser_scan = data
        if self.laser_scan is not None:
            rangeMin = self.laser_scan.angle_min
            rangeMax = self.laser_scan.angle_max
            increment = self.laser_scan.angle_increment
            # Define a narrower front range (e.g., ±45° from center)
            middle = (rangeMin + rangeMax) / 2
            forward_index = int((middle - rangeMin) / increment)
            self.frontAvoidRange = self.laser_scan.ranges[forward_index-35:forward_index+35]

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