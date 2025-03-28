#!/usr/bin/env python
# This shebang line specifies that this script should be executed using the Python interpreter found in the system's environment.

# An example of TurtleBot 3 subscribing to a camera topic, masking colors, finding and displaying contours,
# and moving the robot to center the object in the image frame.
# Written for ROS 2 Humble.
# Reference for cv2 image types: http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython

import rclpy  # ROS 2 Python client library for building nodes
from rclpy.node import Node  # Base class for creating ROS 2 nodes

from sensor_msgs.msg import Image  # ROS 2 message type for image data
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist  # ROS 2 message type for velocity commands (linear and angular)
from cv_bridge import CvBridge  # Converts between ROS 2 Image messages and OpenCV images
import cv2  # OpenCV library for image processing
import numpy as np  # NumPy library for numerical operations (e.g., array manipulation)
import math
from threading import Lock

# Define a custom ROS 2 node class called ColourChaser, inheriting from Node
class ColourChaser(Node):
    def __init__(self):
        # Initialize the parent Node class with the node name 'colour_chaser'
        super().__init__('colour_chaser')
        
        # Initialize velocity variables as instance attributes
        self.turn_vel = 0.0  # Angular velocity (turning speed) in radians/second
        self.forward_vel = 0.0  # Linear velocity (forward speed) in meters/second

        # Create a publisher to send velocity commands to the 'cmd_vel' topic
        # Twist is the message type, 'cmd_vel' is the topic name, 10 is the queue size
        self.pub_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 10)

        # Create a timer to periodically call timer_callback (runs every 0.1 seconds)
        timer_period = 0.1  # Time interval in seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Subscribe to the camera's raw image topic
        # Image is the message type, '/limo/depth_camera_link/image_raw' is the topic,
        # camera_callback is the callback function, 10 is the queue size
        self.create_subscription(Image, '/limo/depth_camera_link/image_raw', self.camera_callback, 10)
        self.create_subscription(LaserScan, '/scan',self.laser_scan_callback,10 )
        # Initialize CvBridge to convert between ROS 2 and OpenCV image formats
        self.br = CvBridge()
        self.forward_range_value = None  # Initialize forward_range_value
        self.initial_distance = None

        self.laser_scan = None #initialize laser_scan
        self.middleValue = None
        self.initial_distance= None        
        self.key=True
        self.lock = Lock()     
        self.chaseMode = False
        self.frontAvoidRange= None


    # Callback function triggered when a new image is received from the camera topic
    def camera_callback(self, data):
        # Convert the ROS 2 Image message to an OpenCV image in BGR8 format (Blue-Green-Red, 8-bit)
        current_frame = self.br.imgmsg_to_cv2(data, desired_encoding='bgr8')

        # Convert the BGR image to HSV color space (Hue, Saturation, Value) for better color detection
        current_frame_hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

        # Create a binary mask to isolate a specific color range (orange in this case)
        # Arguments: HSV image, lower bound (H, S, V), upper bound (H, S, V)
        current_frame_mask = cv2.inRange(current_frame_hsv, (0, 150, 50), (255, 255, 255))  # Orange range

        # Find contours in the masked image
        # cv2.findContours returns contours (list of boundary points) and hierarchy (optional)
        # cv2.RETR_TREE retrieves all contours and organizes them in a hierarchy
        # cv2.CHAIN_APPROX_SIMPLE compresses horizontal/vertical segments into endpoints
        contours, _ = cv2.findContours(current_frame_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Get ALL contours sorted by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Fixed: reverse=True

        image_center_y = current_frame_hsv.shape[0] // 2  # Your existing correct definition
        bottom_contours = []
        top_contours = []

        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cy = int(M['m01'] / M['m00'])
                if cy > image_center_y:
                    bottom_contours.append(contour)
                else:
                    top_contours.append(contour)


        # Now use bottom_contours[0] if it exists

        # Reset velocities to default (stop) before processing
        self.turn_vel = 0.0
        self.forward_vel = 0.0          
        self.forward_vel = 0.0  
        top_color = None
        bottom_color = None


        if len(top_contours) > 0:
            T = cv2.moments(top_contours[0])
            if T['m00'] > 0:
                tx = int(T['m10'] / T['m00'])
                ty = int(T['m01'] / T['m00'])
                # top_color = np.array(current_frame[ty, tx])  # Force NumPy array
                # print(f"Upper color (BGR): {top_color}")


        # Check if any contours were found
        if len(bottom_contours) > 0:
            # Calculate moments of the largest contour to find its centroid
            M = cv2.moments(bottom_contours[0])
            if M['m00'] > 0:  # m00 is the contour area; ensure it's not zero to avoid division errors
                # Compute centroid coordinates (x, y)
                cx = int(M['m10'] / M['m00'])  # x-coordinate of centroid
                cy = int(M['m01'] / M['m00'])  # y-coordinate of centroid
                # bottom_color = np.array(current_frame[cy, cx])  # Force NumPy array
                # print(f"Lower color (BGR): {bottom_color}")

                # Draw a green circle at the centroid on the original image
                # Arguments: image, center (x, y), radius, color (B, G, R), thickness (-1 fills the circle)
                cv2.circle(current_frame, (cx, cy), 5, (0, 255, 0), -1)
                
                # if (bottom_color == top_color).all() and tx <data.width / 3 and tx >= 2 * data.width / 3:
                    # print("I'm inside, do the colours match?", bottom_color == top_color)
                    # Centering logic: Adjust turn_vel based on the object's horizontal position
                if cx < data.width / 3:  # Object is in the left third of the image
                    self.turn_vel = 0.3  # Turn left (positive angular velocity)
                elif cx >= 2 * data.width / 3:  # Object is in the right third of the image
                    self.turn_vel = -0.3  # Turn right (negative angular velocity)
                else:  # Object is in the center third
                    self.turn_vel = 0.0  # Stop turning
                    if all(distance > 0.1 for distance in self.frontAvoidRange):
                        self.forward_vel = 0.2 # Move forward slightly
                # else:
                #     self.turn_vel = 0.3
        else:
            self.turn_vel = 0.3

        # Draw the largest contour on the original image in yellow
        # Arguments: image, contours list, contour index (0), color (B, G, R), thickness
        current_frame_contours = cv2.drawContours(current_frame, bottom_contours, 0, (255, 255, 0), 3)

        # Display the processed image in a window
        # Resize the image to 40% of its original size for better visibility
        cv2.imshow("Image window", cv2.resize(current_frame_contours, (0, 0), fx=0.4, fy=0.4))
        cv2.waitKey(1)  # Wait 1ms to allow the window to refresh

        # Additional display option: Show the mask alongside the original image
        # cv2.imshow("Mask window", cv2.resize(current_frame_mask, (0, 0), fx=0.4, fy=0.4))
    
    def laser_scan_callback(self, data):
        self.laser_scan = data
        if self.laser_scan is not None:
            rangeMin = self.laser_scan.angle_min
            rangeMax = self.laser_scan.angle_max
            increment = self.laser_scan.angle_increment
            middle = (rangeMin+rangeMax) /2
            forward_index = int((middle - rangeMin) / increment)
            # print(forward_index) 
            self.middleValue = self.laser_scan.ranges[forward_index]
            self.frontAvoidRange = self.laser_scan.ranges[forward_index-15:forward_index+15]
            # print(self.frontAvoidRange)
            # print(self.middleValue)          


    # Timer callback to periodically publish velocity commands
    def timer_callback(self):
        # Create a new Twist message
        self.tw = Twist()
        # Set linear velocity (x-axis: forward/backward motion)
        self.tw.linear.x = self.forward_vel
        # Set angular velocity (z-axis: rotation)
        self.tw.angular.z = self.turn_vel
        # Publish the velocity command to the 'cmd_vel' topic
        self.pub_cmd_vel.publish(self.tw)

        # Additional possibility: Log velocities for debugging
        # self.get_logger().info(f"Publishing: linear.x={self.forward_vel}, angular.z={self.turn_vel}")

# Main function to initialize and run the ROS 2 node
def main(args=None):
    print('Starting colour_chaser.py.')  # Print a startup message
    cv2.startWindowThread()  # Start OpenCV window thread for displaying images
    rclpy.init(args=args)  # Initialize the ROS 2 Python client library

    # Create an instance of the ColourChaser node
    colour_chaser = ColourChaser()

    # Spin the node to keep it running and processing callbacks
    rclpy.spin(colour_chaser)

    # Cleanup when the node is stopped (e.g., via Ctrl+C)
    colour_chaser.destroy_node()  # Explicitly destroy the node
    rclpy.shutdown()  # Shutdown the ROS 2 client library

# Entry point of the script
if __name__ == '__main__':
    main()  # Call the main function if this script is run directly

    
        #     # No contours detected: Rotate to search for the object            
        #     print(self.forward_range_value)                 
        #     # self.turn_vel = 0.3  # Turn left to scan
        #     with self.lock:                        
        #         if self.key == True:
        #             self.initial_distance = self.middleValue
        #             print(self.initial_distance)
        #             print(self.middleValue)
        #             print("in")
        #             self.key = False
        #         print("out")
        #         # print(self.initial_distance)
        #         cubeDistance = self.initial_distance - self.middleValue
        #         print(cubeDistance)
        #         print("middle",self.middleValue)
        #         if cubeDistance < 0.2 or self.middleValue > 0.3:
        #             self.forward_vel = 0.1    
        #         else:
        #             print("not yet")
                               

                # self.key = True       

                # Alternative centering logic with proportional control (P-controller)
                # center_x = data.width / 2
                # error = (cx - center_x) / center_x  # Normalized error (-1 to 1)
                # self.turn_vel = -0.5 * error  # Proportional turn speed (faster when farther)

                # Moving forward logic based on contour area (commented out in original)
                # area = cv2.contourArea(contours[0])  # Calculate area of the contour
                # target_area = 50000  # Desired area when object is "close enough"
                # if area < target_area:os
                #     self.forward_vel = 0.1  # Move forward if object is too small (far away)
                # else:
                #     self.forward_vel = 0.0  # Stop if object is large enough (close enough)

                # Alternative forward logic with proportional control
                # max_area = 100000  # Maximum area when too close
                # if area < target_area:
                #     self.forward_vel = 0.2 * (target_area - area) / target_area  # Scale speed
                # elif area > max_area:
                #     self.forward_vel = -0.1  # Back up if too close
                # else:
                #     self.forward_vel = 0.0
                

            # Alternative search behavior: Random turn direction
            # import random
            # self.turn_vel = random.choice([0.3, -0.3])  # Randomly turn left or right
