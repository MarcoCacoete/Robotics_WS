#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from threading import Lock
import sys

class ColourChaser(Node):
    def __init__(self, mode):
        super().__init__('colour_chaser')

        self.declare_parameter('mode', 'gazebo')
        self.mode = self.get_parameter('mode').value
        self.get_logger().info(f"Mode: {self.mode}")
        cameraTopic = '/camera/color/image_raw' if mode == 'bot' else '/limo/depth_camera_link/image_raw'
        
        self.turn_vel = 0.0
        self.forward_vel = 0.0
        self.pub_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 10)
        timer_period = 0.1  # 0.1 seconds per tick
        self.create_subscription(Image, cameraTopic, self.camera_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.laser_scan_callback, 10)
        self.br = CvBridge()

        self.AvoidRange = None
        self.avoid_distance = 0.20 
        self.stateCounter = 0
        self.state = "Searching"  
        self.target_in_view = False
        self.target_centered = False
        self.laser_scan = None
        self.top_color = None
        self.bottom_color = None
        self.turnCounter=0
        self.turnDir=0.0
        self.leftMeanAvoid = None  
        self.rightMeanAvoid = None
        self.contourArea=None
        self.pushCounter =0
        self.pushBack =False
        self.pushBackCounter =0
        self.centered = False
        self.contourArea = None
        self.bottom_contours = None
        self.searchCounter = 0
        self.wanderCounter = 0
        self.batteryCapacity = 10000



    def camera_callback(self, data):
        self.batteryCapacity -=0.01
        self.current_frame = self.br.imgmsg_to_cv2(data, desired_encoding='bgr8')
        current_frame_hsv = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV)

        red_lower1 = cv2.inRange(current_frame_hsv, (0, 100, 50), (10, 255, 255))
        red_lower2 = cv2.inRange(current_frame_hsv, (160, 100, 50), (180, 255, 255))
        red_mask = red_lower1 + red_lower2
        green_mask = cv2.inRange(current_frame_hsv, (40, 100, 50), (80, 255, 255))
        
        current_frame_mask = red_mask + green_mask
        contours, _ = cv2.findContours(current_frame_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        image_center_x = self.current_frame.shape[1] // 2  
        image_center_y = self.current_frame.shape[0] // 2  
        self.bottom_contours = [] 
        top_contour = None
        self.top_color = None
        self.contourArea = 0


        if self.batteryCapacity/100 < 25:
            # Low battery: only look for top contour
            for contour in contours:
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cy = int(M['m01'] / M['m00'])
                    if cy <= image_center_y:  # Top contour only
                        top_contour = contour
                        break
            if top_contour is not None:
                M = cv2.moments(top_contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    self.bottom_contours = [(top_contour, abs(cx - image_center_x))]
        else:
            
            for contour in contours:
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cy = int(M['m01'] / M['m00'])
                    self.contourArea = cv2.contourArea(contour)
                    if cy > image_center_y and self.contourArea > 325:
                        cx = int(M['m10'] / M['m00'])
                        distance_to_center = abs(cx - image_center_x)
                        self.bottom_contours.append((contour, distance_to_center))
                    else:
                        if self.contourArea:  
                            top_contour = contour

        if top_contour is not None:
            try:
                T = cv2.moments(top_contour)
                if T['m00'] > 0:
                    tx = int(T['m10'] / T['m00'])
                    ty = int(T['m01'] / T['m00'])
                    if 0 <= tx < self.current_frame.shape[1] and 0 <= ty < self.current_frame.shape[0]:
                        self.top_color = np.array(self.current_frame[ty, tx])
            except Exception as e:
                self.get_logger().warn(f"Error processing top contour: {str(e)}")

        if self.bottom_contours:
            self.bottom_contours.sort(key=lambda x: x[1]) 
            top_3 = self.bottom_contours[:5]  # Get the first 3 (or fewer if less than 3)
            if top_3:  # Ensure there’s at least one contour
                top_3.sort(key=lambda x: cv2.contourArea(x[0]), reverse=True)  # Sort by area, descending
                # Replace the first 3 in the original list with the sorted ones
                self.bottom_contours = top_3 + self.bottom_contours[5:]

        self.target_in_view = False
        self.target_centered = False
        self.bottom_color = None
        self.cx = None

        if self.bottom_contours:  
            most_centered_contour = self.bottom_contours[0][0]  
            M = cv2.moments(most_centered_contour)
            if M['m00'] > 0:
                self.cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                self.contourArea = cv2.contourArea(most_centered_contour)
                bottom_color = np.array(self.current_frame[cy, self.cx])

                if (self.top_color is not None and
                    ((bottom_color[1] == 102 and self.top_color[1] == 102) or  
                    (bottom_color[2] == 102 and self.top_color[2] > 200))):  
                    self.target_in_view = True
                    self.bottom_color = bottom_color
                    self.targetArea = self.contourArea
                    self.target_centered = (image_center_x - self.current_frame.shape[1] // 3 <= self.cx <= image_center_x + self.current_frame.shape[1] // 3)
        

        current_frame_contours = cv2.drawContours(self.current_frame, [c[0] for c in self.bottom_contours], -1, (255, 255, 0), 3)
        cv2.imshow("Image window", cv2.resize(current_frame_contours, (0, 0), fx=0.4, fy=0.4))
        cv2.waitKey(1)
        
        
    def timer_callback(self):
        if self.AvoidRange is not None:
            # print("Obstacle distance: ",self.avoid_distance)
            min_distance = min(self.AvoidRange)
            # fullRangeObstacle = min(self.fullRange)
            proximityCheck = min_distance < self.avoid_distance
            
        blocksBlocking = False
        if self.bottom_contours and self.current_frame is not None:
            center_range = self.current_frame.shape[1] // 3  
            for contour, _ in self.bottom_contours:
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    if center_range <= cx <= 2 * center_range:   
                        blocksBlocking = True

        if proximityCheck:
            self.state= "Searching"
            self.turnCounter=0
        
        if self.target_centered and not proximityCheck:
            self.state="Pushing"
        
        if not self.target_centered and self.state == "Pushing":
            print(self.pushBackCounter)
            self.pushBackCounter+=1
            if self.pushBackCounter>20:
                self.state = "Searching"


        if self.state == "Searching":
            self.pushBackCounter = 0
            self.batteryCapacity -=0.01
            self.searchCounter+=1            
            self.searcher()
            


        if not self.target_centered and not blocksBlocking and not proximityCheck and self.searchCounter>150 and self.pushCounter == 0:
            self.state ="Wandering"

        if self.state=="Pushing":
            print("Centered", self.target_centered)
            self.batteryCapacity -=0.04
            self.turn_vel= 0.0          
            first = self.current_frame.shape[1] / 3
            second = 2 * self.current_frame.shape[1] / 3
            # print("Blcok Location: ",self.cx,"First 3rd: ",first,"Second 3rd: ",second)
            if self.cx is not None:            
                distToLeft = self.cx - first
                distToRight = second - self.cx 
                speedCalcLeft= 2*distToRight/1000
                speedCalcRight = 2*distToLeft/1000        
                print (f"Small Adjust speed Left: ,{speedCalcLeft:.2f},Small Adjust speed Right: ,{speedCalcRight:.2f}")               
                if self.cx<self.current_frame.shape[1] / 3: 
                    print("Turning left")
                    # if speedCalcLeft<0.4:
                    self.turn_vel= speedCalcLeft
                    self.forward_vel=0.1 
                    # else:
                    #     self.turn_vel= 0.4
                elif self.cx>2 * self.current_frame.shape[1] / 3: 
                    print("Turning Right")
                    # if speedCalcRight <0.4:
                    self.turn_vel= -speedCalcRight
                    self.forward_vel=0.1
                    # else:
                    #     self.turn_vel= -0.4               
                elif self.target_centered: 
                    self.turn_vel= 0.0 
                    self.forward_vel=0.2

                self.bottom_contours= None
                    
        if self.batteryCapacity/100 < 25 and self.searchCounter<500:
            self.pushCounter+=1
            print("Returning to base. ",self.pushCounter )
            print("Search Counter: ",self.searchCounter)
        elif self.batteryCapacity/100 < 25 and self. searchCounter>500:
            self.pushCounter-=3
            print("Bot returned to base, shutting down in: ",self.pushCounter)
            if self.pushCounter<=0:
                print("Robot has shutdown.")
                rclpy.shutdown()               
               
                                        
        if self.state == "Wandering" :
            self.batteryCapacity -=0.04
            self.searchCounter=0
            self.turn_vel=0.0
            self.forward_vel= 0.2
               
        print(f"| Current state: {self.state} | Blocks in front: {blocksBlocking} | Battery remaining(simulated): {self.batteryCapacity/100:.2f}% |")  
                
        self.tw = Twist()
        self.tw.linear.x = float(self.forward_vel)
        self.tw.angular.z = self.turn_vel
        self.pub_cmd_vel.publish(self.tw)
        
        


    def searcher(self):
        self.forward_vel = 0.0
        if self.turnCounter == 0:
            if np.mean(self.avoidLeft)  > np.mean(self.avoidRight):                    
                self.turnDir =0.3
            else:
                self.forward_vel = 0.0  
                self.turnDir =-0.3
        if self.turnCounter<250:
            self.forward_vel = 0.0  
            self.turn_vel= self.turnDir
            self.turnCounter+=1
            if self.turnDir==0.3:
                print("Search Direction Left")
            else:
                print("Search Direction Right")
                                    
        else:
            self.turnCounter=0           
    
   
    def laser_scan_callback(self, data):      
        self.batteryCapacity -=0.01
        print('here')
        self.laser_scan = data
        self.laser_scan_ranges = self.laser_scan.ranges
        
        for idx, value in enumerate(self.laser_scan_ranges):
            if value < data.range_min:
                self.laser_scan_ranges[idx] = 20000
    
        if self.laser_scan_ranges is not None:
            rangeMin = self.laser_scan.angle_min
            rangeMax = self.laser_scan.angle_max
            increment = self.laser_scan.angle_increment
            middle = (rangeMin + rangeMax) / 2
            forward_index = int((middle - rangeMin) / increment)
            # self.AvoidRangeNarrow = self.laser_scan_ranges[forward_index+40:forward_index-40:-1] 
            self.AvoidRange = self.laser_scan_ranges[forward_index+50:forward_index-50:-1]            
            self.avoidLeft = self.laser_scan_ranges[forward_index:]
            self.avoidRight = self.laser_scan_ranges[:forward_index]
            # self.fullRange = self.laser_scan_ranges[:]
            # self.leftMeanAvoid = np.mean(self.avoidLeft)  
            # self.rightMeanAvoid = np.mean(self.avoidRight)  
            self.timer_callback()
        



def main(args=None):
    print('Starting Limo Pusher.')

    botMode= "--bot" in sys.argv

    if botMode:
        sys.argv.remove("--bot")
    
    cv2.startWindowThread()
    rclpy.init(args=sys.argv)
    colour_chaser = ColourChaser(mode="bot" if botMode else "gazebo")
    rclpy.spin(colour_chaser)
    colour_chaser.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

