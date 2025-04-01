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

        self.AvoidRange = None
        self.avoid_distance = 0.20 
        self.stateCounter = 0
        self.state = "SEARCHING"  
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
        self.state = None
        self.contourArea = None
        self.bottom_contours = None
        self.searchCounter = 0
        self.wanderCounter = 0



    def camera_callback(self, data):
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

        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cy = int(M['m01'] / M['m00'])
                self.contourArea = cv2.contourArea(contour)
                
                if cy > image_center_y and self.contourArea > 350:  
                    cx = int(M['m10'] / M['m00'])
                    distance_to_center = abs(cx - image_center_x)
                    self.bottom_contours.append((contour, distance_to_center))
                else:
                    if self.contourArea > 2400:
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
        # print(f"Top color (BGR) for contour:    {self.top_color}")
        # print(f"Bottom color (BGR) for contour: {self.bottom_color}")
        self.stateCounter+=1
        if self.stateCounter > 2000:
            self.stateCounter=0
        if self.laser_scan is None:  
            self.forward_vel = 0.0
            self.turn_vel = 0.0
            print("Waiting for laser data...")
            return 
        if self.AvoidRange is not None:
            # print("Obstacle distance: ",self.avoid_distance)
            min_distance = min(self.AvoidRange)
            fullRangeObstacle = min(self.fullRange)
            proximityCheck = min_distance < self.avoid_distance
            if proximityCheck:
                self.turnCounter=0
            # print("Minimum distance: ",min_distance)
            print(f"| Turn counter: {self.turnCounter} | Search counter: {self.searchCounter} | Wander counter: {self.wanderCounter} | State: {self.state} | Closest Obstacle: {fullRangeObstacle:.2f}m | Area: {self.contourArea} |")   
        blocks_in_front = False
        if self.bottom_contours and self.current_frame is not None:
            image_center_x = self.current_frame.shape[1] // 2
            center_range = self.current_frame.shape[1] // 3  
            for contour, _ in self.bottom_contours:
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    if image_center_x - center_range <= cx <= image_center_x + center_range:
                        blocks_in_front = True
                        break       
        if self.bottom_contours is not None and blocks_in_front is not None:
         print(f"| Bottom contours count: {len(self.bottom_contours)} | Blocks in front: {blocks_in_front} |")

        if self.state == "Wandering"  and blocks_in_front:
            self.state= "Stop wandering"

        if proximityCheck or self.searchCounter<200 or self.state == "Stop wandering":
            self.state = "Searching" 
            self.searchCounter+=1
            self.wanderCounter=0
            self.searcher()  
        elif not blocks_in_front and self.wanderCounter<200 and (self.state != "Pushing"):
            self.state= "Wandering"
            self.wanderCounter+=1
            self.turn_vel= 0.0
            self.forward_vel = 0.2  
            if self.searchCounter>250:
                self.searchCounter=0
           
        if self.target_centered and not proximityCheck :
            self.pushCounter+=1
            self.turn_vel= 0.0          
            self.state="Pushing"
            self.pushBack = True
            self.turnCounter = 0
            self.stateCounter = 0 
            if self.cx<self.current_frame.shape[1] / 3: 
                print("Turning left")
                self.turn_vel= 0.1
                self.forward_vel=0.1 
            elif self.cx>2 * self.current_frame.shape[1] / 3: 
                print("Turning Right")
                self.turn_vel= -0.1
                self.forward_vel=0.1 
            else: 
                self.turn_vel= 0.0 
                self.forward_vel=0.1
        elif not self.target_centered and self.state=="Pushing" and self.pushCounter>50:
            self.state = "Pushing"
            print("Extra push")
            print(self.pushCounter)
            self.turn_vel=0.0
            self.forward_vel=0.2
            self.pushBackCounter+=1
            if self.pushBackCounter>50:
                print("Reversing")
                self.turn_vel=0.0
                self.forward_vel=-0.1 
        if self.pushBackCounter> 50:
            self.pushBackCounter=0
            self.state= "Wandering"
           

            
           
                
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
        self.laser_scan = data
        if self.laser_scan is not None:
            rangeMin = self.laser_scan.angle_min
            rangeMax = self.laser_scan.angle_max
            increment = self.laser_scan.angle_increment
            middle = (rangeMin + rangeMax) / 2
            forward_index = int((middle - rangeMin) / increment)
            self.AvoidRangeNarrow = self.laser_scan.ranges[forward_index+40:forward_index-40:-1] 
            self.AvoidRange = self.laser_scan.ranges[forward_index+50:forward_index-50:-1]            
            self.avoidLeft = self.laser_scan.ranges[forward_index:]
            self.avoidRight = self.laser_scan.ranges[:forward_index]
            self.fullRange = self.laser_scan.ranges[:]
            self.leftMeanAvoid = np.mean(self.avoidLeft)  
            self.rightMeanAvoid = np.mean(self.avoidRight)  


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




# if self.cx<self.current_frame.shape[1] / 3: 
#     self.turn_vel= 0.2 self.forward_vel=0.1 
# elif self.cx>2 * self.current_frame.shape[1] / 3: 
#     self.turn_vel= +0.2 self.forward_vel=0.1 
# else: 
#     self.turn_vel= 0.0 self.forward_vel=0.1

 # # p controller adapted from workshop materials https://github.com/LCAS/teaching/blob/2425-devel/src/cmp3103m_ros2_code_fragments/cmp3103m_ros2_code_fragments/robot_feedback_control_todo.py
            # # Define proportional gains (tune these values based on performance)
            # k_p_turn = 0.01  # Adjust for turning speed
            # k_p_forward = 0.001  # Adjust for forward speed

            # # Compute error based on position in the frame
            # error_x = self.cx - (self.current_frame.shape[1] / 2)

            # # Apply proportional control
            # self.turn_vel = k_p_turn * error_x  # Turns proportionally to error_x
            # self.forward_vel = 0.05 + k_p_forward * abs(error_x)  # Base forward velocity + small proportional adjustment

            # # Ensuring turn velocity is within reasonable limits
            # self.turn_vel = max(min(self.turn_vel, 0.1), -0.1)  # Clamp turn velocity between -0.5 and 0.5






    # def timer_callback(self):     
    #     if self.laser_scan is None:  # Check if laser data is available
    #         self.forward_vel = 0.0
    #         self.turn_vel = 0.0
    #         print("Waiting for laser data...")
    #         return
    #     print(f"Push Counter: {self.pushCounter} State counter: {self.stateCounter}, Turn Counter: {self.turnCounter}, Left average space: {self.leftMeanAvoid:.2f}, Right average space: {self.rightMeanAvoid:.2f}, Dice Area: {self.contourArea}, Top colour: {self.top_color}, Bottom colour: {self.bottom_color}")

    #     # Obstacle detection with better front range
    #     obstacle_detected = False
    #     if self.AvoidRange is not None:
    #         min_distance = min(self.AvoidRange)
    #         if min_distance < self.avoid_distance:
    #             obstacle_detected = True
    #             print(f"Obstacle detected! Min distance: {min_distance:.2f}m")
        
    #     if obstacle_detected:
    #         # Stop forward motion and turn away
    #         self.forward_vel = 0.0  # Stop moving forward 
    #         if self.turnCounter == 0:
    #             if np.mean(self.avoidLeft)  > np.mean(self.avoidRight):
    #                 self.turnDirLock=True
    #                 self.turnDir =0.3
    #                 self.turnCounter+=1
    #             else:
    #                 self.turnDirLock=True
    #                 self.turnDir =-0.3
    #                 self.turnCounter+=1
    #         if self.turnCounter<500:
    #             self.turn_vel= self.turnDir
    #             self.turnCounter+=1
    #             if self.turnDir==0.3:
    #                 print("Turning left")
    #             else:
    #                 print("Turning Right")
    #         else:
    #             self.turnCounter=0        
    #         self.state = "SEARCHING"

    #     if self.state == "SEARCHING":
    #         self.forward_vel = 0.0
    #         if self.stateCounter >500:
    #             self.state= "ROAMING"
    #         print(self.state)
    #         self.stateCounter+=1
    #         if (self.target_in_view 
    #             and self.target_centered 
    #             and ((self.bottom_color is not None and self.top_color is not None)  # Check colors exist
    #                 and ((self.bottom_color[1] == 102 and self.top_color[1] == 102)  # Green condition
    #                     or 
    #                     (self.bottom_color[2] == 102 and self.top_color[2] > 200)))):  # Red condition
    #             self.state = "PUSHING"
    #         else:
    #             if self.turnCounter == 0:
    #                 if np.mean(self.avoidLeft)  > np.mean(self.avoidRight):
    #                     self.turnDirLock=True
    #                     self.turnDir =0.3
    #                     self.turnCounter+=1
    #                 else:
    #                     self.turnDirLock=True
    #                     self.turnDir =-0.3
    #                     self.turnCounter+=1
    #             if self.turnCounter<500:
    #                 self.turn_vel= self.turnDir
    #                 self.turnCounter+=1
    #                 if self.turnDir==0.3:
    #                     print("Turning left")
    #                 else:
    #                     print("Turning Right")
    #             else:
    #                 self.turnCounter=0
    #     elif self.state == "PUSHING":
    #         self.stateCounter=0
    #         if ((self.bottom_color is not None and self.top_color is not None)  # Check colors exist
    #                 and ((self.bottom_color[1] == 102 and self.top_color[1] == 102)  # Green condition
    #                 or  (self.bottom_color[2] == 102 and self.top_color[2] > 200))):
    #             print(self.bottom_color)
    #             if self.cx>self.current_frame.shape[1] / 3:
    #                 self.turn_vel= -0.3
    #                 self.forward_vel=0.2
    #                 print("Adjusting")                   
    #             elif self.cx<2 * self.current_frame.shape[1] / 3:
    #                 self.turn_vel= 0.3
    #                 self.forward_vel=0.2
    #                 print("Adjusting")
    #             else:
    #                 self.forward_vel=0.2
    #                 print(self.state)
    #         else:                
    #             self.turn_vel = 0.0
    #             self.pushCounter +=1
    #             if self.pushCounter>200:
    #                 self.state = "SEARCHING"
    #                 self.pushCounter =0
    #     elif self.state == "ROAMING":   
    #         self.turnCounter=0         
    #         print(self.state)
    #         self.stateCounter+=1
    #         if min(self.AvoidRange) > self.avoid_distance and self.contourArea < 200:
    #             if self.cx<self.current_frame.shape[1] / 3:
    #                 self.forward_vel=0.2
    #                 self.turn_vel= -0.3
    #             elif self.cx>2 * self.current_frame.shape[1] / 3:
    #                 self.forward_vel=0.2
    #                 self.turn_vel= 0.3
    #             else:
    #                 self.forward_vel=0.2   
    #     else:
    #         self.forward_vel=0.2

            
    #     if self.stateCounter >10000:
    #         print("All objects pushed.")
    #         rclpy.shutdown()  