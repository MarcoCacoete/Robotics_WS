#!/usr/bin/env python3
"""
Advanced LIMO Robot Controller
- Combines object tracking, wall following, and recovery behaviors
- Uses ROS2 Humble, OpenCV, and RPLIDAR
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class LimoController(Node):
    def __init__(self):
        super().__init__('limo_controller')
        
        # === Configuration Parameters ===
        # Object Tracking
        self.target_hsv_low = np.array([0, 150, 50])    # Minimum HSV values (Orange)
        self.target_hsv_high = np.array([30, 255, 255])  # Maximum HSV values
        self.min_contour_area = 500                      # Ignore small detections (pixels)
        
        # Wall Following 
        self.wall_follow_distance = 0.4    # Ideal distance from wall (meters)
        self.safety_distance = 0.3         # Emergency stop threshold (meters)
        
        # Recovery Behavior
        self.search_distance = 0.5         # How far to search forward (meters)
        self.max_recovery_time = 4.0       # Timeout before giving up (seconds)

        # === ROS2 Communications ===
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Subscribers
        self.camera_sub = self.create_subscription(
            Image, 
            '/limo/depth_camera_link/image_raw', 
            self.process_camera_frame, 
            qos_profile=10)
            
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.process_lidar_scan,
            qos_profile=10)
        
        # === State Management ===
        self.state = "TRACKING"  # Initial state
        self.bridge = CvBridge() # OpenCV-ROS image converter
        self.last_detection_time = self.get_clock().now()

    def process_camera_frame(self, msg):
        """Process each camera frame for object detection and tracking"""
        try:
            # Convert ROS Image to OpenCV format (BGR)
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # === Color Detection Pipeline ===
            # 1. Convert to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 2. Create binary mask for target color
            mask = cv2.inRange(hsv, self.target_hsv_low, self.target_hsv_high)
            
            # 3. Find contours in the mask
            contours, _ = cv2.findContours(
                mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour by area
                largest_contour = max(contours, key=cv2.contourArea)
                
                if cv2.contourArea(largest_contour) > self.min_contour_area:
                    # === Object Tracking Calculations ===
                    # 1. Calculate contour moments (for centroid)
                    M = cv2.moments(largest_contour)
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # 2. Shape analysis
                    perimeter = cv2.arcLength(largest_contour, True)
                    approx = cv2.approxPolyDP(largest_contour, 0.02*perimeter, True)
                    shape = self.classify_shape(approx)
                    
                    # 3. Update tracking data
                    self.tracking_data = {
                        'centroid': (cx, cy),
                        'area': cv2.contourArea(largest_contour),
                        'shape': shape
                    }
                    self.last_detection_time = self.get_clock().now()
                    self.state = "TRACKING"
                    
                    # === Visualization ===
                    cv2.drawContours(frame, [largest_contour], -1, (0,255,0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)
                    cv2.putText(frame, f"{shape}", (cx-20, cy-20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            
            cv2.imshow("Tracking View", frame)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Camera processing error: {str(e)}")

    def classify_shape(self, contour):
        """Determine if contour is cube-like or rectangular"""
        # Get rotated bounding rectangle
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        aspect_ratio = max(width, height) / min(width, height)
        
        if len(contour) == 4:  # Quadrilateral
            return "CUBE" if 0.9 < aspect_ratio < 1.1 else f"RECTANGLE (AR={aspect_ratio:.1f})"
        return "UNKNOWN"

    def process_lidar_scan(self, msg):
        """Process LiDAR data for navigation and safety"""
        self.lidar_ranges = msg.ranges
        self.front_distance = min(msg.ranges[:30])  # Check 30° frontal cone
        
        # Emergency stop if obstacle too close
        if self.front_distance < self.safety_distance:
            self.emergency_stop()

    def execute_behavior(self):
        """Main control loop decision maker"""
        cmd = Twist()
        
        # === State Machine Logic ===
        if self.state == "TRACKING":
            if hasattr(self, 'tracking_data'):
                # Object tracking control
                error_x = self.tracking_data['centroid'][0] - 320  # Center of 640px image
                cmd.angular.z = -0.01 * error_x  # P-control for centering
                
                # Dynamic speed based on object size
                target_area = 25000
                area_ratio = min(1.0, self.tracking_data['area'] / target_area)
                cmd.linear.x = 0.2 * (1 - area_ratio)  # Slow down as object gets larger
            else:
                # Lost object - switch to recovery
                self.state = "RECOVERY"
                self.recovery_start_time = self.get_clock().now()
                
        elif self.state == "RECOVERY":
            elapsed = (self.get_clock().now() - self.recovery_start_time).nanoseconds / 1e9
            
            if elapsed < self.max_recovery_time / 2:
                # Phase 1: Search forward
                cmd.linear.x = 0.15
            else:
                # Phase 2: Search while reversing
                cmd.linear.x = -0.1
                cmd.angular.z = 0.3  # Gentle rotation
                
            if elapsed > self.max_recovery_time:
                # Fallback to wall following
                self.state = "WALL_FOLLOW"
                
        elif self.state == "WALL_FOLLOW":
            # Left-wall following
            left_scan = min(self.lidar_ranges[90:135])  # 45° left sector
            error = self.wall_follow_distance - left_scan
            cmd.angular.z = 0.8 * error  # P-control
            cmd.linear.x = 0.2
            
        self.cmd_vel_pub.publish(cmd)

    def emergency_stop(self):
        """Immediately halt all movement"""
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)
        self.get_logger().warn("EMERGENCY STOP ACTIVATED")

def main(args=None):
    rclpy.init(args=args)
    controller = LimoController()
    
    # Use a timer for control loop (10Hz)
    controller.create_timer(0.1, controller.execute_behavior)
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()