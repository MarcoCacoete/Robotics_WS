#!/usr/bin/env python3
"""
LIMO 360° Green Cube Scanner
- Completes full rotation while detecting objects
- Only reports green cubes that meet shape criteria
- Outputs final report with angles of detection
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import defaultdict

class CubeScanner(Node):
    def __init__(self):
        super().__init__('cube_scanner')
        
        # === Configuration ===
        self.scan_speed = 0.5  # rad/s rotation speed
        self.target_color_low = np.array([40, 50, 50])   # HSV green range
        self.target_color_high = np.array([80, 255, 255])
        self.min_cube_area = 1000  # minimum pixels to qualify as cube
        self.aspect_ratio_range = (0.8, 1.2)  # width/height for cubes
        
        # === Tracking Variables ===
        self.current_angle = 0.0
        self.start_angle = None
        self.scan_complete = False
        self.detected_cubes = defaultdict(list)  # {angle: [cube_data]}
        
        # === ROS Setup ===
        self.bridge = CvBridge()
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Subscribers
        self.create_subscription(Image, '/limo/depth_camera_link/image_raw', 
                               self.camera_callback, 10)
        self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        
        # Start continuous rotation
        self.start_rotation()

    def start_rotation(self):
        """Begin continuous rotation"""
        twist = Twist()
        twist.angular.z = self.scan_speed
        self.cmd_vel_pub.publish(twist)
        self.start_angle = self.current_angle
        self.get_logger().info("Starting 360° scan for green cubes...")

    def imu_callback(self, msg):
        """Track rotation angle using IMU data"""
        # Simple integration - replace with proper quaternion conversion in real use
        self.current_angle += msg.angular_velocity.z * 0.1  # dt ≈ 0.1
        
        # Normalize angle and check for full rotation
        normalized_angle = self.current_angle % (2*np.pi)
        if not self.scan_complete and abs(normalized_angle - self.start_angle) < 0.1 and len(self.detected_cubes) > 0:
            self.complete_scan()

    def camera_callback(self, msg):
        """Detect and record green cubes"""
        if self.scan_complete:
            return
            
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Create mask for green color
            mask = cv2.inRange(hsv, self.target_color_low, self.target_color_high)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_cube_area:
                    # Check if shape is cube-like
                    rect = cv2.minAreaRect(contour)
                    width, height = rect[1]
                    aspect_ratio = max(width, height) / min(width, height)
                    
                    if self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]:
                        # Valid cube detected
                        angle = self.current_angle % (2*np.pi)
                        self.detected_cubes[angle].append({
                            'angle_deg': np.degrees(angle),
                            'area': area,
                            'contour': contour
                        })
                        
                        # Visual feedback (no pausing)
                        cv2.drawContours(frame, [contour], -1, (0,255,0), 2)
                        cv2.putText(frame, f"Cube @ {np.degrees(angle):.1f}°",
                                  (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            cv2.imshow("Cube Scanner", frame)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Detection error: {str(e)}")

    def complete_scan(self):
        """Stop rotation and report findings"""
        self.scan_complete = True
        self.cmd_vel_pub.publish(Twist())  # Stop robot
        
        self.get_logger().info("\n=== SCAN COMPLETE ===")
        self.get_logger().info(f"Detected {sum(len(v) for v in self.detected_cubes.values())} green cubes:")
        
        for angle, cubes in sorted(self.detected_cubes.items()):
            for i, cube in enumerate(cubes):
                self.get_logger().info(
                    f"Cube {i+1} at {cube['angle_deg']:.1f}° (Size: {cube['area']} px)")
        
        cv2.destroyAllWindows()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    scanner = CubeScanner()
    rclpy.spin(scanner)
    scanner.destroy_node()

if __name__ == '__main__':
    main()