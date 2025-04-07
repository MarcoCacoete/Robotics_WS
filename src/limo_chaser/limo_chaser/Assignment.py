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

        self.mode = mode
        self.get_logger().info(f"Mode: {self.mode}")
        cameraTopic = '/camera/color/image_raw' if mode == 'bot' else '/limo/depth_camera_link/image_raw'
        
        self.turn_vel = 0.0
        self.forward_vel = 0.0
        self.pub_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 10)
        timer_period = 0.1  # 0.1 seconds per tick
        self.create_subscription(Image, cameraTopic, self.camera_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.laser_scan_callback, 10)
        self.br = CvBridge()

        # All the varibles I used, some are not used but kept for possible use.
        self.AvoidRange = None # Angle range shaped like a cone in front of robot so that it may navigate obstacles with less strict checks.
        self.avoid_distance = 0.20 # The minimum distance allowed between robot and obstacles. 20cm
        self.stateCounter = 0
        self.state = "Searching"  # Initial state assigned so that the robot has a starting state.
        self.target_in_view = False
        self.target_centered = False
        self.laser_scan = None
        self.top_color = None
        self.bottom_color = None
        self.turnCounter=0   # I used counters to try to keep memory intact when using some of the callbacks. 
        self.turnDir=0.0 # These two are essential for the program to decide which direction is best to start spinning in case of obstacles
        self.leftMeanAvoid = None  # It picks the side with most space.
        self.rightMeanAvoid = None
        self.contourArea=None  # Area for contours, essential to differentiate between closest and furthest away, to prioritize targeting. 
        self.pushCounter =0 # This counter allows the robot to keep pushing for a bit even after losing track of centered block. Also used to determine states.
        self.pushBackCounter =0
        self.centered = False # Checks boolean for if target block is centered with target same coloured rectangle.
        self.bottom_contours = None # Vector containing only bottom contours so that I can differentiate between top and bottom shapes.
        self.searchCounter = 0 # Counter to keep search going for a while before robot tries to wander around finding open gaps between obstacles and blocks.
        self.wanderCounter = 0 # Limits how long the robot is allowed to wander looking for new openings to target blocks.
        self.batteryCapacity = 10000 # My implementation of an end condition so that the robot can return to base after battery is drained below 25% and shutdown. 
                                     # It is working, the robot does successfuly navigate to green rectangle and after a while shuts down.
                                     # From research limo bot has 10000mah battery, I decrement this value in my callbacks to simulate 0.01% battery drain.
                                     # Wheel movement drains more battery (0.04) than laser or camera. I didn't find a battery topic I could use in the simulation so this was my solution.
                                     # I understand the real robot has a topic for battery drain that I could use for this purpose.

    def camera_callback(self, data):
        self.batteryCapacity -=0.01 # One of the decrements of the battery capacity.

        # A lot of this code below was found on the workshops repo, most specifically, the colour chaser 2 file, I adapted it for my use or added to it.
        # Also learned from https://coderivers.org/blog/python-cv2-in-range/
        self.current_frame = self.br.imgmsg_to_cv2(data, desired_encoding='bgr8')   # Conversion of ros image to opencv, in BGR (Blue, Green, Red) colour space.
        current_frame_hsv = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV)     # This further converts the BGR image to HSV colour space to make segmentation easier.
        red_lower1 = cv2.inRange(current_frame_hsv, (0, 100, 50), (10, 255, 255))   # Mask for red hues ranged between 0-10
        red_lower2 = cv2.inRange(current_frame_hsv, (160, 100, 50), (180, 255, 255))# Same but for upper range 160-180
        red_mask = red_lower1 + red_lower2                                          # Joins both masks for thresholding 
        green_mask = cv2.inRange(current_frame_hsv, (40, 100, 50), (80, 255, 255))
        
        current_frame_mask = red_mask + green_mask                                  # Detect both green and red.
        contours, _ = cv2.findContours(current_frame_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Obtains contours on mask

        image_center_x = self.current_frame.shape[1] // 2  # Determines middle of frame by dividing it in half, [1] for width [0] for height
        image_center_y = self.current_frame.shape[0] // 2  
        self.bottom_contours = [] 
        top_contour = None
        self.top_color = None
        self.contourArea = 0


        if self.batteryCapacity/100 < 25: # This is a method I used to allow the robot to target only the wide rectangle.
            for contour in contours:      # It uses the same logic as the lower block where it checks for contours relative to their location vertically, it assigns vertical contours as top.
                M = cv2.moments(contour)  # However in this first block it "fools" the robot by matching rectangle to rectangle so that it only targets the top shape which will always be centered.
                if M['m00'] > 0:          # This allows it to navigate to it before losing sight of it which triggers my shutdown condition procedure.
                    cy = int(M['m01'] / M['m00']) # This calculates spatial moments,  re-used from colour chaser code from woskhops with tweaks.
                    if cy <= image_center_y:  # If centroid is above image center or borderline it's a top contour. This separation allows me to use top rectangles as "binary traffic lights"
                        top_contour = contour # further down in code if their colour matches blocks colours.
                        break
            if top_contour is not None:
                M = cv2.moments(top_contour)
                if M['m00'] > 0:                  # Checks if area is not 0 to proceed
                    cx = int(M['m10'] / M['m00']) # Horizontal centroid
                    self.bottom_contours = [(top_contour, abs(cx - image_center_x))]  # This is what "fools" the logic by using the top marker as a bottom marker for chasing.
        else:            
            for contour in contours:      # Exactly the same as above only the normal way it is used is to chase the blocks instead of the marker.
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cy = int(M['m01'] / M['m00'])
                    self.contourArea = cv2.contourArea(contour)
                    if cy > image_center_y and self.contourArea > 325: # Filters out some blocks that are too far away (area < 325), so it doens't always chase blocks that are already close to wall or noise.
                        cx = int(M['m10'] / M['m00'])
                        distance_to_center = abs(cx - image_center_x)
                        self.bottom_contours.append((contour, distance_to_center))
                    else:
                        if self.contourArea:           # If the contour is area but doesn't fulfill above conditions it's a top contour.
                            top_contour = contour

        if top_contour is not None:
            try:
                T = cv2.moments(top_contour)           # This block is similar to above, it calculates moments, and validates their centroids coordinates within the image boundaries and to sample colour.
                if T['m00'] > 0:                                        
                    tx = int(T['m10'] / T['m00'])      # X and Y coords
                    ty = int(T['m01'] / T['m00'])
                    if 0 <= tx < self.current_frame.shape[1] and 0 <= ty < self.current_frame.shape[0]:
                        self.top_color = np.array(self.current_frame[ty, tx]) # Samples collurs at centroid
            except Exception as e:
                self.get_logger().warn(f"Error processing top contour: {str(e)}")

        if self.bottom_contours:
            self.bottom_contours.sort(key=lambda x: x[1]) # Sorts contours by closeness to center 
            top5 = self.bottom_contours[:5]  # Gets the first 5 bottom contours from the vector
            if top5:                         
                top5.sort(key=lambda x: cv2.contourArea(x[0]), reverse=True)  # This then sorts by area so that the closest blocks are prioritised when chased, instead of the robot changing targets all the time.
                self.bottom_contours = top5 + self.bottom_contours[5:]        # Replaces the blocks in the new order.

        self.target_in_view = False          # This resets variables per call
        self.target_centered = False
        self.bottom_color = None
        self.cx = None

        if self.bottom_contours:  # Assures there are blocks before executing
            most_centered_contour = self.bottom_contours[0][0]  # Retrieves most centered largest contour
            M = cv2.moments(most_centered_contour)              # Gets area and shape as well as centroid
            if M['m00'] > 0:
                self.cx = int(M['m10'] / M['m00'])              # Obtains exact location
                cy = int(M['m01'] / M['m00'])
                self.contourArea = cv2.contourArea(most_centered_contour)
                bottom_color = np.array(self.current_frame[cy, self.cx]) # Samples colour for comparsion

                if (self.top_color is not None and                             # If the colours for top and bottom line up, for red and green,intensities.
                    ((bottom_color[1] == 102 and self.top_color[1] == 102) or  # These thresholds were picked from checking debug prints.
                    (bottom_color[2] == 102 and self.top_color[2] > 200))):  
                    self.target_in_view = True                                 # Declares both targets as in view, giving the go ahead for the pushing logic.
                    self.bottom_color = bottom_color
                    self.targetArea = self.contourArea
                    # Checks if block is in center of image by dividing it in 3 thirds, and comparing location to inner borders of outer thirds. 
                    # https://coderivers.org/blog/python-cv2-in-range/ same guide was helpful
                    self.target_centered = (image_center_x - self.current_frame.shape[1] // 3 <= self.cx <= image_center_x + self.current_frame.shape[1] // 3) 
        
        # This was obtained from the original chaser code tou output the camera to a window.
        current_frame_contours = cv2.drawContours(self.current_frame, [c[0] for c in self.bottom_contours], -1, (255, 255, 0), 3) 
        cv2.imshow("Image window", cv2.resize(current_frame_contours, (0, 0), fx=0.4, fy=0.4))
        cv2.waitKey(1)
        
        
    def timer_callback(self):
        if self.AvoidRange is not None: # This check is here to make sure the range I set has data, though after I swapped the timer callback call to inside the laser scan to try to make it work with real bot, it might not be needed.
            # print("Obstacle distance: ",self.avoid_distance)
            min_distance = min(self.AvoidRange)                  # Checks if the minimum of the scanned range is smaller than the minimum distance I set above, 
            # fullRangeObstacle = min(self.fullRange)
            proximityCheck = min_distance < self.avoid_distance  # This makes sets to True if robot gets too close to obstacle or wall.
            
        blocksBlocking = False                                   # This block finds shapes in front of the robot, to determine if the wanderer logic is allowed to triger (if there is a gap)
        if self.bottom_contours and self.current_frame is not None:# Very similar as in camera callback. Checks center third of image.
            center_range = self.current_frame.shape[1] // 3  
            for contour, _ in self.bottom_contours:
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    if center_range <= cx <= 2 * center_range:   
                        blocksBlocking = True

        if proximityCheck:                                       # Different states, that trigger depending on context.
            self.state= "Searching"                              # Searching state is default, starts a spin by the robot, turn counter allows it to spin for a while to prevent it shifting 
            self.turnCounter=0                                   # forever when one side has less space than the other as it turns
        
        if self.target_centered and not proximityCheck:          # This initiates the pushing logic if the target is centered with top and same colour, and not close to obstacles or walls.
            self.state="Pushing"
        
        if not self.target_centered and self.state == "Pushing": # This block allows the re-assignment of a state if the pushing state exits too early. Reverts to searching,
            print(self.pushBackCounter)
            self.pushBackCounter+=1
            if self.pushBackCounter>30:                          # It only does this once a block that is now out of sight has been pushed for a bit, otherwise no blocks would get pushed.
                self.state = "Reversing"
            #lif self.pushBackCounter>100:
             #   self.state = "Searching" # This is a failsafe to prevent the robot from getting stuck in a loop, it resets the counter to 0 after a while.
         
                
        if self.state == "Searching":                            # Triggers the searcher function, also resets the counter used to push further
            self.pushBackCounter = 0
            self.forward_vel = 0.0
            self.batteryCapacity -=0.01                          # Another simulated battery drain decrement
            self.searchCounter+=1                                # This counter allows search to do a large rotation before the robot triggering wandering state, so it's not always glued to the walls.
            self.searcher()
            
        # Wandering logic, it checks for gaps, if search has happened long enough and advances to areas that it otherwise wouldn't go without introducing some randomness. Prevents search loops.
        if not self.target_centered and not blocksBlocking and not proximityCheck and self.searchCounter>200 and self.pushCounter == 0: # Also resets the pushCounter
            self.state ="Wandering"
            
        if self.state == "Reversing":  # This state reverses the robot in an attempt to re-aquire the target it is meant to be pushing, 
                                       # this mitigates the tendency for it to try to push the next block in range, instead of stickign with current block for longer.
            self.forward_vel = -0.2    # Reverse speed
            self.pushBackCounter+=1    # Keeps incrementing counter to trigger the search state after a bit, so that it also doesn't reverse forever.
            if self.pushBackCounter>45:# Searching already is set to re-aquire target if it is indeed present again.
                self.forward_vel = 0.0
                self.turn_vel = 0.0
                self.state = "Searching"
                
                
        if self.state=="Pushing":                                # Pushing logic, sets initially turn speed to 0 also drains most simulated battery.
            print("Centered", self.target_centered)
            self.batteryCapacity -=0.04
            self.turn_vel= 0.0          
            first = self.current_frame.shape[1] / 3              # This is how I dynamically determined turn speed precise adjustment, initially I used the colour chaser jerky adjustment, then tried a p-controller I found on a website
            second = 2 * self.current_frame.shape[1] / 3         # I couldnt get the p-controller to work as I wanted, so I kept using the chaser controller until I had the idea to use the distance 
            if self.cx is not None:                              # to center offsets to try and extract a value between 0.0 and 0.4, I managed to do this by dividing it by 1000 and doubling the result
                distToLeft = self.cx - first
                distToRight = second - self.cx                   # The relevant speed is used depending on which way it needs to move. I was pleased that my idea worked.
                speedCalcLeft= 2*distToRight/1000
                speedCalcRight = 2*distToLeft/1000        
                print (f"Small Adjust speed Left: ,{speedCalcLeft:.2f},Small Adjust speed Right: ,{speedCalcRight:.2f}")               
                if self.cx<self.current_frame.shape[1] / 3: 
                    print("Turning left")
                    # if speedCalcLeft<0.4:                       # Initially I limited the upper limit of the speed, however by tweaking forward movement it was no longer needed.
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
                elif self.target_centered:                         # In previous different types of centering logic this sometimes didnt even trigger due to lots of jerkyness.
                    self.turn_vel= 0.0 
                    self.forward_vel=0.2

                self.bottom_contours= None
               # self.turnCounter=0# I reset contours vector again so that my shutdown condition works in the end.
                    
        if self.batteryCapacity/100 < 25 and self.searchCounter<500: # This is not a state but a block of code that I created to shutdown the robot after some counters hit some thresholds.
            self.pushCounter+=1                                      # I picked the thresholds just by experimenting, and the robot seems to have enough time to get to the green maker and wait.
            print("Returning to base. ",self.pushCounter )           # It does this because the top marker is the only object it sees at this point, so it can only navigate to it
            print("Search Counter: ",self.searchCounter)
        elif self.batteryCapacity/100 < 25 and self. searchCounter>500: # Flips when search state has hit hit the necessary threshold, then I decrement the push counter fast (-3)
            self.pushCounter-=3
            print("Bot returned to base, shutting down in: ",self.pushCounter) # At one point I was comparing == 0 so it wasnt shutting down due to me decrementing 3 instead of 1 so it didnt always hit == 0, a silly mistake.
            if self.pushCounter<=0:
                print("Robot has shutdown.") # Shuts robot down with a message. The value of battery was picked as 10000mah like the real robot battery, it can be decreased to test this feature, I had it set at 2505
                rclpy.shutdown()             # It should roughly last 2 hours, which somehow is realistic even though the whole logic is arbitrary.
                                        
        if self.state == "Wandering" :       # Wandering state logic, just pushes forward through gaps, consumes a lot of fake battery. Doesn't turn, it runs till it finds obstacles and then search kicks back in.
            self.batteryCapacity -=0.04
            self.searchCounter=0
            self.turn_vel=0.0
            self.forward_vel= 0.2
               
        print(f"| Current state: {self.state} | Blocks in front: {blocksBlocking} | Battery remaining(simulated): {self.batteryCapacity/100:.2f}% |")  # State labels. With some basic info, cut down from a more verbose version.
                
        self.tw = Twist()                     # Ros publisher commands for movement, also from colour chaser original code.
        self.tw.linear.x = float(self.forward_vel) 
        self.tw.angular.z = self.turn_vel
        self.pub_cmd_vel.publish(self.tw)

    def searcher(self):                        # This searcher function exists because initially I had 2 steps of obstacle detection and wanted to reuse the code for both.
        self.forward_vel = 0.0
        if self.turnCounter == 0:              # It checks if the counter is 0 to assign new direction left or right for a while before re-assigning, to stop it from alternating continuously.
            if np.mean(self.avoidLeft)  > np.mean(self.avoidRight): # I used the lidar by splitting its angular range to determine which side has more space available on average, and assign that direction in the form of the turn velocity.                  
                self.turnDir =0.3
            else:
                self.forward_vel = 0.0  
                self.turnDir =-0.3
        if self.turnCounter<250:               # This counter gives the robot a chance to "change its mind" and not spin always in same direction, picked a value that worked ok throughout testing. 
            self.forward_vel = 0.0  
            self.turn_vel= self.turnDir
            self.turnCounter+=1
            if self.turnDir==0.3:
                print("Search Direction Left") # Indicates which way it is turning.
            else:
                print("Search Direction Right")                                    
        else:
            self.turnCounter=0           
   
   # This laser scan callback wasn't working correctly on the real robot due to false positive negative ranges always triggering, 
   # I didn't get a chance to confirm it now works on it at the time of writing this, not sure I'll have a chance to have access to real robot again for long enough to debug if it doesn't work.
   # it also was not triggering my timer callback before, I now placed the call at the end of this to make sure it has data before triggering it, also untested on real bot.
    def laser_scan_callback(self, data):      
        self.batteryCapacity -=0.01      # Another battery drainer
        self.laser_scan = data                            # Ros laser message in full
        self.laser_scan_ranges = self.laser_scan.ranges   # Array of distance measurements
        
        # This attempts to replace the invalid values with a large enough value so that the original data is discarded
        for idx, value in enumerate(self.laser_scan_ranges):
            if value < data.range_min:                          # The data below 0 that is not useable.
                self.laser_scan_ranges[idx] = 20000
    
        if self.laser_scan_ranges is not None:                  #  Makes sure data is present
            rangeMin = self.laser_scan.angle_min                # Min and max angle ranges to calculate center of range, with increments.
            rangeMax = self.laser_scan.angle_max
            increment = self.laser_scan.angle_increment
            middle = (rangeMin + rangeMax) / 2
            forward_index = int((middle - rangeMin) / increment)# Initially I only used the value for distance directly in front of the robot, but that was just a bad initial idea.
            # self.AvoidRangeNarrow = self.laser_scan_ranges[forward_index+40:forward_index-40:-1] 
            self.AvoidRange = self.laser_scan_ranges[forward_index+50:forward_index-50:-1] # I settled on this range as a cone that, for the most part allows the robot to still be close to obstacles            
           # self.AvoidRange = self.laser_scan_ranges[:] # when it moves to try and push some blocks that are too close to retrieve. Sometimes it still gets stuck.
            self.avoidLeft = self.laser_scan_ranges[forward_index:]         # These two ranges are the ones I used to determine average space on boths sides, to pick which direction the robot should turn.   
            self.avoidRight = self.laser_scan_ranges[:forward_index]                       
            # self.fullRange = self.laser_scan_ranges[:]                    # Option for a safer range for obstacle avoidance.
            # self.leftMeanAvoid = np.mean(self.avoidLeft)                  # Some other ranges used in experimentation
            # self.rightMeanAvoid = np.mean(self.avoidRight)  
            self.timer_callback()

def main(args=None):
    print('Starting Limo Pusher.')              # I added a parameter to run the same submission for both the real bot or on gazebo, lets user add --bot to run in real bot mode.

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

