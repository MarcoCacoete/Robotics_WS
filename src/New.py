def camera_callback(self, data):
        current_frame = self.br.imgmsg_to_cv2(data, desired_encoding='bgr8')
        current_frame_hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
        # if self.directionCounter > 50 and self.leftTurn==True:
        #     self.directionCounter=0
        #     self.leftTurn=False
        # elif self.directionCounter > 50 and self.leftTurn==False:
        #     self.directionCounter=0
        #     self.leftTurn=True

        # if self.leftTurn==True:
        #     self.turnDirVal=0.3
        # else:
        #     self.turnDirVal=-0.3
        # print(self.directionCounter)
        # Define color-specific masks
        red_lower1 = cv2.inRange(current_frame_hsv, (0, 100, 50), (10, 255, 255))
        red_lower2 = cv2.inRange(current_frame_hsv, (160, 100, 50), (180, 255, 255))
        red_mask = red_lower1 + red_lower2
        green_mask = cv2.inRange(current_frame_hsv, (40, 100, 50), (80, 255, 255))
        
        # Combine for general detection, or use separately
        current_frame_mask = red_mask + green_mask  # Adjust based on your goal
        contours, _ = cv2.findContours(current_frame_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        image_center_y = current_frame_hsv.shape[0] // 2
        offset_y = current_frame_hsv.shape[0] // 3
        bottom_contours = []
        top_contours = []

        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cy = int(M['m01'] / M['m00'])
                if cy > image_center_y and cy < image_center_y + offset_y and cv2.contourArea(contour) > 40:
                    bottom_contours.append(contour)
                else:
                    top_contours.append(contour)

        if len(top_contours) > 0:
            T = cv2.moments(top_contours[0])
            if T['m00'] > 0:
                tx = int(T['m10'] / T['m00'])
                ty = int(T['m01'] / T['m00'])
                self.top_color = np.array(current_frame[ty, tx])
                print(f"Upper color (BGR): {self.top_color}")

        if len(bottom_contours) > 0:        

            self.target_in_view = True
            M = cv2.moments(bottom_contours[0])
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                self.bottom_color = np.array(current_frame[cy, cx])
                print(f"Lower color (BGR): {self.bottom_color}")
                self.target_centered = (data.width / 3 <= cx <= 2 * data.width / 3)
                self.targetArea = cv2.contourArea(bottom_contours[0])
                print("Area", self.targetArea)
                x, y, w, h = cv2.boundingRect(bottom_contours[0])
                print(f"Contour width: {w}, height: {h}")
                # if self.target_centered:
                #     if self.bottom_color[1]==102 and self.top_color[1]==102:
                #         self.forward_vel = 0.1
                #     elif self.bottom_color[2]==102 and self.top_color[2]>200:
                #         self.forward_vel = 0.1
                #     else:
                #         self.forward_vel = 0.0
                #         self.turn_vel=self.turnDirVal
                #         self.directionCounter +=1
                # else:
                #     self.forward_vel = 0.0
                #     self.turn_vel=self.turnDirVal
                #     self.directionCounter +=1
        else:
            self.target_in_view = False
            self.target_centered = False
            # if self.frontAvoidRange is not None:
            #     min_distance = min(self.frontAvoidRange)
            #     if min_distance < self.avoid_distance:
            #         self.forward_vel =0.0
            #         self.turn_vel=self.turnDirVal
            #         self.directionCounter +=1
            #         print(f"Obstacle detected! Min distance: {min_distance:.2f}m")
            #         print(self.turnDirVal)
            #     else:
            #         self.turn_vel=0.0
            #         self.forward_vel =0.1        
        # Draw contours
        current_frame_contours = cv2.drawContours(current_frame, bottom_contours, -1, (255, 255, 0), 3)
        cv2.imshow("Image window", cv2.resize(current_frame_contours, (0, 0), fx=0.4, fy=0.4))
        cv2.waitKey(1)