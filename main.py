import cv2
import mediapipe as mp
import numpy as np
import time
from cvzone.ClassificationModule import Classifier
from fingersUpModule import HandDetector
from djitellopy import Tello
from threading import Thread


# Tello code 

tello = Tello()

tello.connect()
tello.streamon()


#initialize the video capture

cap = cv2.VideoCapture(0)
# Get the frame reader
frame_reader = tello.get_frame_read()

detector = HandDetector(detectionCon=0.5, maxHands=2)

# setting up for the white background image

offset = 20  
imgSize = 300
counter = 0

checkCounterTakeOff = 1
checkCounterLand = 1
checkFlip = 1
checkMoveForward = 1
checkMoveBackward = 1
checkTakeImage = 1
checkTakeVideo = 1
checkMoveRight = 1
checkMoveLeft = 1
checkMoveUp = 1
checkMoveDown = 1

moveRightArray = [1,0,1,1,1]
moveLeftArray = [1,1,1,0,0]
moveForwardArray = [0,1,1,1,1]
moveBackwardArray = [0,1,1,1,0]
flipArray = [1,1,0,0,1]
takeVideo = [0,1,0,0,0]
takeImage = [0,1,1,0,0]
moveUp = [0,0,0,0,1]
moveDown = [1,0,0,0,0]

while True:
    # Getting the battery percentage 
    power=tello.get_battery()
    # Get image frame
    success, img = cap.read()
    img1 = frame_reader.frame
    img1 = cv2.resize(img1, (640, 480))
    # img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) 
    
    # Find the hand and its landmarks
    hands, img = detector.findHands(img)  # with draw
    # Draw battery percentage on the image
    cv2.putText(img, f'Battery: {power}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if hands:
        # Information for the first hand detected
        hand1 = hands[0]  # Get the first hand detected
        lmList1 = hand1["lmList"]  # List of 21 landmarks for the first hand
        bbox1 = hand1["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)
        center1 = hand1['center']  # Center coordinates of the first hand
        handType1 = hand1["type"]  # Type of the first hand ("Left" or "Right")

        # Count the number of fingers up for the first hand
        fingers, confidence_score = detector.fingersUp(hand1)
        print(fingers)

        #Tello drone functions 

        # Check if all elements in the array are zero
        if all(finger == 0 for finger in fingers) & checkCounterTakeOff:
            # If all elements are zero, make the drone fly
            checkCounterTakeOff = 0
            tello.takeoff()
            time.sleep(0.2)
            tello.move_up(150)
            cv2.putText(img, "Fly Mode", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img1, "Fly Mode", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif (fingers == moveDown) & checkMoveDown: 
            tello.move_down(100)
            cv2.putText(img, "Moving Down", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img1, "Moving Down", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            checkMoveDown = 0

        elif (fingers == moveUp) & checkMoveUp:
            tello.move_up(100)
            cv2.putText(img, "Moving Up", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img1, "Moving Up", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            checkMoveUp = 0

        elif all(finger == 1 for finger in fingers) & checkCounterLand:
            tello.land() 
            cv2.putText(img, "Landing", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img1, "Landing", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            checkCounterLand = 0

        elif (fingers == flipArray) & checkFlip:
            tello.flip_forward()
            cv2.putText(img, "Flip Forward and Backward", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img1, "Flip Forward and Backward", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            checkFlip = 0
        elif (fingers == moveForwardArray) & checkMoveForward: 
            tello.move_forward(150)
            cv2.putText(img, "Moving Forward", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img1, "Moving Forward", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            checkMoveForward = 0
        elif (fingers == moveBackwardArray) & checkMoveBackward:
            tello.move_back(150)
            cv2.putText(img, "Moving Backward", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img1, "Moving Backward", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            checkMoveBackward = 0
        elif (fingers == moveRightArray) & checkMoveRight: 
            tello.rotate_clockwise(90)
            tello.move_forward(150)
            tello.rotate_clockwise(-90)
            cv2.putText(img, "Moving Right", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img1, "Moving Right", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            checkMoveRight = 0
        elif (fingers == moveLeftArray) & checkMoveLeft:
            tello.rotate_clockwise(-90)
            tello.move_forward(150)
            tello.rotate_clockwise(90)
            cv2.putText(img, "Moving Left", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img1, "Moving Left", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            checkMoveLeft = 0
        elif ( fingers == takeImage) & checkTakeImage:
            cv2.imwrite("Picture1.png", img1)
            cv2.putText(img, "Taking Image", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img1, "Taking Image", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            checkTakeImage = 0
        elif (fingers == takeVideo) & checkTakeVideo:
            keepRecording = True
            checkTakeVideo = 0
            def videoRecorder():
                # create a VideoWrite object, recoring to ./video.avi
                # 创建一个VideoWrite对象，存储画面至./video.avi
                height, width, _ = frame_reader.frame.shape
                video = cv2.VideoWriter('video2.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

                while keepRecording:
                    video.write(frame_reader.frame)
                    time.sleep(1 / 30)

                video.release()

            recorder = Thread(target=videoRecorder)
            recorder.start()
            tello.rotate_clockwise(360)
            keepRecording = False
            recorder.join()
    
    
    # Display
    cv2.imshow("Image", img)
    cv2.imshow("TelloDrone", img1)
    cv2.waitKey(1)
cv2.destroyAllWindows()
tello.streamoff()
