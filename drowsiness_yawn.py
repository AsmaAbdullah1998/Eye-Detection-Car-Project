#python drowniness_yawn.py --webcam webcam_index


#IMPORTS-Demonstration
''''
*-face_utils: opensource wrapper library fot the most common face detection methods
*-argparse: recomended command-line parsing module in python stansard library
*-imutils: A series of convenience functions to make basic image processing fucntions such as translation, rotation, resizing 
*-face_utils submodule of imutils to access our helper fucntions
*-VideoStram: built-in webcam/USB camera/Raspberry Pi camera module
*-imutils library: a set of convenience functions to make working with openCV easier

'''

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os

import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(26, GPIO.OUT)  #buzzero
GPIO.setup(24, GPIO.OUT)   #Relay
GPIO.output(24, GPIO.HIGH)


''''
HARDWARE
'''
def alarm(msg):
    global alarm_status
    global alarm_status2
    global saying

    while alarm_status:
        print('call')
        s = 'espeak "'+msg+'"'
        os.system(s)

    if alarm_status2:
        print('call')
        saying = True
        s = 'espeak "' + msg + '"'
        os.system(s)
        saying = False



''''
EYE ASPECT RATIO FUNCTION 
'''
def eye_aspect_ratio(eye):

    #vertical eye landmarks (x,y)-coordinates computations 
    A = dist.euclidean(eye[1], eye[5]) 
    B = dist.euclidean(eye[2], eye[4])
    
    #horizontal eye landmarks (x,y)-coordinates computations
    C = dist.euclidean(eye[0], eye[3])

    #compute the eye aspect ratio 
    ear = (A + B) / (2.0 * C)

    return ear

''''
average the eye aspect ratio together for both eyes
'''
def final_ear(shape):
    #grab the indexes of the faical landmardks for the left and right eye 
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)





#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser() #
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())



''''
define two constants, one for the eyes aspecct ration to indicaate blink and then 
a second constant for the number of consective 
frames the eye must be below the threshold
'''
EYE_AR_THRESH = 0.3 #the defaulte of threshold value that is the best in many application 
EYE_AR_CONSEC_FRAMES = 40 #sleepy: indicate that 30 successive frames with an eye aspect ration less than eye-ar-thresh must happen in order to say "sleeping"
EYE_AR_CONSEC_FRAMES2 = 20 #tired 
EYE_AR_CONSEC_FRAMES3 = 3 #blinking times -normal


alarm_status = False
alarm_status2 = False
saying = False



#indicate the total number of succesive frames that have an eye aspect ratio less than eye-ar-thers  
COUNTER = 0
#TOTAL is the total number of blinks that have taken place while the script has been running
TOTAL = 0


print("-> Loading the predictor and detector...")
#detector = dlib.get_frontal_face_detector() for pc 
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    #Faster but less accurate
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  #loads the facial landmark predictonr


print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
#vs= VideoStream(usePiCamera=True).start()       //For Raspberry Pi
time.sleep(1.0)

''''
loop over frames from the video stream
'''
while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=450) #size the vid
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces in the grayscale frame
    #rects = detector(gray, 0)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
        minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

           
    ''''
    loop over the face detections
    '''
    #for rect in rects:
    for (x, y, w, h) in rects:
        #dlib.rectangle(left:float, top:float, right:float, bottom:float)
        rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
        
        ''''
        determine the facial landmarks for the face region 
        then convert the faical landmark (x,y)-coordinates to a NumPy
        array
        '''
        shape = predictor(gray, rect) #giving us the 68 (x,y)-coordinates that map to the specific facial features
        shape = face_utils.shape_to_np(shape) #converts the dlib shape object to a numpy array with shape (68,2)

        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye [1]
        rightEye = eye[2]

        

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

       

        if ear < EYE_AR_THRESH: #check to see if the eye aspect ratio is below the blink threshold and if so increment the blink frame counter
            COUNTER += 1

            
            if COUNTER >= EYE_AR_CONSEC_FRAMES: #sleeping
                cv2.putText(frame, "sleeping, Stop", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                GPIO.output(24, GPIO.LOW)
                
                
            
            elif COUNTER >= EYE_AR_CONSEC_FRAMES2: #tired
                cv2.putText(frame, "tired, Buzzer", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                GPIO.output(26, GPIO.HIGH)
                time.sleep(0.2)
                
            GPIO.output(26, GPIO.LOW)
            
                 
                           
                             

        else: #otherwise, the eye aspect ratio is not below the blink threshold
            if COUNTER >= EYE_AR_CONSEC_FRAMES3:
                TOTAL +=1


            COUNTER = 0 #reset the eye frame counter 
            alarm_status = False

        

        cv2.putText(frame, "Blinks: {:.2f}".format(TOTAL), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        


    cv2.imshow("Frame", frame) #show the output video frame
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()










''''
NoteImportant
EAR-equation

EAR = ((P2=P6)+(P3-P5))/2(P1-P4)
P1,....P6 are 2D facial landmarkds locations 
numerator: compute the distance between the vertical eye landmarks 
denominator: compute the distance between horizontal eye landmarks
multiply the denominator by 2 >> to weighting it appropriately since there is only
one set of horizontal points but two sets of vertical points 

EAR value is constant while the eye is open, but will rapidly fall to zero when a blink is taking place



'''
