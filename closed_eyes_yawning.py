from PIL import Image, ImageDraw
	
from scipy.spatial import distance
import face_recognition
import cv2
import numpy as np


thresh_eye = 0.25
thresh_mouth=1.1
flag=0
frame_check=5

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear


def lips_aspect_ratio(UL,LL):
	A=distance.euclidean(UL[5],LL[5])
	B=distance.euclidean(UL[6],LL[6])
	C=distance.euclidean(UL[7],LL[7])
	ear_l=(A+B)/(2.0*C)
	return ear_l


#builiding for web cam
#video_capture=cv2.VideoCapture(0)
flag_eye=0
flag_lips=0

def detect(image):
    #image=image[:, :, ::-1]
    face_landmarks_list = face_recognition.face_landmarks(image)

    print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

    # Create a PIL imagedraw object so we can draw on the picture
    #pil_image = Image.fromarray(image)
    #d = ImageDraw.Draw(pil_image)
    
    for face_landmarks in face_landmarks_list:

    # Print the location of each facial feature in this image
    #for facial_feature in face_landmarks.keys():
     #   print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

    # Let's trace out each facial feature in the image with a line!
    #for facial_feature in face_landmarks.keys():
        #d.line(face_landmarks['left_eye'], width=5)
        #d.line(face_landmarks['right_eye'], width=5)
        my_eyel = np.asarray(face_landmarks['left_eye'])
        my_eyel=my_eyel.reshape(-1,1,2)
        cv2.drawContours(image, my_eyel, -1, (0,255,0), 1)
        
        my_eyer = np.asarray(face_landmarks['right_eye'])
        my_eyer=my_eyer.reshape(-1,1,2)
        cv2.drawContours(image, my_eyer, -1, (0,255,0), 1)
        
        my_upper = np.asarray(face_landmarks['top_lip'])
        my_upper=my_upper.reshape(-1,1,2)
        cv2.drawContours(image, my_upper, -1, (0,255,0), 1)
        
        
        my_lower = np.asarray(face_landmarks['bottom_lip'])
        my_lower=my_lower.reshape(-1,1,2)
        cv2.drawContours(image, my_lower, -1, (0,255,0), 1)
        
       # nose_bridge = np.asarray(face_landmarks['nose_bridge'])
       # nose_bridge=nose_bridge.reshape(-1,1,2)
       # cv2.drawContours(image, nose_bridge, -1, (0,255,0), 2)
        
       # nose_tip = np.asarray(face_landmarks['nose_tip'])
       # nose_tip=nose_tip.reshape(-1,1,2)
       # cv2.drawContours(image, nose_tip, -1, (0,255,0), 2)


        ear_leye=eye_aspect_ratio(my_eyel)
        ear_reye=eye_aspect_ratio(my_eyer)
        eye_ear=(ear_leye+ear_reye)/2.0
        global flag_eye

        if eye_ear < thresh_eye:
        	flag_eye = flag_eye+1
        	print ("E"+str(flag_eye))
        	if flag_eye >= frame_check:
        		cv2.putText(image, "******ALERT!*****", (200, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 4)
        		cv2.putText(image, "*****Sleeping*****", (200,60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 4)
        else:
        	flag_eye = 0


        lips_ear=lips_aspect_ratio(my_upper,my_lower)
        global flag_lips

        if lips_ear > thresh_mouth:
        	flag_lips = flag_lips+1
        	print ("L"+str(flag_lips))
        	if flag_lips >= frame_check:
        		cv2.putText(image, "******ALERT!*****", (200, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 4)
        		cv2.putText(image, "*****Yawning*****", (200,60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 4)
        else:
        	flag_lips = 0




        
        
        #myarrayr= 
    return image

#builiding for web cam
video_capture=cv2.VideoCapture(0)
while True:
    _,frame=video_capture.read()
    #gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas=detect(frame)
    cv2.imshow('video',canvas)
   # pil_image.show()
    
    if(cv2.waitKey(1)&0xff==ord('q')):
        break
video_capture.release()
cv2.destroyAllWindows()