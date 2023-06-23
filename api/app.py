import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from pathlib import Path
from playsound import playsound

from flask import Flask, render_template, redirect, Response

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

imWth = 450
imHgt = 450

cnn_model = load_model('cnn_model_trlw.h5')

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

def generateFrames():

    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(static_image_mode = False,min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
    
            ret, frame = cap.read()

            frame = cv2.resize(frame,(imWth,imHgt))
        
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
      
            # Make detection
            results = pose.process(image)
    
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
            # Extract landmarks
            try:

                landmarks = results.pose_landmarks.landmark

                img = np.array(image, dtype='float64') / 255.0

                print(img.shape)

                img = np.reshape(img,(1,imWth,imHgt,3))

                print(img.shape)

                y_pred = cnn_model.predict(img)
                print(y_pred)

                ypose = np.argmax(y_pred)

                if (ypose==0):

                    cv2.putText(image, 'warrior', (300,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    print('warrior')

                    right_index_x = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x
                    right_index_y = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y 

                    right_shoulder_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
                    right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y 

                    left_index_x = landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x
                    left_index_y = landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y 

                    angle = calculate_angle([right_index_x,right_index_y],[right_shoulder_x,right_shoulder_y],[left_index_x,left_index_y])

                    if angle!=180:

                        audio = Path().cwd() / "warrior.mp3"
                        playsound(audio)

                elif(ypose==1):

                    cv2.putText(image, 'tree right', (300,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
                    print('tree right')

                    right_waist_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x
                    right_waist_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y 

                    right_knee_x = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x
                    right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y 

                    right_ankle_x = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x
                    right_ankle_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y 

                    angle = calculate_angle([right_waist_x,right_waist_y],[right_knee_x,right_knee_y],[right_ankle_x,right_ankle_y])

                    if angle!=45:

                        audio = Path().cwd() / "tree.mp3"
                        playsound(audio)

                elif(ypose==2):

                    cv2.putText(image, 'tree left', (300,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
                    print('tree left')

                    left_waist_x = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x
                    left_waist_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y 

                    left_knee_x = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x
                    left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y 

                    left_ankle_x = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x
                    left_ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y 

                    angle = calculate_angle([left_waist_x,left_waist_y],[left_knee_x,left_knee_y],[left_ankle_x,left_ankle_y])

                    if angle!=45:

                        audio = Path().cwd() / "tree.mp3"
                        playsound(audio)
            
            except:
                pass
        
        
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )  
       
        
            cv2.imshow('Yoga pose analyser', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analysepose")
def analysePose():
    return Response(generateFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()