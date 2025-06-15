import streamlit as st
import pandas as pd
import tensorflow as tf
import mediapipe as mp
import cv2
import numpy as np
import pickle
import csv
import os

@st.cache_resource
def load_model():
    """Load and cache the ML model."""
    with open('Model/body_language.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

st.write("""# Body Language Detection""")
mp_drawing = mp.solutions.drawing_utils  # Drawing Helpers
mp_holistic = mp.solutions.holistic  # Mediapipe Solutions

run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

if run:
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while camera.isOpened():
            ret, frame = camera.read()
            if not ret:
                break

            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make Detections
            results = holistic.process(image)

            # Recolor back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Drawing landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                     mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                     mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            try:
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())  

                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())  

                row = pose_row + face_row

                # Make Detections
                X = pd.DataFrame([row]) 
                
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]

                coords = tuple(np.multiply(np.array((results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                                                     results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)),
                                         [640, 480]).astype(int))

                cv2.rectangle(image, 
                             (coords[0], coords[1] + 5), 
                             (coords[0] + len(body_language_class)*20, coords[1] - 30), 
                             (245, 117, 16), -1)

                cv2.putText(image, body_language_class, coords, 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Display Class
                cv2.putText(image, 'CLASS', (95, 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                cv2.putText(image, body_language_class.split(' ')[0], 
                            (90, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Display Probabilities
                cv2.putText(image, 'PROB', (15, 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)), 
                            (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            except:
                pass

            FRAME_WINDOW.image(image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    
    camera.release()
    cv2.destroyAllWindows()
else:
    st.write('Stopped')
