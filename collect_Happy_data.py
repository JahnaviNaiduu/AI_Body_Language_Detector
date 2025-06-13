import mediapipe as mp
import csv
import cv2
import numpy as np

class_name="Happy"
cap = cv2.VideoCapture(0)

# Initiate holistic model
with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
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

        # 1. Face landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            image, 
            results.face_landmarks, 
            mp.solutions.holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        )

        # 2. Right hand
        mp.solutions.drawing_utils.draw_landmarks(
            image, 
            results.right_hand_landmarks, 
            mp.solutions.holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )

        # 3. Left hand
        mp.solutions.drawing_utils.draw_landmarks(
            image, 
            results.left_hand_landmarks, 
            mp.solutions.holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )

        # 4. Pose
        mp.solutions.drawing_utils.draw_landmarks(
            image, 
            results.pose_landmarks, 
            mp.solutions.holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        # 5. Export Co-ordinates
        try:
            # Extract Pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = [np.array([landmark.x, landmark.y, landmark.z, landmark.visibility]) for landmark in pose]
            pose_row = np.array(pose_row).flatten()

            # Extract Face landmarks
            face = results.face_landmarks.landmark
            face_row = [np.array([landmark.x, landmark.y, landmark.z, landmark.visibility]) for landmark in face]
            face_row = np.array(face_row).flatten()

            # Combine
            row = np.concatenate([pose_row, face_row])

            # Prepend class name
            row = np.insert(row, 0, "Sad")

            # Export to CSV
            with open('Dataset/coords.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)

        except Exception:
            pass

        # Display
        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

