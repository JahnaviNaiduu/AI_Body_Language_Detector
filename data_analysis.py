import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv(r"Dataset\coords.csv", header=None)
print("Raw dataset:\n", df)

# Rename columns: first column is class, rest are features
df.columns = ['class'] + [f'Landmark_{i}' for i in range(1, df.shape[1])]

print("Head of dataset:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nOriginal class distribution:")
print(df['class'].value_counts())

# Balance the dataset
min_count = df['class'].value_counts().min()
balanced_df = df.groupby('class').apply(lambda x: x.sample(min_count)).reset_index(drop=True)

print("\nBalanced class distribution:")
print(balanced_df['class'].value_counts())


# Class distribution
balanced_df['class'].value_counts().plot(kind='bar', title='Class Distribution After Balancing')
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.grid(True)
plt.show()
plt.close()

X = balanced_df.drop('class', axis=1)
y = balanced_df['class']

# Histogram of a few sample features
for col in X.columns[:5]:
    plt.figure(figsize=(6, 3))
    sns.histplot(X[col], kde=True, bins=30)
    plt.title(f'Univariate: Distribution of {col}')
    plt.tight_layout()
    plt.show()
    plt.close()


# Boxplots: Feature vs Emotion class
for col in X.columns[:3]:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=y, y=X[col])
    plt.title(f'Bivariate: {col} vs Emotion Class')
    plt.tight_layout()
    plt.show()
    plt.close()


# Feature correlation heatmap
plt.figure(figsize=(15, 12))
corr_matrix = X.corr()
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title('Multivariate: Feature Correlation Heatmap')
plt.tight_layout()
plt.show()
plt.close()

#Model Building
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

pipelines = {
    'lr': make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), LogisticRegression(max_iter=1000)),
    'rc': make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), RidgeClassifier()),
    'rf': make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), RandomForestClassifier()),
    'gb': make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), GradientBoostingClassifier()),
}

fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model

# model evaluation
from sklearn.metrics import accuracy_score
import pickle
import os

for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    print("Accuracy", algo, accuracy_score(y_test, yhat))

# save model
os.makedirs(r'Model', exist_ok=True)
with open(r'Model\body_language.pkl', 'wb') as f:
    pickle.dump(fit_models['lr'], f)

with open(r'Model\body_language.pkl', 'rb') as f:
    model = pickle.load(f)
print(type(model))

import mediapipe as mp
import csv
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        try:
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[l.x, l.y, l.z, l.visibility] for l in pose]).flatten())
            face = results.face_landmarks.landmark
            face_row = list(np.array([[l.x, l.y, l.z, l.visibility] for l in face]).flatten())

            row = pose_row + face_row
            X_live = pd.DataFrame([row], columns=X_train.columns)

            body_language_class = model.predict(X_live)[0]
            body_language_prob = model.predict_proba(X_live)[0]

            print(body_language_class, body_language_prob)

            coords = tuple(np.multiply(
                np.array((
                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)),
                [640, 480]).astype(int))

            cv2.rectangle(image, (coords[0], coords[1] + 5), (coords[0] + len(body_language_class) * 20, coords[1] - 30), (245, 117, 16), -1)
            cv2.putText(image, body_language_class, coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)
            cv2.putText(image, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0], (98, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")

        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
