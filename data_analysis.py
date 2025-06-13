import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # Correct import
# Load dataset
df = pd.read_csv(r"C:\Workspace\AI_Body_Language_Detector\Dataset\coords.csv", header=None)

# Assign column names
# Assuming first column is class label and rest are features
df.columns = ['class'] + [f'feature_{i}' for i in range(1, df.shape[1])]

# Display first few rows
print("Head of dataset:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

# Show original class distribution
print("\nOriginal class distribution:")
print(df['class'].value_counts())

# Balance the dataset (undersample to smallest class count)
min_count = df['class'].value_counts().min()
balanced_df = df.groupby('class').apply(lambda x: x.sample(min_count)).reset_index(drop=True)

# Show balanced class distribution
print("\nBalanced class distribution:")
print(balanced_df['class'].value_counts())

# Separate features and target
X = balanced_df.drop('class', axis=1)
y = balanced_df['class']

# Plot class distribution
y.value_counts().plot(kind='bar', title='Class Distribution After Balancing')
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.grid(True)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

#Model Building
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.impute import SimpleImputer

pipelines={
    'lr':make_pipeline(SimpleImputer(strategy='mean'),StandardScaler(),LogisticRegression()),
    'rc':make_pipeline(SimpleImputer(strategy='mean'),StandardScaler(),RidgeClassifier()),
    'rf':make_pipeline(SimpleImputer(strategy='mean'),StandardScaler(),RandomForestClassifier()),
    'gb':make_pipeline(SimpleImputer(strategy='mean'),StandardScaler(),GradientBoostingClassifier()),
}

fit_models={}
for algo,pipeline in pipelines.items():
    model=pipeline.fit(X_train,y_train)
    fit_models[algo]=model

#ModelDeployment

from sklearn.metrics import accuracy_score
import pickle

for algo,model in fit_models.items():
    yhat=model.predict(X_test)
    print(algo,accuracy_score(y_test,yhat))


import os
# Ensure directory exists
os.makedirs(r'C:\Users\91944\Documents\AI_Body_Language_Detector\Model', exist_ok=True)
# Save the model
with open(r'C:\Users\91944\Documents\AI_Body_Language_Detector\Model\body_language.pkl', 'wb') as f:
    pickle.dump(fit_models['lr'], f)
with open(r'C:\Users\91944\Documents\AI_Body_Language_Detector\Model\body_language.pkl', 'rb') as f:
    model=pickle.load(f)
