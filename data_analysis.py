import pandas as pd
from sklearn.model_selection import train_test_split

# Read the CSV file (replace with your actual path)
df = pd.read_csv("C:\Workspace\AI_Body_Language_Detector\Dataset\coords.csv")  # File must be in the same folder

# View data
print("First 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

X = df.drop('class', axis=1)  # Features (all columns except 'class')
y = df['class'] 
