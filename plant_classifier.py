import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from utils.feature_extraction import extract_features

# Load CSV
df = pd.read_csv('train.csv')
df['label_encoded'] = LabelEncoder().fit_transform(df['label'])
image_folder = 'data/images'

# Feature Extraction
X, y = [], []
for _, row in tqdm(df.iterrows(), total=len(df)):
    img_path = os.path.join(image_folder, row['image_id'] + '.jpg')
    if os.path.exists(img_path):
        X.append(extract_features(img_path))
        y.append(row['label_encoded'])

X = np.array(X)
y = np.array(y)

# Scaling and Splitting
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()
