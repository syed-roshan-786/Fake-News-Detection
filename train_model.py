import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from utils.preprocessing import clean_text
from utils.feature_extraction import extract_features

# Make sure model folder exists
os.makedirs('models', exist_ok=True)

# Load dataset
print("Loading dataset...")
fake = pd.read_csv('dataset/Fake.csv')
true = pd.read_csv('dataset/True.csv')

fake['label'] = 0
true['label'] = 1
data = pd.concat([fake, true]).reset_index(drop=True)

# Batch text cleaning
chunk_size = 5000
data['text_clean'] = ''

print("Cleaning text in batches...")
for start in range(0, len(data), chunk_size):
    end = min(start + chunk_size, len(data))
    data.loc[start:end-1, 'text_clean'] = data['text'].iloc[start:end].apply(clean_text)
    print(f"Processed rows {start} to {end-1}")
# Features and labels
X_text = data['text_clean']
y = data['label']

# Extract features (limit max_features to speed up)
X_final, vectorizer = extract_features(X_text, max_features=3000)  # Pass max_features if function supports

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Initialize models
svm = LinearSVC(class_weight='balanced', max_iter=5000)


svm = CalibratedClassifierCV(svm)
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced')
gb = GradientBoostingClassifier(n_estimators=200)


# Train models
print("Training SVM...")
svm.fit(X_train, y_train)
print("SVM training completed.")

# Optional: Train RF/GB on a smaller sample to avoid freezing
sample_size = 10000  # adjust according to your system memory
print(f"Training RandomForest and GradientBoosting on {sample_size} samples...")
sample_idx = data.sample(n=sample_size, random_state=42).index
X_train_sample = X_final[sample_idx]
y_train_sample = y.iloc[sample_idx]
print("Training Random Forest...")
rf.fit(X_train_sample, y_train_sample)
print("Training Gradient Boosting...")
gb.fit(X_train_sample, y_train_sample)

# Save models
joblib.dump(svm, 'models/svm_model.pkl')
joblib.dump(rf, 'models/rf_model.pkl')
joblib.dump(gb, 'models/gb_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

print("Training completed and models saved.")
