import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.ensemble import VotingClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from utils.preprocessing import clean_text, encode_metadata
from wordcloud import WordCloud
import shap
import warnings
warnings.filterwarnings("ignore")

# ------------------ Load Dataset ------------------
fake = pd.read_csv('dataset/Fake.csv')
true = pd.read_csv('dataset/True.csv')

fake['label'] = 0
true['label'] = 1
data = pd.concat([fake, true], ignore_index=True)

# ------------------ Clean Text ------------------
if 'text' in data.columns:
    data['text_clean'] = data['text'].apply(clean_text)
else:
    raise ValueError("No 'text' column found in dataset!")

# ------------------ Encode Metadata if exists ------------------
if 'Author' in data.columns:
    data, le_author = encode_metadata(data, 'Author')
    X_meta = data[['Author_enc']]
else:
    le_author = None
    X_meta = None

X_text = data['text_clean']
y = data['label']

# ------------------ Load Vectorizer ------------------
vectorizer = joblib.load('models/vectorizer.pkl')

# Combine text features with metadata if exists
if X_meta is not None:
    X_final = hstack([vectorizer.transform(X_text), X_meta.values])
else:
    X_final = vectorizer.transform(X_text)

# ------------------ Train-Test Split ------------------
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# ------------------ Load Trained Models ------------------
svm = joblib.load('models/svm_model.pkl')
rf = joblib.load('models/rf_model.pkl')
gb = joblib.load('models/gb_model.pkl')

# ------------------ Evaluate Models ------------------
for model, name in zip([svm, rf, gb], ['SVM','RandomForest','GradientBoosting']):
    print(f"\n========== Evaluating {name} ==========")
    
    y_pred = model.predict(X_test)
    print(f"{name} Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt='d')
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # Probabilities if available
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:,1]

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0,1],[0,1],'k--')
        plt.title(f"{name} - ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.show()

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)
        plt.figure(figsize=(6,4))
        plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}")
        plt.title(f"{name} - Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.show()

        # Probability Distribution
        plt.figure(figsize=(6,4))
        sns.histplot(y_proba[y_test==0], color='red', label='FAKE', kde=True)
        sns.histplot(y_proba[y_test==1], color='green', label='REAL', kde=True)
        plt.title(f"{name} - Predicted Probability Distribution")
        plt.xlabel('Predicted Probability for REAL')
        plt.legend()
        plt.show()

        plt.figure(figsize=(10,5))


    # Feature Importance for tree-based models
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = importances.argsort()[-20:][::-1]  # top 20
        feature_names = list(vectorizer.get_feature_names_out())
        if X_meta is not None:
            feature_names += list(X_meta.columns)
        top_features = [feature_names[i] for i in indices]

        plt.figure(figsize=(10,6))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), top_features)
        plt.title(f'{name} - Top 20 Feature Importances')
        plt.gca().invert_yaxis()
        plt.show()

# ------------------ Train & Save Ensemble (Optional) ------------------
# Use smaller subset if full dataset is too slow
X_train_small = X_train[:2000]
y_train_small = y_train[:2000]
ensemble = VotingClassifier(estimators=[('svm',svm),('rf',rf),('gb',gb)], voting='soft')
ensemble.fit(X_train_small, y_train_small)
joblib.dump(ensemble, 'models/ensemble_model.pkl')
print("\n✅ Ensemble Model Trained & Saved Successfully.")

# ------------------ Word Clouds ------------------
fake_words = " ".join(data[data['label']==0]['text_clean'])
real_words = " ".join(data[data['label']==1]['text_clean'])
wc_fake = WordCloud(width=400, height=200, background_color='white').generate(fake_words)
wc_real = WordCloud(width=400, height=200, background_color='white').generate(real_words)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(wc_fake, interpolation='bilinear')
plt.axis('off')
plt.title('FAKE News Words')
plt.subplot(1,2,2)
plt.imshow(wc_real, interpolation='bilinear')
plt.axis('off')
plt.title('REAL News Words')
plt.show()
