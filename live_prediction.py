from newspaper import Article
import joblib
from utils.preprocessing import clean_text
from utils.feature_extraction import transform_features
from scipy.sparse import hstack

# Load models
ensemble = joblib.load('models/ensemble_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')
le_author = joblib.load('models/le_author.pkl')  # if author encoding used

# Input URL
url = input("Enter news article URL: ")
article = Article(url)
article.download()
article.parse()
text_clean = clean_text(article.text)

# Metadata (example: author unknown)
author_enc = le_author.transform(['Unknown'])[0]

X_live = transform_features([text_clean], [[author_enc]], vectorizer)
prediction = ensemble.predict(X_live)
proba = ensemble.predict_proba(X_live)[0]

print("Prediction:", "REAL" if prediction[0]==1 else "FAKE")
print(f"Confidence - REAL: {proba[1]*100:.2f}%, FAKE: {proba[0]*100:.2f}%")
