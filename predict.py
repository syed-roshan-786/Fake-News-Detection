import joblib
from utils.preprocessing import clean_text
from utils.feature_extraction import transform_features
from scipy.sparse import hstack

# Load models
ensemble = joblib.load('models/ensemble_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')
le_author = joblib.load('models/le_author.pkl')

def predict_news(news_text):
    """
    Predict if the news is REAL or FAKE.
    """
    # Clean the text
    text_clean = clean_text(news_text)

    # Metadata (example: unknown author)
    author_enc = le_author.transform(['Unknown'])[0]

    # Transform features
    X_live = transform_features([text_clean], [[author_enc]], vectorizer)

    # Make prediction
    prediction = ensemble.predict(X_live)[0]

    # Return result
    if prediction == 1:
        return "REAL NEWS ✅"
    else:
        return "FAKE NEWS ❌"

# Main
if __name__ == "__main__":
    news = input("\nEnter the news text to verify:\n\n> ")
    result = predict_news(news)
    print("\nPrediction:", result)
