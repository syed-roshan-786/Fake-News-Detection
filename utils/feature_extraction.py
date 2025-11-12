from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import joblib

def extract_features(text_series, metadata_df=None, max_features=None):
    """
    Extract TF-IDF features from text and optionally combine with metadata.
    """
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    X_text = vectorizer.fit_transform(text_series)

    # Save vectorizer
    joblib.dump(vectorizer, 'models/vectorizer.pkl')

    if metadata_df is not None:
        # If you have metadata, combine it here
        X_final = hstack([X_text, metadata_df])
    else:
        X_final = X_text

    return X_final, vectorizer

def transform_features(text_list, metadata_list, vectorizer):
    """Transform new data using saved vectorizer and metadata."""
    X_tfidf = vectorizer.transform(text_list)
    X_final = hstack([X_tfidf, metadata_list])
    return X_final
