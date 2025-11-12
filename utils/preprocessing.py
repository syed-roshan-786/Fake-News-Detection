import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Clean text by removing non-alphabetic characters and stopwords."""
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower().split()
    text = ' '.join([word for word in text if word not in stop_words])
    return text

def encode_metadata(df, column):
    """Encode categorical metadata using LabelEncoder."""
    le = LabelEncoder()
    df[column] = df[column].fillna('Unknown')
    df[column+'_enc'] = le.fit_transform(df[column])
    return df, le
