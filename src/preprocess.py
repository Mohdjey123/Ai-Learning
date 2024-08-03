import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path, encoding='latin-1', header=None)
    df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
    
    # Preprocess text
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Map sentiment to binary
    df['sentiment'] = df['sentiment'].map({0: 'negative', 4: 'positive'})
    
    return df[['processed_text', 'sentiment']]
