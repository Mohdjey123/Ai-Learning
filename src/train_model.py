import gdown
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to download dataset from Google Drive
def download_dataset():
    url = 'https://drive.google.com/uc?id=1X3qHMtE1zqhUnelA3QL1LumOA8LDam-d'
    output = 'sentiment140.csv'
    if not os.path.exists(output):
        print("Downloading dataset...")
        gdown.download(url, output, quiet=False)
    else:
        print("Dataset already downloaded.")

# Download the dataset
download_dataset()

# Load the dataset
try:
    # Read the CSV file without assigning column names first
    df = pd.read_csv('sentiment140.csv', encoding='latin1', on_bad_lines='skip', header=None)

    # Print the first few rows and columns to understand the structure
    print("First few rows of the dataset:")
    print(df.head())

    # Inspect the number of columns
    print(f"Number of columns in the dataset: {df.shape[1]}")

    # Assign column names based on the actual structure
    df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# Preprocessing and cleaning
df['text'] = df['text'].str.replace(r'@\w+', '', regex=True)  # Remove mentions
df['text'] = df['text'].str.replace(r'http\S+', '', regex=True)  # Remove URLs
df['text'] = df['text'].str.replace(r'[^a-zA-Z\s]', '', regex=True)  # Remove punctuation
df['text'] = df['text'].str.lower()  # Convert to lowercase

# Feature extraction and model training
X = df['text']
y = df['sentiment'].apply(lambda x: 'positive' if x == 4 else 'negative')  # Convert labels to 'positive' or 'negative'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

joblib.dump(model, 'sentiment_model.pkl')