# predict.py
import joblib
from .preprocess import preprocess_text

def predict_sentiment(text):
    # Load the trained model
    model = joblib.load('sentiment_model.pkl')
    
    # Preprocess the input text
    processed_text = preprocess_text(text)
    
    # Predict the sentiment
    prediction = model.predict([processed_text])
    
    # Return the prediction
    return prediction[0]

if __name__ == "__main__":
    sample_text = "I love this product!"
    print(f"Predicted sentiment: {predict_sentiment(sample_text)}")
