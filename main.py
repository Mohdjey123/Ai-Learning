from src.predict import predict_sentiment


def main():
    user_input = input("Enter a sentence for sentiment analysis: ")
    sentiment = predict_sentiment(user_input)
    print(f"Sentiment: {sentiment}")

if __name__ == "__main__":
    main()
