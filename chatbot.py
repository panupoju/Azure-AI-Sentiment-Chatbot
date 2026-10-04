# Azure AI Sentiment Analysis Chatbot

import os
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

AZURE_ENDPOINT = os.environ.get("AZURE_LANGUAGE_ENDPOINT")
AZURE_KEY = os.environ.get("AZURE_LANGUAGE_KEY")

if not AZURE_ENDPOINT or not AZURE_KEY:
    raise ValueError(
        "Azure credentials are missing. Set AZURE_LANGUAGE_ENDPOINT "
        "and AZURE_LANGUAGE_KEY environment variables."
    )

client = TextAnalyticsClient(
    endpoint=AZURE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_KEY)
)


def analyze_user_sentiment(text):
    try:
        response = client.analyze_sentiment([text])[0]

        if response.is_error:
            return None, None

        scores = {
            "positive": response.confidence_scores.positive,
            "neutral": response.confidence_scores.neutral,
            "negative": response.confidence_scores.negative
        }

        return response.sentiment, scores

    except Exception as e:
        print("Azure AI service error:", str(e))
        return None, None


def show_capabilities():
    print("\nAzure AI Chatbot Capabilities")
    print("1. Respond to basic greetings")
    print("2. Analyze message sentiment using Azure AI Language")
    print("3. Identify positive, neutral, negative, or mixed sentiment")
    print("4. Display sentiment confidence scores")
    print("5. Handle empty input")
    print("6. Type 'help' to view capabilities")
    print("7. Type 'quit' to exit\n")


def chatbot():
    print("=" * 55)
    print("AZURE AI SENTIMENT ANALYSIS CHATBOT")
    print("=" * 55)
    print("Connected to Azure AI Language.")
    print("Type 'help' for capabilities or 'quit' to exit.")

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            print("Bot: I didn't receive any text. Please enter a message.")
            continue

        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Bot: Thank you for using the Azure AI Chatbot. Goodbye!")
            break

        if user_input.lower() in ["help", "capabilities", "what can you do"]:
            show_capabilities()
            continue

        if user_input.lower() in ["hello", "hi", "hey"]:
            print("Bot: Hello! Nice to meet you. Tell me how you are feeling today.")
            continue

        sentiment, scores = analyze_user_sentiment(user_input)

        if sentiment is None:
            print("Bot: Sorry, I could not analyze that message.")
            continue

        print(f"Bot: Azure AI detected your sentiment as: {sentiment.upper()}")
        print(
            f"Bot: Confidence Scores -> "
            f"Positive: {scores['positive']:.3f}, "
            f"Neutral: {scores['neutral']:.3f}, "
            f"Negative: {scores['negative']:.3f}"
        )

        if sentiment == "positive":
            print("Bot: That sounds positive! I'm glad to hear that.")
        elif sentiment == "negative":
            print("Bot: That sounds negative. I hope things improve soon.")
        elif sentiment == "neutral":
            print("Bot: Your message appears to be mostly neutral.")
        else:
            print("Bot: Your message contains mixed emotions.")


if __name__ == "__main__":
    chatbot()
