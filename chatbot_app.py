"""
AI-Powered Chatbot (NLP + ML)
----------------------------
Features:
- Intent classification using ML (TF-IDF + Logistic Regression)
- REST API using Flask
- Context-based responses

Author: Rathesh.R
"""

import random
import nltk
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download tokenizer
nltk.download('punkt')

# ==============================
# Training Data (Intents)
# ==============================
intents = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "Hey", "Good morning", "Good evening"],
            "responses": ["Hello!", "Hi there!", "Hey! How can I help you?"]
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "See you", "Goodbye", "Catch you later"],
            "responses": ["Goodbye!", "See you soon!", "Bye! Have a great day!"]
        },
        {
            "tag": "name",
            "patterns": ["What is your name?", "Who are you?", "Your name"],
            "responses": ["I'm your AI chatbot.", "You can call me ChatBot."]
        },
        {
            "tag": "help",
            "patterns": ["Help me", "I need help", "Support", "Can you help me"],
            "responses": ["Sure! Tell me your problem.", "I'm here to help."]
        },
        {
            "tag": "thanks",
            "patterns": ["Thanks", "Thank you", "That's helpful"],
            "responses": ["You're welcome!", "Glad I could help!"]
        }
    ]
}

# ==============================
# Prepare Training Data
# ==============================
corpus = []
labels = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        corpus.append(pattern)
        labels.append(intent["tag"])

# ==============================
# Vectorization (ML)
# ==============================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# ==============================
# Model Training
# ==============================
model = LogisticRegression()
model.fit(X, labels)

# ==============================
# Chatbot Functions
# ==============================
def predict_intent(text):
    X_test = vectorizer.transform([text])
    return model.predict(X_test)[0]


def get_response(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, I didn't understand that."


# ==============================
# Flask App
# ==============================
app = Flask(__name__)


@app.route('/')
def home():
    return "AI Chatbot API is running 🚀"


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message'"}), 400

    user_input = data["message"]
    intent = predict_intent(user_input)
    response = get_response(intent)

    return jsonify({
        "intent": intent,
        "response": response
    })


# ==============================
# Run Server
# ==============================
if __name__ == "__main__":
    app.run(debug=True)