# chatbot_app.py

import random
import json
import nltk
import numpy as np
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('punkt')

# -------------------------------
# Sample Training Data (Intents)
# -------------------------------
data = {
    "intents": [
        {"tag": "greeting",
         "patterns": ["Hi", "Hello", "Hey", "Good morning"],
         "responses": ["Hello!", "Hi there!", "Hey! How can I help?"]},

        {"tag": "goodbye",
         "patterns": ["Bye", "See you", "Goodbye"],
         "responses": ["Goodbye!", "See you soon!", "Bye!"]},

        {"tag": "name",
         "patterns": ["What is your name?", "Who are you?"],
         "responses": ["I'm your AI chatbot.", "You can call me ChatBot."]},

        {"tag": "help",
         "patterns": ["Help me", "I need help", "Support"],
         "responses": ["Sure, I'm here to help!", "Tell me your problem."]},
    ]
}

# -------------------------------
# Prepare Training Data
# -------------------------------
corpus = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        corpus.append(pattern)
        labels.append(intent['tag'])

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Train ML Model
model = LogisticRegression()
model.fit(X, labels)

# -------------------------------
# Chatbot Logic
# -------------------------------
def predict_intent(user_input):
    X_test = vectorizer.transform([user_input])
    prediction = model.predict(X_test)[0]
    return prediction

def get_response(tag):
    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I don't understand."

# -------------------------------
# Flask API
# -------------------------------
app = Flask(__name__)

@app.route('/')
def home():
    return "AI Chatbot is running!"

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")

    if not user_input:
        return jsonify({"response": "Please send a message."})

    tag = predict_intent(user_input)
    response = get_response(tag)

    return jsonify({
        "intent": tag,
        "response": response
    })

# -------------------------------
# Run Server
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)