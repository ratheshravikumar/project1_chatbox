# project1_chatbox

# 🤖 AI-Powered Chatbot (NLP + ML)

A simple chatbot built using **Machine Learning + NLP** that can understand user intent and respond accordingly.

---

## 🚀 Features

* Intent classification using TF-IDF + Logistic Regression
* REST API with Flask
* Predefined conversational responses

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python chatbot_app.py
```

Server will run at:

```
http://127.0.0.1:5000
```

---

## 🧪 API Usage

### Endpoint:

```
POST /chat
```

### Request:

```json
{
  "message": "Hello"
}
```

### Response:

```json
{
  "intent": "greeting",
  "response": "Hi there!"
}
```

---

## 🧠 ML Concept Used

* TF-IDF Vectorization
* Logistic Regression (Intent Classification)

---

## 📌 Future Improvements

* Add deep learning (Transformers / BERT)
* Store chat history in database
* Add frontend UI (React / HTML)

---

## 👨‍💻 Author

Rathesh.R
