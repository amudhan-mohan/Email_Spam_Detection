import os
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Corrected model path for Render (model folder is inside the same backend folder)
model_path = os.path.join('model', 'spam_classifier.pkl')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file was not found at {model_path}. Please ensure the file exists.")

with open(model_path, 'rb') as f:
    vectorizer, model = pickle.load(f)

@app.route('/')
def index():
    return "Email Spam Detector API is Live!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    email_id = data.get('email_id', '')
    subject = data.get('subject', '')
    message = data.get('message', '')

    full_text = f"{email_id} {subject} {message}"
    vector = vectorizer.transform([full_text])
    prediction = model.predict(vector)[0]

    return jsonify({"result": "SPAM" if prediction == 1 else "NOT SPAM"})

if __name__ == '__main__':
    app.run(debug=True)
