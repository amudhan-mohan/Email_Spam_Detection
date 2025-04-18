from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

# Load model
model_path = os.path.join('model', 'spam_classifier.pkl')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file was not found at {model_path}.")

with open(model_path, 'rb') as f:
    vectorizer, model = pickle.load(f)

@app.route('/')
def home():
    return "📨 Email Spam Detector is Live on Render!"

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
