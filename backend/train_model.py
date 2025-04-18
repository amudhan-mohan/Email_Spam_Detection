import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle
import os

# Load dataset
df = pd.read_csv('dataset/email_dataset.csv')
df['text'] = df['email_id'] + ' ' + df['subject'] + ' ' + df['body']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
os.makedirs('model', exist_ok=True)
with open('model/spam_classifier.pkl', 'wb') as f:
    pickle.dump((vectorizer, model), f)
