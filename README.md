# Email Spam Detection - Random Forest

## Overview
This project is an Email Spam Detection system that uses machine learning to classify emails as either "spam" or "ham" (not spam). The system is built using a **Random Forest** classifier and provides a web-based interface for users to input email details and get predictions.

## Features
- **Dataset**: A curated dataset of emails labeled as spam or ham.
- **Machine Learning Model**: Random Forest classifier for accurate predictions.
- **Web Interface**: User-friendly web app to input email details and get predictions.
- **API Integration**: Backend API to handle predictions.

## Project Structure
```
Email_Spam_Detection/
├── backend/
│   ├── dataset/                                # Contains the email dataset
│   │   ├── email_dataset_full_with_email.csv   # Full dataset with labeled emails
│   ├── model/                                  # Machine learning model files
│   |   ├──spam_classifier.pkl                  # Pre-trained Random Forest model
│   ├── app.py                                  # Backend API implementation
│   ├── requirements.txt                        # Package requirements for this project
│   ├── train_model.py                          # Script to train the model
├── frontend/
│   ├── index.html                              # HTML templates for the web app
│   ├── style.css                               # CSS styles for the web app
├── README.md                                   # Project documentation
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/amudhan-mohan/Email_Spam_Detection.git
   cd Email_Spam_Detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Train the Model**: Run the following command to train the Random Forest model. This will generate the `spam_classifier.pkl` file in the `backend/model/` directory.
    ```bash
    python backend/train_model.py
    ```
2. **Start the Backend Server**: Launch the Flask backend server to handle API requests.
    ```bash
    python backend/app.py
    ```
3. **Access the Web App**: Open your browser and navigate to `http://localhost:5000` to access the web interface.
4. **Make Predictions**: Enter the email details (email ID, subject, and body) in the web app and click "Submit" to classify the email as spam or ham.

## Dataset
The dataset is located in `backend/dataset/email_dataset_full_with_email.csv`. It contains labeled examples of spam and ham emails.

## Model
The **Random Forest** model is trained on the dataset and saved in the `backend/model/` directory. The model is loaded during runtime to make predictions.

## Model Performance
📊 **Model Accuracy**: 93.18%

📋 **Classification Report**:
```
              precision    recall  f1-score   support

         Ham       0.95      0.91      0.93        23
        Spam       0.91      0.95      0.93        21

    accuracy                           0.93        44
   macro avg       0.93      0.93      0.93        44
weighted avg       0.93      0.93      0.93        44
```

## Contributing
Contributions are welcome! Feel free to submit issues or pull requests to improve the project.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
