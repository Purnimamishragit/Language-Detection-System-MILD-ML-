# import numpy as np
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, classification_report
# import speech_recognition as sr
# from enum import Enum
# from flask import Flask, request, jsonify, render_template

# # Load and preprocess the data
# data = pd.read_csv(r"C:\Users\KIIT\Desktop\personal projects\Languagedetection\Languagedetection\Book.csv", encoding='iso-8859-1')
# if data.isnull().values.any():
#     print("Dataset contains missing values. Handling them...")
#     data.dropna(inplace=True)

# # Features and labels
# x = np.array(data['Text'])
# y = np.array(data['Language'])

# # TF-IDF vectorization
# tfidf = TfidfVectorizer()
# X = tfidf.fit_transform(x)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# # Define models
# models = {
#     "Naive Bayes": MultinomialNB(),
#     "Logistic Regression": LogisticRegression(max_iter=1000),
#     "SVM": SVC(kernel='linear')
# }

# # Train models and calculate metrics
# model_metrics = {}

# for model_name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     report = classification_report(y_test, y_pred, output_dict=True)
#     f1 = report["weighted avg"]["f1-score"]
#     precision = report["weighted avg"]["precision"]
#     recall = report["weighted avg"]["recall"]
#     model_metrics[model_name] = {
#         "accuracy": accuracy,
#         "precision": precision,
#         "recall": recall,
#         "f1_score": f1,
#     }
#     print(f"\n{model_name} Metrics:")
#     print(f"Accuracy: {accuracy:.2f}")
#     print(f"Precision: {precision:.2f}")
#     print(f"Recall: {recall:.2f}")
#     print(f"F1-Score: {f1:.2f}")

# # Select Naive Bayes as the model for Flask API
# best_model_name = "Naive Bayes"
# best_model = models[best_model_name]
# selected_model_metrics = model_metrics[best_model_name]

# print(f"\nSelected Model: {best_model_name}")
# print(f"Metrics: {selected_model_metrics}")

# # Enum for supported languages
# class Language(Enum):
#     ENGLISH = 'en-US'
#     FRENCH = 'fr-FR'
#     GERMAN = 'de-DE'
#     ITALIAN = 'it-IT'
#     SPANISH = 'es-ES'
#     PORTUGUESE = 'pt-BR'

# # Speech-to-Text functionality
# class SpeechToText:
#     @staticmethod
#     def speech_to_text(device_index=1, language=Language.ENGLISH):
#         r = sr.Recognizer()
#         with sr.Microphone(device_index=device_index) as source:
#             print("Recording...")
#             audio = r.listen(source)
#             print("Recording Complete.")
#             try:
#                 text = r.recognize_google(audio, language=language.value)
#                 return text
#             except sr.UnknownValueError:
#                 print("Could not understand the audio.")
#                 return None
#             except sr.RequestError as e:
#                 print(f"Request error: {e}")
#                 return None

# # Flask Application
# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict_text', methods=['POST'])
# def predict_text():
#     data = request.json
#     text = data.get('text')
#     if text:
#         transformed_data = tfidf.transform([text]).toarray()
#         predicted_language = best_model.predict(transformed_data)[0]
#         return jsonify({'language': predicted_language})
#     return jsonify({'error': 'No text provided'}), 400

# @app.route('/predict_audio', methods=['POST'])
# def predict_audio():
#     transcribed_text = SpeechToText.speech_to_text()
#     if transcribed_text:
#         transformed_data = tfidf.transform([transcribed_text]).toarray()
#         predicted_language = best_model.predict(transformed_data)[0]
#         return jsonify({'language': predicted_language})
#     return jsonify({'error': 'Audio not understood'}), 400

# # Standalone functionality
# def predict_language_from_text(input_text):
#     data = tfidf.transform([input_text]).toarray()
#     predicted_language = best_model.predict(data)[0]
#     print(f"Predicted Language: {predicted_language}")

# def predict_language_from_audio(device_index=1, language=Language.ENGLISH):
#     transcribed_text = SpeechToText.speech_to_text(device_index, language)
#     if transcribed_text:
#         data = tfidf.transform([transcribed_text]).toarray()
#         predicted_language = best_model.predict(data)[0]
#         print(f"Predicted Language: {predicted_language}")

# if __name__ == "__main__":
#     app.run(debug=True)


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
import speech_recognition as sr
from enum import Enum
from flask import Flask, request, jsonify, render_template

# Load and preprocess data
data = pd.read_csv(r"C:\Users\KIIT\Desktop\personal projects\Languagedetection\Languagedetection\Booki.csv", encoding='iso-8859-1')
data.dropna(inplace=True)

x = np.array(data['Text'])
y = np.array(data['Language'])

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# TF-IDF and Count Vectorization
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(x)

cv = CountVectorizer(max_features=5000)
X_cv = cv.fit_transform(x).toarray()

# Split data
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, y_encoded, test_size=0.33, random_state=42)
X_train_cv, X_test_cv = train_test_split(X_cv, test_size=0.33, random_state=42)

# Traditional ML Models
models_ml = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel='linear')
}

model_metrics = {}

for model_name, model in models_ml.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    model_metrics[model_name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1_score": report["weighted avg"]["f1-score"]
    }

# Print metrics to terminal
for model_name, metrics in model_metrics.items():
    print(f"{model_name} Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    print("-" * 40)

# CNN Model (without Embedding Layer)
def create_cnn_model(input_dim):
    model = models.Sequential()
    model.add(layers.Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(input_dim, 1)))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(filters=64, kernel_size=5, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(len(label_encoder.classes_), activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

input_dim = X_train_cv.shape[1]
X_train_reshaped = X_train_cv.reshape((X_train_cv.shape[0], X_train_cv.shape[1], 1))
X_test_reshaped = X_test_cv.reshape((X_test_cv.shape[0], X_test_cv.shape[1], 1))

cnn_model = create_cnn_model(input_dim)
cnn_model.fit(X_train_reshaped, y_train, epochs=3, batch_size=32, validation_data=(X_test_reshaped, y_test))

cnn_y_pred = np.argmax(cnn_model.predict(X_test_reshaped), axis=1)
report_cnn = classification_report(y_test, cnn_y_pred, output_dict=True)

model_metrics["CNN"] = {
    "accuracy": report_cnn['accuracy'],
    "precision": report_cnn["weighted avg"]["precision"],
    "recall": report_cnn["weighted avg"]["recall"],
    "f1_score": report_cnn["weighted avg"]["f1-score"]
}

print("CNN Metrics:")
for metric, value in model_metrics["CNN"].items():
    print(f"{metric.capitalize()}: {value:.4f}")
print("-" * 40)

# Speech-to-Text and Flask Integration
class Language(Enum):
    ENGLISH = 'en-US'
    FRENCH = 'fr-FR'

class SpeechToText:
    @staticmethod
    def speech_to_text(device_index=1, language=Language.ENGLISH):
        r = sr.Recognizer()
        with sr.Microphone(device_index=device_index) as source:
            print("Recording...")
            audio = r.listen(source)
            try:
                return r.recognize_google(audio, language=language.value)
            except sr.UnknownValueError:
                print("Could not understand audio")
                return None

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_text', methods=['POST'])
def predict_text():
    text = request.json.get('text')
    if text:
        transformed_data = tfidf.transform([text]).toarray()
        predicted_language = models_ml['Naive Bayes'].predict(transformed_data)[0]
        return jsonify({'language': label_encoder.inverse_transform([predicted_language])[0]})
    return jsonify({'error': 'No text provided'}), 400

if __name__ == "__main__":
    app.run(debug=True)
