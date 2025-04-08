# Language-Detection-System-MILD-ML-
This project was designed to solve the problem of identifying spoken language in real-time by combining machine learning techniques and an intuitive frontend. A custom dataset was created to train the models, resulting in a 50% accuracy improvement over standard datasets. The final system compares multiple models to select the best-performing one.
A robust and real-time system to detect spoken languages using multiple machine learning models and modalities. This project leverages traditional ML classifiers and deep learning (CNN) to accurately identify spoken language from audio input, along with a user-friendly interface for seamless interaction.

## 🚀 Project Overview

This project was designed to solve the problem of identifying spoken language in real-time by combining machine learning techniques and an intuitive frontend. A custom dataset was created to train the models, resulting in a 50% accuracy improvement over standard datasets. The final system compares multiple models to select the best-performing one.

## 🔍 Features

- 🎯 Accurate detection of spoken languages from audio input
- 🤖 Utilizes ML algorithms: Multinomial Naive Bayes, Random Forest, and CNN
- 🧠 Custom dataset designed for improved performance
- 🖥️ Real-time prediction through an interactive web interface
- 📊 Comparative analysis to evaluate model performance

## 🛠️ Technologies Used

- **Python 3.7+**
- **NumPy, Pandas, Scikit-learn** - For data preprocessing and ML models
- **TensorFlow / Keras** - For implementing CNN
- **Flask / Streamlit** - For building the frontend interface
- **Librosa** - For audio feature extraction
- **Matplotlib / Seaborn** - For visualization and analysis

## 📂 Project Structure

multi-modal-language-detection/ ├── dataset/ │ └── custom_audio_files/ ├── models/ │ ├── naive_bayes_model.pkl │ ├── random_forest_model.pkl │ └── cnn_model.h5 ├── src/ │ ├── data_preprocessing.py │ ├── train_models.py │ ├── evaluate_models.py │ └── audio_predictor.py ├── interface/ │ └── app.py (Flask or Streamlit UI) ├── requirements.txt └── README.md

## 📊 Model Comparison

| Model                  | Accuracy |
|-----------------------|----------|
| Multinomial Naive Bayes | 78%     |
| Random Forest          | 84%     |
| CNN                    | 92%  |

> CNN achieved the highest accuracy due to its ability to capture deeper patterns from audio spectrograms.

## 🧪 How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/multi-modal-language-detection.git
   cd multi-modal-language-detection
Install Dependencies
pip install -r requirements.txt
Train Models
python src/train_models.py
Run the Interface
streamlit run interface/app.py
Upload an Audio File
Use the interface to upload an audio file and get real-time predictions.

🏆 Achievements
📄 Research paper co-authored and presented at IEEE GC4T25

🚀 Achieved 50% improvement in accuracy through custom dataset design

🎓 Mentored peers, improving student performance by 60%

👩‍💻 Author
Purnima
3rd Year B.Tech CSE Student, Silicon University


![Screenshot 2024-12-07 153727](https://github.com/user-attachments/assets/23aef3d0-f34e-4da0-8588-f0ca10acb697)
![Screenshot 2024-12-07 103126](https://github.com/user-attachments/assets/b68647b8-d9b5-4681-ad4c-460e3347b53d)

![audio-ezgif com-optimize](https://github.com/user-attachments/assets/61a44622-0b44-4902-acbe-8adc2a98290d)
![text-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/79293320-a497-44b7-8cd5-19a00d652638)

