# Emotion-Based Mental Health Assessment with Explainable AI (XAI)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)
![Frameworks](https://img.shields.io/badge/frameworks-TensorFlow%20%7C%20Keras-orange.svg)

A multimodal deep learning framework for real-time mental health assessment by analyzing facial expressions and vocal tones, with a core focus on transparency through Explainable AI (XAI)[cite: 228, 288]. This project was developed as a B.Tech. final year project at Netaji Subhas University of Technology (NSUT), New Delhi[cite: 241, 248].

---

## 🚀 The Problem

Traditional mental health assessments often rely on subjective self-reports, which can be inconsistent, biased, and inaccessible due to stigma or cost[cite: 22, 284]. Furthermore, most AI-driven solutions act as "black boxes," providing predictions without transparent reasoning, which limits their trustworthiness in sensitive healthcare applications[cite: 24, 311].

---

## 💡 Our Solution

This project introduces a proactive and transparent approach to mental health monitoring. By fusing insights from multiple emotional channels and making the AI's logic interpretable, we aim to create a reliable tool for early emotional distress detection[cite: 17, 19].

### Key Features
* **🧠 Multimodal Emotion Recognition**: The system integrates two parallel deep learning models for a holistic emotional analysis:
    * **Facial Emotion Recognition (FER)**: A VGGNet-based CNN classifies emotions from real-time facial expressions[cite: 36, 287].
    * **Voice Emotion Recognition (VER)**: A hybrid CNN-LSTM architecture analyzes vocal features like tone and pitch to detect emotion in speech[cite: 80, 287].
* **🔍 Explainable AI (XAI) for Transparency**: To build trust and enable validation, the model's decisions are made transparent[cite: 288].
    * **Grad-CAM** generates visual heatmaps on faces, highlighting the key regions (e.g., eyes, mouth) that influence a prediction[cite: 205, 516].
    * **SHAP and LIME** assign importance scores to vocal features, explaining *why* a certain emotion was detected from audio[cite: 205, 414].
* **🏆 High Performance**: The integrated multimodal system achieves a robust overall accuracy of **91.42%** on test datasets[cite: 698].

---

## ⚙️ System Architecture

The system processes video and audio inputs through two distinct branches before fusing the results for a final classification.

1.  **Facial Branch** → A face is detected from a video frame, preprocessed, and fed into the **VGGNet model**[cite: 34, 454].
2.  **Vocal Branch** → Audio is preprocessed to extract **MFCCs**, which are then fed into the **CNN-LSTM model**[cite: 78, 464].
3.  **Fusion** → Feature vectors from both branches are concatenated and passed to a final decision network for a unified and robust emotion prediction[cite: 493].

---

## 🛠️ Technology Stack

* **Core Frameworks**: Python, TensorFlow, Keras
* **Data Processing**: OpenCV, Librosa, Scikit-learn
* **AI Models**: VGGNet, CNN-LSTM
* **XAI Libraries**: SHAP, LIME, Grad-CAM
* **Datasets Used**:
    *FER2013 & CK+ (for facial expressions) [cite: 36, 438]
    * RAVDESS (for voice emotions) [cite: 82, 439]

---

## 📊 Results & Explainability

The model effectively distinguishes between seven core emotions. The confusion matrix below shows high per-class accuracy, such as 98.61% for 'Angry' and 97.51% for 'Happy'[cite: 42].

#### Grad-CAM Visualization
The following image demonstrates the XAI in action. [cite_start]The heatmaps show the exact facial regions the model focused on to make its prediction, making the decision process transparent[cite: 205].

![Grad-CAM Heatmaps](https://github.com/user-attachments/assets/4c564753-7378-4ac7-ad06-08c60ac4e835)
*Heatmaps for 'Disgust', 'Happy', 'Sad', 'Surprise', and 'Angry' emotions.*

---

## 📦 Setup and Installation

To get a local copy up and running, follow these simple steps.

1.  **Clone the repo**
    ```sh
    git clone [https://github.com/your_username/your_repository.git](https://github.com/your_username/your_repository.git)
    ```
2.  **Navigate to the project directory**
    ```sh
    cd your_repository
    ```
3.  **Install Python packages**
    ```sh
    pip install -r requirements.txt
    ```

---

## ▶️ Usage

1.  Download the required datasets and place them in the `data/` directory.
2.  Run the training script to train the models:
    ```sh
    python train.py
    ```
3.  Launch the real-time application:
    ```sh
    python app.py
    ```

---

## 🔮 Future Work

We have identified several directions for enhancing this project:
* **Incorporate more modalities** like text sentiment and physiological signals[cite: 218, 715].
* **Develop longitudinal tracking** to monitor emotional trends over time for early detection of chronic conditions like depression or ADHD[cite: 719].
* **Design user-centric XAI interfaces** to make the insights more accessible to non-technical users[cite: 221, 721].
* **Deploy as a mobile or web application** with a focus on privacy and on-device processing[cite: 222, 723].

---

## 👥 Authors & Acknowledgments

This project was created by:
* **Saksham Raj** (2021UCS1713)
* **Shashank Kumar** (2021UCS1710) 

We extend our sincere gratitude to our supervisor, **Dr. Anand Gupta**, for his invaluable guidance and support throughout this project[cite: 6, 273].

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
