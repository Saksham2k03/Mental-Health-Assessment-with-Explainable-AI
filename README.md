# Mental-Health-Assessment-with-Explainable-AI
This repository contains the B.Tech. final year project titled "Emotion-Based Mental Health Assessment with Explainable AI (XAI)," developed at Netaji Subhas University of Technology (NSUT), New Delhi.
🌟 Overview
Mental health is a critical component of overall well-being, yet traditional assessment methods are often subjective, inaccessible, and reactive. This project introduces a proactive approach to mental health monitoring by developing a multimodal AI system that analyzes emotional states in real-time.



Our system leverages deep learning to interpret emotional cues from both 

facial expressions and vocal tones. The cornerstone of this project is its commitment to transparency through 


Explainable AI (XAI). By integrating techniques like SHAP, LIME, and Grad-CAM, we make the AI's decision-making process understandable, fostering trust among users and clinicians.




The ultimate goal is to create a supportive tool that provides early warnings for emotional distress, encouraging individuals to take proactive steps toward emotional wellness.



✨ Key Features
Multimodal Analysis: The system fuses two critical channels for a more accurate and holistic emotional assessment:


Facial Emotion Recognition (FER): A VGGNet-based Convolutional Neural Network (CNN) analyzes real-time video to classify facial expressions into seven emotional states.





Voice Emotion Recognition (VER): A hybrid CNN-LSTM model processes audio features (MFCCs) to identify emotions from speech patterns, independent of the content.



Explainability and Transparency: We address the "black box" problem in AI by implementing leading XAI techniques:


Grad-CAM: Generates heatmaps on facial images to visually highlight the specific regions (e.g., eyes, mouth) that influence an emotion prediction.





SHAP & LIME: Quantify the contribution of different audio features to a voice emotion prediction, providing clear, numerical justifications.



High Performance: The combined multimodal framework demonstrates robust performance with an overall classification accuracy of 91.42%.


Real-Time Processing: The architecture is designed for real-time analysis, making it suitable for interactive applications.


🔧 Tech Stack & Methodology
Frameworks & Libraries: Python, TensorFlow, Keras, OpenCV, Librosa, Scikit-learn.

Models:

FER: VGGNet-based CNN 

VER: CNN + LSTM Hybrid 


XAI Techniques: SHAP, LIME, Grad-CAM 

Datasets:

FER: FER2013, CK+ 


VER: RAVDESS 


Preprocessing:


Images: Face detection, resizing to 48
times48, normalization, and data augmentation (flipping, rotation).



Audio: Noise reduction and extraction of Mel-Frequency Cepstral Coefficients (MFCCs).

⚙️ System Pipeline
The system operates through two parallel branches that are fused for a final, unified prediction:

Facial Branch (VGGNet):

Input video is processed frame-by-frame.

A face is detected and extracted.

The facial image is fed into the VGGNet model to predict a facial emotion.

Grad-CAM provides a heatmap visualizing the decision.

Vocal Branch (CNN-LSTM):

Input audio is captured and preprocessed to extract MFCCs.

The MFCC spectrogram is fed into the CNN-LSTM model to predict a vocal emotion.

SHAP explains which audio features were most influential.

Multimodal Fusion:

The feature vectors from both branches are concatenated and passed through a final dense network to yield a single, robust emotion classification.

📊 Results
The models demonstrated strong performance, with the ability to accurately distinguish between nuanced emotions.

Metric (Combined Inference)	Score
Accuracy		
91.42% 

Precision (Happy)	
0.92 

Recall (Happy)	
0.90 

F1-Score (Happy)	
0.91 

Precision (Sad)	
0.89 

Recall (Sad)	
0.88 

F1-Score (Sad)	
0.885 


Export to Sheets
XAI-Generated Heatmap (Grad-CAM):


This image shows how the model focuses on key areas like the mouth for "Happy" and "Surprise," and eyebrows for "Angry".



🚀 Future Scope

Incorporate More Modalities: Extend the system to include text sentiment analysis and physiological signals (e.g., heart rate) for an even more comprehensive assessment.


Longitudinal Monitoring: Develop temporal models to track emotional trends over time, helping to detect early signs of chronic conditions like depression or anxiety.


User-Centric XAI Interfaces: Design intuitive dashboards to translate XAI outputs into actionable feedback for users.


Deployment: Optimize and deploy the model in mobile and web applications, ensuring privacy and compliance with standards like GDPR and HIPAA.


👥 Authors
Shashank Kumar ([Your LinkedIn/GitHub Profile Link])

Saksham Raj ([Your LinkedIn/GitHub Profile Link])

Under the supervision of Dr. Anand Gupta, Department of Computer Science & Engineering, NSUT.
