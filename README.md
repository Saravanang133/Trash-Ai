# ♻ Trash AI: Camera Vision Based Trash Classification

*Trash AI* is a smart waste management system designed to detect, classify, and segregate trash in real-time using Deep Learning. It helps municipalities monitor waste accumulation and alerts cleaning agents automatically.

## 🛠 Tech Stack
* *Language:* Python 3.7+
* *Frameworks:* Flask (Web App), TensorFlow/Keras (Deep Learning)
* *Database:* MySQL
* *Frontend:* HTML, CSS, JavaScript, Bootstrap
* *Tools:* OpenCV, Pandas, NumPy

## ✨ Key Features
* *TrashNet Model:* Uses Convolutional Neural Networks (CNN) to classify waste into categories like Plastic, Paper, Metal, etc.
* *Real-Time Detection:* Integrates with CCTV cameras to detect trash on roads or bins using Temporal Convolutional Networks (TCN).
* *Municipality Dashboard:* A web-based admin panel to monitor waste levels and manage collection routes.
* *Automated Alerts:* Sends *SMS and Email notifications* to cleaning agents when trash is detected.
* *Smart Segregation:* Helps in separating biodegradable and non-biodegradable waste effectively.

## 🚀 How to Run
1. Install Python and required libraries:
   ```bash
   pip install flask tensorflow opencv-python mysql-connector-python
