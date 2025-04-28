# Arthamudhra - Malayalam Sign Language Translator  

> Empowering communication through technology.

---

## 📖 About the Project  
**Arthamudhra** is a real-time **Malayalam Sign Language Translator** developed to bridge the communication gap for the deaf and mute community. The system captures hand gestures through a webcam, processes them using **OpenCV** and **MediaPipe Hands**, and translates them into Malayalam text using a trained **Random Forest Classifier**. It offers a simple, accessible web interface built with **Flask**, allowing seamless interaction.

---

## 🚀 Features  
1. Real-time hand gesture recognition  
2. Translation of gestures into Malayalam text  
3. Intuitive and responsive web-based user interface  
4. Live camera feed with text overlay  
5. Option to clear translated text and restart camera feed  
6. Malayalam font support for correct script rendering  

---

## 🛠️ Built With  
- Python 3.8+  
- Flask (Web Framework)  
- OpenCV (Real-time video capture and processing)  
- MediaPipe Hands (Hand landmark detection)  
- Scikit-Learn (Random Forest Classifier for gesture recognition)  
- Pillow (Malayalam font rendering)  
- HTML, CSS, JavaScript (Frontend development)

---

## 📂 Project Structure  
```
├── app.py                # Main Flask backend application
├── clt_image.py          # Script for dataset image collection (OpenCV)
├── dataset.py            # Script for feature extraction (MediaPipe)
├── training.py           # Script for training the machine learning model
├── templates/
│   ├── index.html        # Home page
│   ├── camera.html       # Camera and live translation page
│   ├── about.html        # About the project
│   ├── contact.html      # Contact page
├── static/               # Static CSS and JS files (optional)
├── model.p               # Trained Random Forest model
├── data.pickle           # Preprocessed feature dataset
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 🧠 How It Works  
1. Collect gesture images using OpenCV.  
2. Extract hand landmark features using MediaPipe.  
3. Train a Random Forest Classifier on the features.  
4. Predict hand gestures in real-time and display translated text.  

---

## ⚙️ Installation and Running Locally  

1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/arthamudhra.git
   cd arthamudhra
   ```

2. Install required dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask application:  
   ```bash
   python app.py
   ```

4. Open your browser and visit:  
   ```
   http://localhost:5000/
   ```

---

## 🎯 Requirements  
- A working webcam  
- Proper lighting conditions  
- Malayalam font file (`MLW-TTRevathi.ttf`) placed correctly in the project directory  
- Python 3.8 or above installed  

---

## 🤝 Contributions  
Contributions and suggestions are welcome!  
Please feel free to fork this repository and submit a pull request.

---

## 📄 License  
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 💬 Acknowledgments  
- Inspired by the mission to make technology accessible to everyone.  
- Developed with dedication, teamwork, and a commitment to inclusion.
