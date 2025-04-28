from flask import Flask, render_template, Response, redirect, url_for
import cv2
import pickle
import mediapipe as mp
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)

# Load the trained model
model_dict = pickle.load(open(r"C:\Users\Asus\OneDrive\Desktop\PROJECT MAIN\model.p", "rb"))  # Ensure model path is correct
model = model_dict["model"]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Load Malayalam font
malayalam_font_path = r"C:\Users\Asus\OneDrive\Desktop\PROJECT MAIN\MLW-TTRevathi.ttf"

# Label dictionary (update as per your dataset)
labels_dict = {
    0: "A", 1: "B", 2: "C", 3: " ", 4: "D", 5: "E", 6: "F",
    7: "G", 8: "H", 9: "I", 10: "J", 11: "K", 12: "L", 13: "M",
    14: "N", 15: "O", 16: "P", 17: "R", 18: "S", 19: "T", 20: "U",
    21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z", 26: "[", 27: '\\',
    28: "]", 29: "^", 30: "_", 31: "/", 32: "a", 33: "b", 34: "c",
    35: "e", 36: "h", 37: "i", 38: "j", 39: "k", 40: "l", 41: "f",
    42: "g", 43: "d", 44: "m", 45: "n", 46: "0", 47: "p", 48: "q",
    49: "s", 50: "t", 51: "y", 52: "v", 53:"r"
}

camera = None  # Initialize camera as None
confirmed_word = ""  # Store the translated text

def initialize_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)

def generate_frames():
    global confirmed_word
    last_prediction = None
    prediction_start_time = None

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        data_aux = []
        x_left, y_left = [], []
        x_right, y_right = [], []

        left_hand, right_hand = None, None
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = results.multi_handedness[idx].classification[0].label
                if label == "Left":
                    left_hand = hand_landmarks
                else:
                    right_hand = hand_landmarks

        def extract_features(hand_landmarks, x_list, y_list):
            hand_data = []
            if hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x_list.append(landmark.x)
                    y_list.append(landmark.y)
                for landmark in hand_landmarks.landmark:
                    hand_data.append(landmark.x - min(x_list))
                    hand_data.append(landmark.y - min(y_list))
            else:
                hand_data.extend([0] * 42)  # Fill missing hand with zeros
            return hand_data

        data_aux.extend(extract_features(left_hand, x_left, y_left))
        data_aux.extend(extract_features(right_hand, x_right, y_right))

        predicted_character = ""

        if left_hand or right_hand:
            data_aux = np.asarray(data_aux).reshape(1, -1)
            prediction = model.predict(data_aux)[0]
            predicted_character = labels_dict[int(prediction)]

            if prediction == last_prediction:
                elapsed_time = time.time() - prediction_start_time
                if elapsed_time >= 2:
                    confirmed_word += predicted_character
                    last_prediction = None
            else:
                last_prediction = prediction
                prediction_start_time = time.time()

        # Convert OpenCV frame to PIL for Malayalam text overlay
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype(malayalam_font_path, 50)
        except:
            print("Error: Malayalam font not found!")
            break

        # Display translation result
        draw.text((50, 50), predicted_character, font=font, fill=(0, 255, 0))
        draw.text((50, 400), confirmed_word, font=font, fill=(255, 0, 0))

        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Convert frame to JPEG format for streaming
        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/")
def home():
    return render_template("index1.html")

@app.route("/camera")
def camera_page():
    initialize_camera()  # ✅ Ensure camera is reinitialized
    return render_template("camera.html")

@app.route("/camera_feed")
def camera_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/clear_text")
def clear_text():
    global confirmed_word
    confirmed_word = ""  # Clear translated text
    return redirect(url_for("camera_page"))

@app.route("/stop_camera")
def stop_camera():
    global camera
    if camera is not None:
        camera.release()  # ✅ Properly release the camera
        camera = None  # ✅ Reset camera so it can be restarted
    return redirect(url_for("home"))
@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/sign")
def sign():
    return render_template("sign.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
