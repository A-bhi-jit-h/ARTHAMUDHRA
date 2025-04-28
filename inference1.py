import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont

# Load the trained model
model_dict = pickle.load(open(r"C:\Users\Asus\OneDrive\Desktop\PROJECT MAIN\model.p", "rb"))  
model = model_dict["model"]

cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Malayalam labels
labels_dict = {0: "A", 1: "`", 2: "n", 3: "P", 4: "X", 5: "B", 6: " "}

# Load Malayalam font
malayalam_font_path = r"C:\Users\Asus\OneDrive\Desktop\PROJECT MAIN\MLW-TTRevathi.ttf"

last_prediction = None
prediction_start_time = None
confirmed_word = ""

while True:
    ret, frame = cap.read()
    if not ret:
        continue

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
            hand_data.extend([0] * 42)  
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

    # Save translated text to a file
    with open(r"C:\Users\Asus\OneDrive\Desktop\PROJECT MAIN\translation.txt", "w", encoding="utf-8") as f:
        f.write(confirmed_word)

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("d"):
        confirmed_word = ""

cap.release()
cv2.destroyAllWindows()
