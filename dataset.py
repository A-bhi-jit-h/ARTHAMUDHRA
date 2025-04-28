import os
import pickle
import mediapipe as mp
import cv2

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = r"C:\Users\Asus\OneDrive\Desktop\PROJECT MAIN\data"

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_left, y_left = [], []
        x_right, y_right = [], []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        if img is None:
            continue  # Skip unreadable images

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

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
                # If a hand is missing, append zeros for consistency
                hand_data.extend([0] * 42)
            return hand_data

        # Extract both hands' features
        data_aux.extend(extract_features(left_hand, x_left, y_left))
        data_aux.extend(extract_features(right_hand, x_right, y_right))

        data.append(data_aux)
        labels.append(dir_)

# Save the dataset
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Dataset saved with {len(data)} samples.")
