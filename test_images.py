import os
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


DATASET_PATH = "asl_alphabet_test"

model = load_model("sign_model.h5")
labels = sorted(list(set(np.load("labels.npy"))))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

for letter in os.listdir(DATASET_PATH):
    letter_path = os.path.join(DATASET_PATH, letter)

    if not os.path.isdir(letter_path):
        continue

    for img_name in os.listdir(letter_path):
        img_path = os.path.join(letter_path, img_name)
        image = cv2.imread(img_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                landmarks = np.array(landmarks).reshape(1, -1)
                prediction = model.predict(landmarks)
                predicted = labels[np.argmax(prediction)]

                print(f"Image: {img_name} | True: {letter} | Predicted: {predicted}")
