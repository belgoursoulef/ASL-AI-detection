import os
import cv2
import mediapipe as mp
import numpy as np

DATASET_PATH = "asl_alphabet_test"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

data = []
labels = []

for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(label_path):
        continue

    print(f"Processing letter: {label}")

    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                data.append(landmarks)
                labels.append(label)

print("Total samples collected:", len(data))

np.save("landmarks.npy", np.array(data))
np.save("labels.npy", np.array(labels))

print("✅ Conversion complete!")
print("Saved: landmarks.npy and labels.npy")
