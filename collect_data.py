import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils


letters = ['A', 'B', 'C'] 
samples_per_letter = 50  

data = []
labels = []

cap = cv2.VideoCapture(0)

for letter in letters:
    print(f"Prepare to record letter: {letter}")
    input("Press Enter when ready...")
    count = 0
    while count < samples_per_letter:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Flatten landmarks to one list
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                data.append(landmarks)
                labels.append(letter)
                count += 1
                cv2.putText(frame, f"Collected: {count}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        cv2.imshow("Collecting Data", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

np.save('landmarks.npy', np.array(data))
np.save('labels.npy', np.array(labels))
print("Data collection finished!")
