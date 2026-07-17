# collect_data.py
import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# CSV File Path
CSV_PATH = "dataset.csv"

# Request class label
class_label = input("Enter the label name for this sign (e.g. 'hello', 'thanks', 'a', 'b'): ").strip().lower()

# Prepare CSV header if file doesn't exist
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        # 21 landmarks * 3 coordinates (x, y, z) + 1 label column = 64 columns
        header = [f"point_{i}_{coord}" for i in range(21) for coord in ['x', 'y', 'z']] + ['label']
        writer.writerow(header)

cap = cv2.VideoCapture(0)
print(f"\n---> Preserving Class: '{class_label.upper()}'")
print("---> HOLD UP your sign, then PRESS and HOLD the 'S' key to record data.")
print("---> Press 'Q' to quit.")

collected_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # If hands detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract and normalize coordinates relative to Wrist (Landmark 0)
            wrist = hand_landmarks.landmark[0]
            landmarks = []
            for lm in hand_landmarks.landmark:
                # Normalizing positions relative to the wrist
                landmarks.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

            # If user holds 'S', write features to CSV
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') or (cv2.waitKey(1) & 0xFF == ord('s')):
                with open(CSV_PATH, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(landmarks + [class_label])
                collected_count += 1
                cv2.putText(frame, f"RECORDING... Samples: {collected_count}", (10, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame, f"Class: {class_label.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Hold 'S' to Save. Q to quit.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow("Data Collector Panel", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Successfully collected {collected_count} coordinate frames for '{class_label}'!")