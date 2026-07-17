import cv2
from cvzone.HandTrackingModule import HandDetector
import csv
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, detectionCon=0.8)

# Get the word label from the user
word_label = input("Enter the name of the WORD you want to train: ").upper().strip()
dataset_file = "words_dataset.csv"

# Create CSV file with headers if it doesn't exist
if not os.path.exists(dataset_file):
    with open(dataset_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        headers = [f"lm_{i}" for i in range(21 * 3)] + ["Label"]
        writer.writerow(headers)

print(f"\nReady to record for '{word_label}'.")
print("Position your hand in the camera frame and press 's' to start saving 100 frames...")

recording = False
counter = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    
    hands, frame = detector.findHands(frame, draw=True)
    
    if hands and recording:
        hand = hands[0]
        lm_list = hand['lmList'] # Get all 21 landmark points (x, y, z)
        
        # Flatten the list into a single row of 63 numbers
        row = []
        for lm in lm_list:
            row.extend([lm[0], lm[1], lm[2]])
        row.append(word_label)
        
        # Append data directly to CSV
        with open(dataset_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
        counter += 1
        cv2.putText(frame, f"Saved: {counter}/100", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if counter >= 100:
            print(f"Successfully recorded 100 samples for {word_label}!")
            break
            
    elif not recording:
        cv2.putText(frame, "Press 'S' to Start Recording", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Data Collector", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') or key == ord('S'):
        recording = True
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()