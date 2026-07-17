import cv2
from cvzone.HandTrackingModule import HandDetector
import joblib

# Load the AI model
try:
    model = joblib.load("word_recognition_model.pkl")
    print("AI Word Brain Loaded Successfully!")
except:
    print("Error: Train the model first using train_words.py!")
    exit()

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, detectionCon=0.8)

sentence = []
last_word = None
word_stable_counter = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    
    hands, frame = detector.findHands(frame, draw=True)
    display_text = "Sentence: " + " ".join(sentence)
    
    if hands:
        hand = hands[0]
        row = []
        for lm in hand['lmList']:
            row.extend([lm[0], lm[1], lm[2]])
            
        # AI instantly predicts which word matches these coordinates
        predicted_word = model.predict([row])[0]
        
        # Sentence Building Engine
        if predicted_word == last_word:
            word_stable_counter += 1
            if word_stable_counter == 25: # Held for ~1 second
                if not sentence or sentence[-1] != predicted_word:
                    sentence.append(predicted_word)
        else:
            last_word = predicted_word
            word_stable_counter = 0
            
        cv2.putText(frame, f"Sign: {predicted_word}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    cv2.putText(frame, display_text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("AI Smart Word Translator", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()