import cv2
from cvzone.HandTrackingModule import HandDetector

# Initialize the webcam and the hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, detectionCon=0.8)

# Define your word dictionary based on standard finger counts/postures
# You can customize these to match specific hand actions!
# Expanded dictionary mapping [Thumb, Index, Middle, Ring, Pinky] to full words
word_dictionary = {
    # HELLO: Just index finger up (allowing thumb to be 0 or 1)
    (0, 1, 0, 0, 0): "HELLO",
    (1, 1, 0, 0, 0): "HELLO",
    
    # THANK YOU: Index and Middle finger up (Victory/Peace sign)
    (0, 1, 1, 0, 0): "THANK YOU",
    (1, 1, 1, 0, 0): "THANK YOU",
    
    # YES: Making a Fist, but sticking just the THUMB up (Thumbs Up gesture!)
    (1, 0, 0, 0, 0): "YES",
    
    # NO: Sticking just the PINKY finger up
    (0, 0, 0, 0, 1): "NO",
    
    # HELP: All fingers completely open
    (1, 1, 1, 1, 1): "HELP",
    
    # CLEAR: Complete flat closed fist
    (0, 0, 0, 0, 0): "CLEAR"
}

sentence = []
last_word = None
word_stable_counter = 0

print("Word-level Translator Engine Started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    hands, frame = detector.findHands(frame, draw=True)
    
    display_text = "Sentence: " + " ".join(sentence)
    
    if hands:
        hand = hands[0]
        # Get a list of which fingers are up: [Thumb, Index, Middle, Ring, Pinky]
        # 1 = Up, 0 = Down
        fingers = detector.fingersUp(hand)
        
        # Match the finger configuration to our word dictionary
        detected_word = word_dictionary.get(tuple(fingers), None)
        
        if detected_word:
            if detected_word == "CLEAR":
                sentence = []
                last_word = None
            elif detected_word == last_word:
                word_stable_counter += 1
                # If held for ~20 frames, confirm the word
                if word_stable_counter == 20:
                    if not sentence or sentence[-1] != detected_word:
                        sentence.append(detected_word)
            else:
                last_word = detected_word
                word_stable_counter = 0
                
            cv2.putText(frame, f"Action: {detected_word}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Render the active sentence building on the screen
    cv2.putText(frame, display_text, (10, 450), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
    cv2.imshow("Full Word Sign Translator", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()