import streamlit as st
import cv2
from cvzone.HandTrackingModule import HandDetector
import joblib
import numpy as np

# 1. Setup the Web Page Layout
st.set_page_config(page_title="Sign Language Translator", layout="centered")
st.title("🤟 Real-Time Sign Language Translator")
st.write("Hold your hand up to the webcam to translate gestures instantly.")

# 2. Load your custom Machine Learning brain
@st.cache_resource
def load_model():
    return joblib.load("word_recognition_model.pkl")

model = load_model()
detector = HandDetector(maxHands=1, detectionCon=0.8)

# 3. Handle the live video stream placeholder in the browser
FRAME_WINDOW = st.image([]) 
sentence_placeholder = st.empty()

# 4. Initialize state variables for tracking words
if "sentence" not in st.session_state:
    st.session_state.sentence = []

# Create a toggle button to start/stop the camera feed
run_camera = st.checkbox("Turn on Webcam", value=True)
camera = cv2.VideoCapture(0)

while run_camera:
    success, frame = camera.read()
    if not success:
        st.warning("Failed to access the webcam.")
        break
        
    # Process the frame
    frame = cv2.flip(frame, 1)
    hands, frame = detector.findHands(frame)
    
    if hands:
        hand = hands[0]
        lmList = hand['lmList']
        
        # Flatten your 63 landmarks (21 points * 3 coordinates)
        data = np.array(lmList).flatten().reshape(1, -1)
        
        # Make the live AI prediction
        prediction = model.predict(data)
        detected_word = prediction[0]
        
        # Draw the text on the browser video frame
        cv2.putText(frame, detected_word, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # (Optional: Add your custom stabilization logic here before appending to sentence)
        
    # Convert OpenCV BGR to Web RGB format and update the browser window
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
    
    # Render sentence output underneath the camera view
    sentence_placeholder.markdown(f"### Detected Sentence: **{' '.join(st.session_state.sentence)}**")

else:
    camera.release()
    st.write("Webcam is turned off.")
