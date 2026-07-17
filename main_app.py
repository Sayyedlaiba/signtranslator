# main_app.py
import os
import sys
print(sys.executable)
print(sys.version)
import cv2
print(cv2.__version__)
import pickle
import numpy as np
import tkinter as tk
from tkinter import ttk
import mediapipe as mp
from PIL import Image, ImageTk
import cv2
from cvzone.HandTrackingModule import HandDetector
import joblib
from PIL import Image, ImageTk
# ... keep your other existing tkinter imports here

class DoubleWayTranslator:
    def __init__(self, root):
        # Store root window reference
        self.root = root
        
        # Initialize UI elements first
        self.setup_ui()
        
        # Load the new Smart Word AI Brain
        try:
            self.word_model = joblib.load("word_recognition_model.pkl")
            print("AI Word Brain Loaded into GUI Successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.word_model = None

        # Initialize the cvzone Hand Detector
        self.detector = HandDetector(maxHands=1, detectionCon=0.8)

        # Variables for building words and sentences smoothly
        self.sentence_list = []
        self.last_word = None
        self.word_stable_counter = 0
        self.hold_threshold = 25  # Held for ~1 second to confirm the word

        # Camera initialization
        self.cap = cv2.VideoCapture(0)

        # Start the camera loop AFTER all parameters are built
        self.start_camera_loop()
    def setup_ui(self):
        # Header title
        header = tk.Label(self.root, text="🤟 Bidirectional Sign Language Bridge 🤟", 
                          font=("Helvetica", 20, "bold"), bg="#11111b", fg="#cdd6f4")
        header.pack(pady=10)

        # Main horizontal split
        main_frame = tk.Frame(self.root, bg="#11111b")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        main_frame.pack_propagate(False)

        # ================= LEFT PANEL: SIGN TO TEXT (CAMERA) =================
        left_panel = tk.Frame(main_frame, bg="#1e1e2e", bd=2, relief="groove")
        left_panel.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        lt_title = tk.Label(left_panel, text="Sign-to-Text (Live Camera)", font=("Helvetica", 14, "bold"), bg="#1e1e2e", fg="#a6e3a1")
        lt_title.pack(pady=5)

        self.cam_label = tk.Label(left_panel, bg="#11111b", width=440, height=330)
        self.cam_label.pack(pady=5)
        # self.cam_label.pack_propagate(False)

        # Detected Output Display
        self.output_text_var = tk.StringVar(value="Sign Detected: None")
        self.output_box = tk.Label(left_panel, textvariable=self.output_text_var, font=("Helvetica", 16, "bold"), bg="#313244", fg="#f5c2e7", height=2)
        self.output_box.pack(fill="x", padx=15, pady=10)

        # ================= RIGHT PANEL: TEXT TO SIGN (ANIMATION) =================
        right_panel = tk.Frame(main_frame, bg="#1e1e2e", bd=2, relief="groove")
        right_panel.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        rt_title = tk.Label(right_panel, text="Text-to-Sign (Visualizer)", font=("Helvetica", 14, "bold"), bg="#1e1e2e", fg="#89b4fa")
        rt_title.pack(pady=5)

        self.anim_canvas = tk.Canvas(right_panel, width=300, height=300, bg="#11111b", highlightthickness=0)
        self.anim_canvas.pack(pady=5)
        self.canvas_placeholder_text = self.anim_canvas.create_text(150, 150, text="Type a message\nbelow to begin...", fill="#a6adc8", font=("Helvetica", 13), justify="center")

        # Translation status monitor
        self.anim_status_var = tk.StringVar(value="Status: Idle")
        self.status_label = tk.Label(right_panel, textvariable=self.anim_status_var, font=("Helvetica", 11, "italic"), bg="#1e1e2e", fg="#bac2de")
        self.status_label.pack(pady=5)

        # Interactive user entry
        input_frame = tk.Frame(right_panel, bg="#1e1e2e")
        input_frame.pack(pady=10)

        self.text_input = ttk.Entry(input_frame, font=("Helvetica", 12), width=24)
        self.text_input.pack(side="left", padx=5)
        self.text_input.insert(0, "hello clean example")
        self.text_input.bind("<Return>", lambda e: self.trigger_translation())

        translate_btn = ttk.Button(input_frame, text="Translate", command=self.trigger_translation)
        translate_btn.pack(side="right", padx=5)

    # Camera processing loop for Sign-to-Text
    def start_camera_loop(self):
        ret, frame = self.cap.read()
        if not ret:
            self.output_text_var.set("Status: Failed to grab frame from camera")
            self.root.after(10, self.start_camera_loop)
            return

        # Flip frame for natural mirror view
        frame = cv2.flip(frame, 1)
        
        # Detect hands using cvzone within the GUI loop
        hands, frame = self.detector.findHands(frame, draw=True)
        
        # Pre-build display string
        display_text = "Sentence: " + " ".join(self.sentence_list)
        
        if hands and self.word_model:
            hand = hands[0]
            row = []
            for lm in hand['lmList']:
                row.extend([lm[0], lm[1], lm[2]])
            
            try:
                # Predict the word using your trained AI brain
                predicted_word = self.word_model.predict([row])[0]
                
                # Sentence Stabilization Engine
                if predicted_word == self.last_word:
                    self.word_stable_counter += 1
                    if self.word_stable_counter == self.hold_threshold:
                        # Append word if sentence is empty or if it's not a duplicate of the immediate last word
                        if not self.sentence_list or self.sentence_list[-1] != predicted_word:
                            self.sentence_list.append(predicted_word)
                else:
                    self.last_word = predicted_word
                    self.word_stable_counter = 0
                
                # Update Tkinter interface with live detection text
                self.output_text_var.set(f"Sign Detected: {predicted_word} | {display_text}")
            except Exception:
                self.output_text_var.set(f"Processing... | {display_text}")
        else:
            # If no hands are on screen, just display the compiled sentence
            self.output_text_var.set(display_text)

        # Convert OpenCV image format to Tkinter compatible format
        resized_frame = cv2.resize(frame, (440, 330))
        img = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.cam_label.imgtk = imgtk
        self.cam_label.configure(image=imgtk)

        # Loop the function every 10 milliseconds
        self.root.after(10, self.start_camera_loop)

    def results_drawn(self, frame, results):
        for hand_landmarks in results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return results.multi_hand_landmarks

    # Parsing engine for Text-to-Sign
    def trigger_translation(self):
        # Terminate running animation sequence loops
        if self.active_animation_loop:
            self.root.after_cancel(self.active_animation_loop)
            self.active_animation_loop = None

        raw_text = self.text_input.get().lower()
        # Sanitize punctuation
        clean_words = "".join(ch for ch in raw_text if ch.isalnum() or ch.isspace()).split()
        
        self.animation_queue = []

        for word in clean_words:
            word_file = os.path.join(self.assets_dir, f"{word}.gif")
            
            # Check for Whole-Word Asset matches
            if os.path.exists(word_file):
                self.animation_queue.append((word_file, f"Word: {word.upper()}"))
            else:
                # FALLBACK: Execute sequential fingerspelling
                self.animation_queue.append((None, f"Spelling: {word.upper()}"))
                for char in word:
                    char_file = os.path.join(self.assets_dir, f"{char}.gif")
                    if os.path.exists(char_file):
                        self.animation_queue.append((char_file, f"Letter: {char.upper()}"))
                    else:
                        self.animation_queue.append((None, f"Missing: '{char.upper()}'"))

        self.animation_queue.append((None, "Finished"))
        self.play_next_sign()

    # Playback loop
    def play_next_sign(self):
        if not self.animation_queue:
            self.anim_status_var.set("Status: Idle")
            return

        img_path, message = self.animation_queue.pop(0)
        self.anim_status_var.set(f"Status: {message}")
        self.anim_canvas.delete("all")

        if img_path:
            try:
                img = Image.open(img_path).resize((300, 300), Image.Resampling.LANCZOS)
                self.current_frame = ImageTk.PhotoImage(img)
                self.anim_canvas.create_image(150, 150, image=self.current_frame)
                
                # Overlay current symbol identifier
                self.anim_canvas.create_text(150, 280, text=message, fill="#a6e3a1", font=("Helvetica", 12, "bold"))
                
                # Speed tuning: Letters play at 600ms, words play at 1200ms
                duration = 1200 if "Word:" in message else 600
            except Exception:
                self.anim_canvas.create_text(150, 150, text="Rendering Error", fill="#f38ba8", font=("Helvetica", 12))
                duration = 800
        else:
            self.anim_canvas.create_text(150, 150, text=message, fill="#f9e2af", font=("Helvetica", 16, "bold"))
            duration = 800

        self.active_animation_loop = self.root.after(duration, self.play_next_sign)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = DoubleWayTranslator(root)
    root.mainloop()