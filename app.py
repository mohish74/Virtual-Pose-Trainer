import tkinter
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyttsx3
import threading
import time

# --- App Configuration ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
REPS_PER_SET = 10


class FitnessApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Fitness Trainer")
        self.geometry("1200x800")

        # --- State & Logic Variables ---
        self.cap = None
        self.video_paused = True
        self.feedback = "Choose an exercise & source"
        self.rep_counter = 0
        self.set_counter = 1
        self.current_exercise = "Squats"
        self.exercise_state = "UP"
        self.ai_status = "N/A"
        self.last_feedback = ""
        self.descent_start_time = None
        self.last_rep_duration = 0
        self.tempo_feedback = "N/A"

        # --- Load AI Model, MediaPipe, and Audio ---
        try:
            self.model = tf.keras.models.load_model('best_squat_model_2.h5')
            print("Squat AI model loaded successfully.")
        except Exception as e:
            print(f"Note: Squat AI model not found. Squat analysis will use rules only.")
            self.model = None
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        try:
            self.tts_engine = pyttsx3.init()
        except Exception as e:
            print(f"Could not initialize TTS engine: {e}")
            self.tts_engine = None

        self.create_widgets()
        self.update_frame()

    def create_widgets(self):
        # Main layout
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)
        self.video_frame = ctk.CTkFrame(self.main_frame, fg_color="black")
        self.video_frame.pack(side="left", padx=10, pady=10, expand=True, fill="both")
        self.sidebar_frame = ctk.CTkFrame(self.main_frame, width=300)
        self.sidebar_frame.pack(side="right", padx=10, pady=10, fill="y")
        self.sidebar_frame.pack_propagate(False)

        # Video display
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack(expand=True, fill="both")

        # Sidebar UI Elements
        self.exercise_menu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Squats"], command=self.change_exercise,
                                               font=ctk.CTkFont(size=14))
        self.exercise_menu.pack(pady=10, padx=20, fill="x")

        self.reps_label = ctk.CTkLabel(self.sidebar_frame, text="REPS", font=ctk.CTkFont(size=20, weight="bold"))
        self.reps_label.pack(pady=(10, 5), padx=20)
        self.reps_value = ctk.CTkLabel(self.sidebar_frame, text=str(self.rep_counter),
                                       font=ctk.CTkFont(size=50, weight="bold"))
        self.reps_value.pack(pady=5, padx=20)

        self.sets_label = ctk.CTkLabel(self.sidebar_frame, text="SETS", font=ctk.CTkFont(size=20, weight="bold"))
        self.sets_label.pack(pady=(10, 5), padx=20)
        self.sets_value = ctk.CTkLabel(self.sidebar_frame, text=str(self.set_counter),
                                       font=ctk.CTkFont(size=50, weight="bold"))
        self.sets_value.pack(pady=5, padx=20)

        self.tempo_label = ctk.CTkLabel(self.sidebar_frame, text="TEMPO (SECONDS)",
                                        font=ctk.CTkFont(size=20, weight="bold"))
        self.tempo_label.pack(pady=(10, 5), padx=20)
        self.tempo_value = ctk.CTkLabel(self.sidebar_frame, text="--", font=ctk.CTkFont(size=24, weight="bold"),
                                        text_color="cyan")
        self.tempo_value.pack(pady=5, padx=20)

        ## FIX: Add the missing AI Status widgets here ##
        self.ai_status_label = ctk.CTkLabel(self.sidebar_frame, text="AI STATUS",
                                            font=ctk.CTkFont(size=20, weight="bold"))
        self.ai_status_label.pack(pady=(10, 5), padx=20)
        self.ai_status_text = ctk.CTkLabel(self.sidebar_frame, text=self.ai_status,
                                           font=ctk.CTkFont(size=24, weight="bold"), text_color="cyan")
        self.ai_status_text.pack(pady=5, padx=10)

        # Feedback Display
        self.feedback_label = ctk.CTkLabel(self.sidebar_frame, text="COACH'S ADVICE",
                                           font=ctk.CTkFont(size=20, weight="bold"))
        self.feedback_label.pack(pady=(20, 5), padx=20)
        self.feedback_text = ctk.CTkLabel(self.sidebar_frame, text=self.feedback, font=ctk.CTkFont(size=18),
                                          wraplength=280)
        self.feedback_text.pack(pady=5, padx=10, expand=True)

        # Control buttons
        self.start_stop_button = ctk.CTkButton(self.sidebar_frame, text="Start / Pause", command=self.toggle_video)
        self.start_stop_button.pack(side="bottom", pady=10, padx=20, fill="x")
        self.select_file_button = ctk.CTkButton(self.sidebar_frame, text="Select Video File",
                                                command=self.select_video_file)
        self.select_file_button.pack(side="bottom", pady=10, padx=20, fill="x")
        self.webcam_button = ctk.CTkButton(self.sidebar_frame, text="Use Webcam", command=self.use_webcam)
        self.webcam_button.pack(side="bottom", pady=10, padx=20, fill="x")

    def change_exercise(self, new_exercise):
        self.current_exercise = new_exercise
        self.rep_counter = 0
        self.set_counter = 1
        self.exercise_state = "DOWN" if new_exercise == "Bicep Curls" else "UP"
        self.feedback = f"Ready to start {new_exercise}"
        self.ai_status = "N/A"

    def use_webcam(self):
        if self.cap: self.cap.release()
        self.cap = cv2.VideoCapture(0)
        self.video_paused = False

    def select_video_file(self):
        if self.cap: self.cap.release()
        filepath = tkinter.filedialog.askopenfilename()
        if filepath:
            self.cap = cv2.VideoCapture(filepath)
            self.video_paused = False

    def toggle_video(self):
        self.video_paused = not self.video_paused

    def speak_feedback(self, text):
        if self.tts_engine and text and text != self.last_feedback:
            self.last_feedback = text
            threading.Thread(target=self._run_tts, args=(text,), daemon=True).start()

    def _run_tts(self, text):
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180.0 else angle

    def process_squats(self, landmarks):
        if self.model:
            pose_row = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten()
            X = np.expand_dims(pose_row, axis=0)
            prediction = self.model.predict(X, verbose=0)[0][0]
            self.ai_status = 'CORRECT' if prediction > 0.5 else 'BAD'

        knee_angle = self.calculate_angle(
            [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP].x,
             landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP].y],
            [landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE].x,
             landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE].y],
            [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].x,
             landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].y]
        )

        if self.exercise_state == "UP":
            if knee_angle < 160:
                self.descent_start_time = time.time()
                self.exercise_state = "DESCENDING"

        elif self.exercise_state == "DESCENDING":
            self.feedback = "Lower your hips"
            if knee_angle < 95:
                self.last_rep_duration = time.time() - self.descent_start_time
                self.exercise_state = "DOWN"

        elif self.exercise_state == "DOWN":
            self.feedback = "Great depth! Drive up!"
            if knee_angle > 100:
                self.exercise_state = "ASCENDING"

        elif self.exercise_state == "ASCENDING":
            self.feedback = "Finish the rep strong!"
            if knee_angle > 160:
                self.rep_counter += 1
                self.exercise_state = "UP"
                self.tempo_feedback = "Too fast!" if self.last_rep_duration < 1.0 else "Good tempo!"
                self.feedback = f"REP COMPLETE! {self.tempo_feedback}"

                if self.rep_counter >= REPS_PER_SET:
                    self.set_counter += 1
                    self.rep_counter = 0
                    self.feedback = "SET COMPLETE! Take a rest."
                    self.speak_feedback(self.feedback)
                    self.video_paused = True

    def update_frame(self):
        if self.cap and not self.video_paused:
            ret, frame = self.cap.read()
            if ret:
                if hasattr(self.cap, 'get') and self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) < 1000:
                    frame = cv2.flip(frame, 1)

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(image_rgb)
                mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                try:
                    if results.pose_landmarks:
                        if self.current_exercise == "Squats":
                            self.process_squats(results.pose_landmarks.landmark)
                except Exception as e:
                    pass

                self.speak_feedback(self.feedback)

                # Corrected video resizing logic
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img_w, img_h = img.size
                label_w, label_h = self.video_label.winfo_width(), self.video_label.winfo_height()

                if img_w > 0 and img_h > 0 and label_w > 0 and label_h > 0:
                    img_aspect = img_w / img_h
                    label_aspect = label_w / label_h
                    if label_aspect > img_aspect:
                        new_h = label_h
                        new_w = int(new_h * img_aspect)
                    else:
                        new_w = label_w
                        new_h = int(new_w / img_aspect)
                    img = img.resize((new_w, new_h), Image.LANCZOS)

                ctk_img = ImageTk.PhotoImage(image=img)
                self.video_label.configure(image=ctk_img, text="")
                self.video_label.image = ctk_img

        # Update UI Labels
        self.reps_value.configure(text=str(self.rep_counter))
        self.sets_value.configure(text=str(self.set_counter))
        self.feedback_text.configure(text=self.feedback)
        self.tempo_value.configure(text=f"{self.last_rep_duration:.2f}s" if self.last_rep_duration > 0 else "--")
        self.ai_status_text.configure(text=self.ai_status)

        self.after(20, self.update_frame)


if __name__ == "__main__":
    app = FitnessApp()
    app.mainloop()