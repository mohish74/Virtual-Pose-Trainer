import tkinter
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Note: Audio is left out for now as you said it wasn't working,
# but this logic can be added back to the audio script easily.

# Set the appearance mode and default color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class FitnessApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Fitness Trainer")
        self.geometry("1200x800")

        # --- LOGIC & STATE VARIABLES ---
        self.cap = None
        self.video_paused = True
        self.feedback = "Select a video to start"
        self.rep_counter = 0
        self.set_counter = 1
        self.squat_state = "UP"  # ## NEW: Our state variable

        # --- LOAD AI MODEL & MEDIAPIPE ---
        try:
            self.model = tf.keras.models.load_model('best_squat_model_2.h5')
        except Exception as e:
            self.feedback = "Error: Model not found"
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.create_widgets()
        self.update_frame()

    def create_widgets(self):
        # This function remains the same as before
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)
        self.video_frame = ctk.CTkFrame(self.main_frame, fg_color="black")
        self.video_frame.pack(side="left", padx=10, pady=10, expand=True, fill="both")
        self.sidebar_frame = ctk.CTkFrame(self.main_frame, width=300)
        self.sidebar_frame.pack(side="right", padx=10, pady=10, fill="y")
        self.sidebar_frame.pack_propagate(False)
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack(expand=True, fill="both")
        self.reps_label = ctk.CTkLabel(self.sidebar_frame, text="REPS", font=ctk.CTkFont(size=20, weight="bold"))
        self.reps_label.pack(pady=(20, 5), padx=20)
        self.reps_value = ctk.CTkLabel(self.sidebar_frame, text=str(self.rep_counter),
                                       font=ctk.CTkFont(size=60, weight="bold"))
        self.reps_value.pack(pady=5, padx=20)
        self.sets_label = ctk.CTkLabel(self.sidebar_frame, text="SETS", font=ctk.CTkFont(size=20, weight="bold"))
        self.sets_label.pack(pady=(20, 5), padx=20)
        self.sets_value = ctk.CTkLabel(self.sidebar_frame, text=str(self.set_counter),
                                       font=ctk.CTkFont(size=60, weight="bold"))
        self.sets_value.pack(pady=5, padx=20)
        self.feedback_label = ctk.CTkLabel(self.sidebar_frame, text="FEEDBACK",
                                           font=ctk.CTkFont(size=20, weight="bold"))
        self.feedback_label.pack(pady=(20, 5), padx=20)
        self.feedback_text = ctk.CTkLabel(self.sidebar_frame, text=self.feedback, font=ctk.CTkFont(size=18),
                                          wraplength=280)
        self.feedback_text.pack(pady=5, padx=10, expand=True)
        self.select_file_button = ctk.CTkButton(self.sidebar_frame, text="Select Video", command=self.select_video)
        self.select_file_button.pack(side="bottom", pady=10, padx=20, fill="x")
        self.start_stop_button = ctk.CTkButton(self.sidebar_frame, text="Start / Pause", command=self.toggle_video)
        self.start_stop_button.pack(side="bottom", pady=10, padx=20, fill="x")

    def select_video(self):
        # This function remains the same
        filepath = tkinter.filedialog.askopenfilename()
        if filepath:
            if self.cap: self.cap.release()
            self.cap = cv2.VideoCapture(filepath)
            self.video_paused = False

    def toggle_video(self):
        # This function remains the same
        self.video_paused = not self.video_paused

    def calculate_angle(self, a, b, c):
        # This function remains the same
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180.0 else angle

    def update_frame(self):
        if self.cap and not self.video_paused:
            ret, frame = self.cap.read()
            if ret:
                # Process the frame with MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(image_rgb)
                mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                # --- INTELLIGENT FEEDBACK & REP COUNTING LOGIC ---
                try:
                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark

                        # Calculate knee angle
                        left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                    landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                     landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                        left_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                      landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                        knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)

                        # ## NEW: Rep Counting State Machine Logic
                        # Check if user is at the bottom of the squat
                        if knee_angle < 95:
                            # If they were previously standing, now they're down
                            if self.squat_state == "UP":
                                self.feedback = "GREAT DEPTH!"
                            self.squat_state = "DOWN"

                        # Check if user is standing up from a squat
                        if knee_angle > 160 and self.squat_state == "DOWN":
                            # This is the moment a rep is completed
                            self.rep_counter += 1
                            self.squat_state = "UP"
                            self.feedback = "REP COMPLETE!"

                except Exception as e:
                    self.feedback = "No person detected"

                # Convert frame for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                w, h = img.size
                label_w, label_h = self.video_label.winfo_width(), self.video_label.winfo_height()
                if w > 0 and h > 0 and label_w > 0 and label_h > 0:
                    aspect_ratio = w / h
                    if w > label_w: w, h = label_w, int(label_w / aspect_ratio)
                    if h > label_h: h, w = label_h, int(h * aspect_ratio)
                    img = img.resize((w, h), Image.LANCZOS)
                ctk_img = ImageTk.PhotoImage(image=img)
                self.video_label.configure(image=ctk_img, text="")
                self.video_label.image = ctk_img

        # Update UI Labels
        self.reps_value.configure(text=str(self.rep_counter))
        self.sets_value.configure(text=str(self.set_counter))
        self.feedback_text.configure(text=self.feedback)

        # Schedule next update
        self.after(20, self.update_frame)


if __name__ == "__main__":
    app = FitnessApp()
    app.mainloop()