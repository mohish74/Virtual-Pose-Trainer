import cv2
import os
import tkinter as tk
from tkinter import filedialog
import time


def select_video_file():
    """Opens a file dialog for the user to select a video file."""
    root = tk.Tk()
    root.withdraw()  # Hide the empty tkinter window
    filepath = filedialog.askopenfilename(
        title="Select a Video to Label",
        filetypes=(("Video Files", "*.mp4 *.avi *.mov"), ("All files", "*.*"))
    )
    return filepath


def get_exercise_name():
    """Prompts the user to enter the name of the exercise."""
    name = input("--> Please enter the name for this exercise (e.g., squats, pushups): ").lower().strip()
    return name


def annotate_video(video_path, exercise_name):
    """
    Processes a video, allowing the user to label each frame and save it
    into the correct exercise and label folder.
    """
    # --- 1. Create Directories Dynamically ---
    print(f"Preparing folders for exercise: '{exercise_name}'")
    base_path = os.path.join('data', exercise_name)
    correct_path = os.path.join(base_path, 'correct')
    bad_path = os.path.join(base_path, 'bad')

    # os.makedirs with exist_ok=True is a robust way to ensure directories exist
    os.makedirs(correct_path, exist_ok=True)
    os.makedirs(bad_path, exist_ok=True)

    # --- 2. Setup Video Capture ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    frame_num = 0

    # --- 3. Loop Through Frames and Label ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of the video

        cv2.imshow(f"Labeling for '{exercise_name}' - Press 'c' (correct), 'b' (bad), 's' (skip), 'q' (quit)", frame)
        key = cv2.waitKey(0) & 0xFF

        # Generate a unique name for the frame to prevent overwrites
        # Using a timestamp makes it unique across different videos and runs
        unique_frame_name = f"{video_filename}_{int(time.time() * 1000)}_{frame_num}.jpg"

        if key == ord('c'):  # Correct
            save_path = os.path.join(correct_path, unique_frame_name)
            cv2.imwrite(save_path, frame)
            print(f"Saved: {save_path}")
        elif key == ord('b'):  # Bad
            save_path = os.path.join(bad_path, unique_frame_name)
            cv2.imwrite(save_path, frame)
            print(f"Saved: {save_path}")
        elif key == ord('s'):  # Skip
            print("Skipped frame.")
            pass
        elif key == ord('q'):  # Quit
            print("Quitting annotation for this video.")
            break

        frame_num += 1

    cap.release()
    cv2.destroyAllWindows()


def main():
    """Main loop to orchestrate the annotation process for multiple videos."""
    while True:
        # --- Step 1: Select a video file ---
        video_path = select_video_file()
        if not video_path:
            print("No video selected. Exiting the tool.")
            break

        # --- Step 2: Get the exercise name from the user ---
        exercise_name = get_exercise_name()
        if not exercise_name:
            print("No exercise name entered. Skipping this video.")
            continue

        # --- Step 3: Run the annotation process for the video ---
        annotate_video(video_path, exercise_name)

        # --- Step 4: Ask to continue or exit ---
        another = input("\nDo you want to label another video? (y/n): ").lower()
        if another != 'y':
            print("Annotation session finished.")
            break


if __name__ == '__main__':
    main()