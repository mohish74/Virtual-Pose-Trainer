# Virtual-Pose-Trainer

![AI Fitness Trainer Demo](demo.gif)

## 📖 Description

The AI Fitness Trainer is a real-time application that uses computer vision and a custom-trained neural network to analyze and provide corrective feedback on exercises. This project acts as a virtual personal trainer, helping users improve their form, count reps and sets, and track their tempo, all from their webcam or a video file.

---

## 📂 Project Structure

This project is composed of several key modules that work together to create a full machine learning pipeline:

* `app.py`: The main application file. This runs the CustomTkinter GUI, handles the video feed, and integrates all the analysis features.
* `annotation_tool.py`: A powerful tool to create labeled image datasets for new exercises.
* `ai_training.py`: A script to train a new image classification model using the datasets created by the annotation tool.
* `best_squat_model.h5`: The pre-trained model for squat analysis (landmark-based).
* `requirements.txt`: A list of all Python dependencies required to run the project.

---

## ✨ Features

* **Real-Time Pose Estimation:** Utilizes MediaPipe to detect 33 body landmarks.
* **Custom AI for Form Classification:** Includes a pre-trained model for squats and a training script to build new models from image data.
* **Intelligent Rule-Based Feedback:** Provides specific, actionable advice on form, posture, and arm position.
* **Live Audio Coach:** Uses text-to-speech to speak feedback aloud for an interactive experience.
* **Tempo & Speed Analysis:** Measures the descent speed of each rep to encourage controlled movements.
* **Automatic Rep & Set Counting:** A robust state machine automatically counts repetitions and tracks workout sets.
* **Modern GUI:** Built with CustomTkinter for a clean, professional, and user-friendly interface.

---

## 🛠️ Tech Stack

* **Python**
* **TensorFlow / Keras:** For building and training neural networks.
* **OpenCV:** For handling the video feed.
* **MediaPipe:** For real-time pose estimation.
* **CustomTkinter:** For the graphical user interface.
* **Pandas & Scikit-learn:** For data analysis.
* **pyttsx3:** For text-to-speech audio feedback.

---

## 🚀 Setup and Usage

There are two ways to use this project: running the pre-built application or training your own model.

### A. Running the Application with the Pre-trained Model

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv .venv
    .\.venv\Scripts\activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    python app.py
    ```
    Use the UI to select the "Squats" exercise and choose your video source.

### B. Training a New Model for a New Exercise

1.  **Step 1: Collect Data**
    Run the annotation tool to create a labeled dataset. It will ask for an exercise name and create the necessary folders inside the `data/` directory.
    ```bash
    python annotation_tool.py
    ```

2.  **Step 2: Train the Model**
    Run the training script. It will automatically find the image folders inside `data/` and train a new model using transfer learning.
    ```bash
    python ai_training.py
    ```
    This will save a new `image_classifier_model.h5` file.

3.  **Step 3: Integrate (Manual Step)**
    To use your new model, you would need to:
    * Load your `image_classifier_model.h5` in `app.py`.
    * Create a new `process_<exercise_name>` function with the logic to use your new model and rules.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
