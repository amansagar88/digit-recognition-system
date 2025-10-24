# ✍️ Digit Recognizer - Web App with Live Feedback Loop 🧠

[![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Framework](https://img.shields.io/badge/Flask-2.x-green.svg)](https://flask.palletsprojects.com/)
[![ML Library](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Frontend](https://img.shields.io/badge/React-18.x-cyan.svg)](https://reactjs.org/)
[![Hosting](https://img.shields.io/badge/🤗%20Spaces-Hosted-yellow.svg)](https://huggingface.co/spaces)

---

## 🌟 Overview

Welcome to the Digit Recognizer project! This is a web application where users can draw a digit (0-9) on a canvas, and a basic Neural Network model predicts the digit in real-time. What makes this project special is its built-in feedback loop: users can flag incorrect predictions and provide the correct label, helping to collect valuable data for future model retraining! 🚀

**Demo Video**

![Demo gif](./assets/demo-video.gif)

---

## ✨ Features

* **✍️ Interactive Canvas:** Draw digits directly in your browser using mouse or touch.
* **🧠 Real-time Prediction:** Sends the drawing to a backend API for prediction using a TensorFlow/Keras Neural Network model.
* **📊 Confidence Score:** Displays the model's confidence in its prediction.
* **👍👎 Feedback Mechanism:** Users can mark predictions as correct or incorrect.
* **🏷️ Data Collection:** Incorrect predictions prompt the user for the correct label, which is saved (along with the image pixels) to a CSV file on the backend.
* **🔄 Retraining Ready:** Collected data (`user_pixel_data.csv`) is formatted perfectly for easy model retraining.
* **📱 Responsive Design:** Works on both desktop and mobile devices.
* **☀️🌙 Light/Dark Theme:** Adapts to your preferred theme.

---

## 🛠️ Tech Stack

* **Frontend:**
    * [React](https://reactjs.org/)
    * HTML5 Canvas API
    * CSS3
* **Backend:**
    * [Flask](https://flask.palletsprojects.com/) (Python Web Framework)
    * [TensorFlow](https://www.tensorflow.org/) / [Keras](https://keras.io/) (for loading and running the ML model)
    * [NumPy](https://numpy.org/) (for numerical operations)
    * [OpenCV (headless)](https://pypi.org/project/opencv-python-headless/) (for image preprocessing)
    * [Pillow](https://python-pillow.org/) (for image handling)
    * [Gunicorn](https://gunicorn.org/) (WSGI Server)
* **Hosting:**
    * [Hugging Face Spaces 🤗](https://huggingface.co/spaces) (using Docker)

---

## 🚀 Live Demo

**➡️ Check out the live demo at: [https://www.amansagar.dev/digit-recognition-system](https://www.amansagar.dev/digit-recognition-system)**

*(Note: The first request might take a few seconds if the Space has gone to sleep)*

---

## 🌐 Using the Hosted API

You can interact with the deployed Flask API endpoints directly.
### 1. Predict Digit (`/predict`)

* **Method:** `POST`
* **URL:** `https://aman881-digit-recognition-system.hf.space/predict`
* **Body (JSON):**
    ```json
    {
      "image": "data:image/png;base64,iVBORw0KGgoAAAANS..." // Base64 encoded image string from a 280x280 canvas
    }
    ```
* **Success Response (200 OK):**
    ```json
    {
      "prediction": 7,
      "confidence": 0.9987
    }
    ```
* **Error Response (400/500):**
    ```json
    {
      "error": "Descriptive error message"
    }
    ```

### 2. Save Feedback (`/save_feedback`)

* **Method:** `POST`
* **URL:** `https://aman881-digit-recognition-system.hf.space/save_feedback`
* **Body (JSON):**
    ```json
    {
      "image": "data:image/png;base64,iVBORw0KGgoAAAANS...", // Base64 encoded image string from the canvas when prediction was made
      "label": 5 // Integer (0-9), the correct label provided by the user
    }
    ```
* **Success Response (200 OK):**
    ```json
    {
      "status": "success",
      "message": "Feedback saved successfully."
    }
    ```
* **Error Response (400/500):**
    ```json
    {
      "error": "Descriptive error message"
    }
    ```

### 3. Download CSV Data (`/download_csv`)

* **Method:** `GET`
* **URL:** `https://aman881-digit-recognition-system.hf.space/download_csv`
* **Response:** Triggers a file download of `user_pixel_data.csv`. The file contains the label and 784 pixel values (0-255) for each submitted feedback entry.
* **Error Response (404 Not Found):** If no feedback has been submitted yet and the file doesn't exist.
    ```json
    {
      "error": "CSV file not found. Submit some feedback first."
    }
    ```

---

## 🏗️ Architecture

1.  **React Frontend:** Captures user drawing on the canvas, converts it to a base64 image string.
2.  **API Call (`/predict`):** Sends the image data to the Flask backend.
3.  **Flask Backend (`/predict`):**
    * Decodes the image.
    * Preprocesses it (resize, normalize).
    * Feeds it into the loaded Keras model.
    * Returns the predicted digit and confidence score as JSON.
4.  **React Frontend:** Displays the prediction and confidence. Shows feedback buttons.
5.  **API Call (`/save_feedback`):** If the user submits a correction, the frontend sends the *original image data* and the *correct label* to this endpoint.
6.  **Flask Backend (`/save_feedback`):**
    * Validates the input.
    * Processes the image to get 28x28 pixel values.
    * Appends the `label` and `pixel_values` (784 columns) to `user_pixel_data.csv`.
7.  **Data Download (`/download_csv`):** A separate endpoint allows downloading the collected CSV data.

---

## ⚙️ Setup and Installation (Locally)

### Backend (Flask API)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/amansagar88/digit-recognition-system
    cd digit-recognition-system
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Flask app:**
    ```bash
    python app.py
    ```
    The API will be running, typically at `http://127.0.0.1:5000`.

## 📊 Feedback & Retraining

* When a user flags a prediction as incorrect and provides the correct label, the backend saves the **label** and the **784 raw pixel values** (0-255) of the 28x28 grayscale image to `user_pixel_data.csv`.
* You can download this CSV from the `/download_csv` endpoint of the deployed backend.
* This data can then be used to fine-tune or retrain the Keras model (`digit_recognizer_model.keras`) using a separate Python script, potentially leading to improved accuracy over time.

---

## 🌱 Future Improvements

* Implement the "Training Mode" feature, where user can train the model.
* Automate the retraining process (e.g., a script that downloads data, retrains, and uploads the new model).
* Add more sophisticated image preprocessing on the backend.
* Visualize the collected data.
* Experiment with different model architectures.

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions or find bugs, please feel free to open an issue or submit a pull request.

---

## 🙏 Acknowledgements

* Thanks to the creators of React, Flask, TensorFlow, OpenCV, and Hugging Face!

---

## 🧑‍💻 Author

**<a href="www.amansagar.dev" style="text-decoration:none;">Aman Sagar</a>**  
Data Science Enthusiast | Passionate about ML Algorithms

---
