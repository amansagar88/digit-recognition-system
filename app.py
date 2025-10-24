from flask import Flask, request, jsonify, send_file # Import send_file
from flask_cors import CORS
import base64
import numpy as np
import cv2
from tensorflow import keras
import io
from PIL import Image
import os
import csv
import sys
import time # Import time for potential debugging delays

# --- Configuration ---
MODEL_FILENAME = 'digit_recognizer_model.keras'
MODEL_PATH = MODEL_FILENAME
CSV_FILENAME = 'user_pixel_data.csv' # New CSV filename for pixel data
# --- New CSV Header ---
# Create header: label, pixel0, pixel1, ..., pixel783
PIXEL_HEADERS = [f'pixel{i}' for i in range(28*28)]
CSV_HEADER = ['label'] + PIXEL_HEADERS

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ABSOLUTE_CSV_PATH = os.path.join(APP_DIR, CSV_FILENAME)


# --- Load the Keras Model ---
# ... (model loading code remains the same) ...
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}", file=sys.stderr)
    exit()
try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}", file=sys.stderr)
    exit()

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app) # Allow requests from your React app

# --- Health Check Route ---
@app.route('/', methods=['GET'])
def health_check():
    csv_exists = os.path.isfile(ABSOLUTE_CSV_PATH)
    return jsonify({"status": "healthy", "csv_found": csv_exists}), 200

# --- Prediction Endpoint (Unchanged) ---
@app.route('/predict', methods=['POST'])
def predict():
    # ... (Prediction logic remains exactly the same) ...
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data found in request'}), 400
        image_data = data['image']
        # 1. Decode Base64
        try:
            header, encoded = image_data.split(",", 1)
            binary_data = base64.b64decode(encoded)
        except Exception as e:
            return jsonify({'error': f'Invalid base64 image data: {str(e)}'}), 400
        # 2. Convert to image
        try:
            image = Image.open(io.BytesIO(binary_data)).convert('L')
            img_np = np.array(image)
        except Exception as e:
             return jsonify({'error': f'Could not decode image data: {str(e)}'}), 400
        if img_np is None:
             return jsonify({'error': 'Failed to decode image using OpenCV'}), 400
        # 3. Preprocess
        try:
            img_resized = cv2.resize(img_np, (28, 28), interpolation=cv2.INTER_AREA)
            img_processed = img_resized
            img_normalized = img_processed / 255.0
            img_reshaped = img_normalized.reshape(1, 28, 28, 1)
        except Exception as e:
            return jsonify({'error': f'Error during image preprocessing: {str(e)}'}), 500
        # 4. Predict
        try:
            prediction_raw = model.predict(img_reshaped)
            predicted_digit = np.argmax(prediction_raw)
            confidence_score = float(np.max(prediction_raw))
        except Exception as e:
             return jsonify({'error': f'Error during model prediction: {str(e)}'}), 500
        # 5. Return Response
        return jsonify({
            'prediction': int(predicted_digit),
            'confidence': confidence_score
        })
    except Exception as e:
        print(f"Unexpected error during prediction: {e}", file=sys.stderr)
        return jsonify({'error': f'An unexpected server error occurred: {str(e)}'}), 500


# --- UPDATED: Feedback/Training Data Saving Endpoint (Saves Pixels) ---
@app.route('/save_feedback', methods=['POST'])
def save_feedback():
    try:
        data = request.get_json()
        # 1. Validate Input Data
        # ... (Validation code remains the same) ...
        if 'image' not in data or 'label' not in data:
            return jsonify({'error': 'Missing image or label data in request'}), 400
        image_base64 = data['image']
        label = data['label']
        try:
            label_int = int(label)
            if not (0 <= label_int <= 9):
                 raise ValueError("Label must be between 0 and 9.")
        except (ValueError, TypeError):
             return jsonify({'error': 'Invalid label. Must be an integer between 0 and 9.'}), 400
        if not image_base64 or not image_base64.startswith('data:image/png;base64,'):
             return jsonify({'error': 'Invalid image data format.'}), 400

        # --- NEW: Process image to get pixels ---
        try:
            # Decode Base64
            header, encoded = image_base64.split(",", 1)
            binary_data = base64.b64decode(encoded)
            # Convert to image array
            image = Image.open(io.BytesIO(binary_data)).convert('L') # Grayscale
            img_np = np.array(image)
            # Resize to 28x28 (saving raw 0-255 values)
            img_resized_raw = cv2.resize(img_np, (28, 28), interpolation=cv2.INTER_AREA)
            # Flatten the 28x28 array into a 1D array of 784 pixels
            pixel_values = img_resized_raw.flatten()
            # Convert pixel values to integers (they should be already, but ensures type)
            pixel_values_int = [int(p) for p in pixel_values]
        except Exception as e:
            print(f"Error processing image for CSV: {e}", file=sys.stderr)
            return jsonify({'error': f'Could not process image data for saving: {str(e)}'}), 500
        # --- End image processing ---

        print(f"Attempting to write CSV to: {ABSOLUTE_CSV_PATH}")

        # 2. Prepare data row (label + 784 pixel values)
        new_row = [label_int] + pixel_values_int # Combine label and pixel list
        file_exists = os.path.isfile(ABSOLUTE_CSV_PATH)

        # 3. Append data to CSV
        with open(ABSOLUTE_CSV_PATH, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(CSV_HEADER)
                print(f"Created new CSV file: {ABSOLUTE_CSV_PATH} with header.")
            writer.writerow(new_row)

        print(f"Successfully appended pixel data for label: {label_int} to {ABSOLUTE_CSV_PATH}")
        # Optional: Add a small delay if file system seems slow
        # time.sleep(0.1) 
        return jsonify({'status': 'success', 'message': 'Feedback saved successfully.'}), 200

    except Exception as e:
        print(f"Error saving feedback: {e}", file=sys.stderr)
        print(f"Error occurred while trying to write to: {ABSOLUTE_CSV_PATH}", file=sys.stderr)
        return jsonify({'error': f'An internal server error occurred while saving feedback: {str(e)}'}), 500


# --- UPDATED: Endpoint to Download the CSV File (with pixel data) ---
@app.route('/download_csv', methods=['GET'])
def download_csv():
    # ... (Download logic remains largely the same, just serving the new file format) ...
    try:
        if not os.path.exists(ABSOLUTE_CSV_PATH):
            return jsonify({"error": "CSV file not found. Submit some feedback first."}), 404
        return send_file(
            ABSOLUTE_CSV_PATH,
            mimetype='text/csv',
            download_name=CSV_FILENAME, 
            as_attachment=True
        )
    except Exception as e:
        print(f"Error downloading CSV: {e}", file=sys.stderr)
        return jsonify({'error': f'An internal server error occurred while preparing the download: {str(e)}'}), 500

# Note: No app.run() needed for Hugging Face deployment

