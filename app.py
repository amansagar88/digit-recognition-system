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

# --- Get the absolute path within the container ---
# This ensures we always read/write from the correct location inside /app
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ABSOLUTE_CSV_PATH = os.path.join(APP_DIR, CSV_FILENAME)


# --- Load the Keras Model ---
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}", file=sys.stderr) # Log errors to stderr
    # In a real deployment, you might want to raise an exception or handle this gracefully
    # For simplicity, we exit if the model isn't found during startup.
    exit()
try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}", file=sys.stderr) # Log errors to stderr
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
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data found in request'}), 400
        image_data = data['image']

        # 1. Decode Base64
        try:
            # Check if header exists before splitting
            if "," not in image_data:
                 raise ValueError("Invalid base64 header")
            header, encoded = image_data.split(",", 1)
            binary_data = base64.b64decode(encoded)
        except Exception as e:
            print(f"Base64 Decoding Error: {e}", file=sys.stderr)
            return jsonify({'error': f'Invalid base64 image data: {str(e)}'}), 400

        # 2. Convert to image
        try:
            image = Image.open(io.BytesIO(binary_data)).convert('L')
            img_np = np.array(image)
        except Exception as e:
             print(f"Image Conversion Error: {e}", file=sys.stderr)
             return jsonify({'error': f'Could not decode image data: {str(e)}'}), 400

        if img_np is None:
             print("OpenCV decoding resulted in None", file=sys.stderr)
             return jsonify({'error': 'Failed to decode image using OpenCV'}), 400

        # 3. Preprocess
        try:
            # Ensure input is grayscale
            if len(img_np.shape) > 2 and img_np.shape[2] != 1:
                 img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY) # Convert if needed

            img_resized = cv2.resize(img_np, (28, 28), interpolation=cv2.INTER_AREA)
            # Assuming canvas sends black background/white digit (MNIST format)
            img_processed = img_resized
            img_normalized = img_processed / 255.0
            img_reshaped = img_normalized.reshape(1, 28, 28, 1)
        except Exception as e:
            print(f"Image Preprocessing Error: {e}", file=sys.stderr)
            return jsonify({'error': f'Error during image preprocessing: {str(e)}'}), 500

        # 4. Predict
        try:
            prediction_raw = model.predict(img_reshaped)
            predicted_digit = np.argmax(prediction_raw)
            confidence_score = float(np.max(prediction_raw))
        except Exception as e:
             print(f"Model Prediction Error: {e}", file=sys.stderr)
             return jsonify({'error': f'Error during model prediction: {str(e)}'}), 500

        # 5. Return Response
        return jsonify({
            'prediction': int(predicted_digit),
            'confidence': confidence_score
        })
    except Exception as e:
        print(f"Unexpected error during prediction: {e}", file=sys.stderr)
        return jsonify({'error': f'An unexpected server error occurred: {str(e)}'}), 500


# --- UPDATED: Feedback/Training Data Saving Endpoint (Accepts "NaN") ---
@app.route('/save_feedback', methods=['POST'])
def save_feedback():
    try:
        data = request.get_json()

        # 1. Validate Input Data
        if 'image' not in data or 'label' not in data:
            return jsonify({'error': 'Missing image or label data in request'}), 400

        image_base64 = data['image']
        label_input = data['label'] # Keep as received (could be string 'NaN' or digit string)
        validated_label = None # Will store the final label to save

        # --- UPDATED Validation ---
        if label_input == "NaN":
            validated_label = "NaN" # Accept the string "NaN"
        else:
            try:
                # Try converting other inputs to int 0-9
                label_int = int(label_input)
                if 0 <= label_int <= 9:
                    validated_label = label_int # Store the valid integer
                else:
                    raise ValueError("Label must be between 0 and 9.")
            except (ValueError, TypeError):
                 # If it's not "NaN" and not an int 0-9, it's invalid
                 print(f"Invalid label received: {label_input}", file=sys.stderr)
                 return jsonify({'error': 'Invalid label. Must be an integer between 0 and 9, or the string "NaN".'}), 400
        # --- End Validation Update ---

        if not image_base64 or not isinstance(image_base64, str) or not image_base64.startswith('data:image/png;base64,'):
             print(f"Invalid image data format received.", file=sys.stderr)
             return jsonify({'error': 'Invalid image data format.'}), 400

        # --- Process image to get pixels ---
        try:
            # Decode Base64
            if "," not in image_base64: raise ValueError("Invalid base64 header")
            header, encoded = image_base64.split(",", 1)
            binary_data = base64.b64decode(encoded)
            # Convert to image array
            image = Image.open(io.BytesIO(binary_data)).convert('L') # Grayscale
            img_np = np.array(image)
            # Resize to 28x28 (saving raw 0-255 values)
            # Ensure it's grayscale before resizing
            if len(img_np.shape) > 2 and img_np.shape[2] != 1:
                 img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
            img_resized_raw = cv2.resize(img_np, (28, 28), interpolation=cv2.INTER_AREA)
            # Flatten the 28x28 array into a 1D array of 784 pixels
            pixel_values = img_resized_raw.flatten()
            # Convert pixel values to integers
            pixel_values_int = [int(p) for p in pixel_values]
        except Exception as e:
            print(f"Error processing image for CSV: {e}", file=sys.stderr)
            return jsonify({'error': f'Could not process image data for saving: {str(e)}'}), 500

        print(f"Attempting to write CSV to: {ABSOLUTE_CSV_PATH}")

        # 2. Prepare data row (validated_label + pixels)
        new_row = [validated_label] + pixel_values_int
        file_exists = os.path.isfile(ABSOLUTE_CSV_PATH)

        # 3. Append data to CSV
        try:
            with open(ABSOLUTE_CSV_PATH, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(CSV_HEADER)
                    print(f"Created new CSV file: {ABSOLUTE_CSV_PATH} with header.")
                writer.writerow(new_row)
        except Exception as e:
             # Catch potential file writing errors
             print(f"Error writing to CSV file '{ABSOLUTE_CSV_PATH}': {e}", file=sys.stderr)
             return jsonify({'error': f'Could not write to CSV file: {str(e)}'}), 500


        print(f"Successfully appended pixel data for label: {validated_label} to {ABSOLUTE_CSV_PATH}")
        # Optional: Add a small delay if file system seems slow - might not be needed
        # time.sleep(0.1)
        return jsonify({'status': 'success', 'message': 'Feedback saved successfully.'}), 200

    except Exception as e:
        print(f"Error saving feedback: {e}", file=sys.stderr)
        print(f"Error occurred while trying to write to: {ABSOLUTE_CSV_PATH}", file=sys.stderr)
        return jsonify({'error': f'An internal server error occurred while saving feedback: {str(e)}'}), 500


# --- Download CSV Endpoint (Unchanged) ---
@app.route('/download_csv', methods=['GET'])
def download_csv():
    try:
        print(f"Download request received for: {ABSOLUTE_CSV_PATH}") # Log download attempt
        if not os.path.exists(ABSOLUTE_CSV_PATH):
            print(f"CSV file not found at {ABSOLUTE_CSV_PATH} for download.", file=sys.stderr)
            return jsonify({"error": "CSV file not found. Submit some feedback first."}), 404

        # Return the file as an attachment
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

