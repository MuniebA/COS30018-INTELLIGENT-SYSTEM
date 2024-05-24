from flask import Flask, render_template, request, redirect, url_for, jsonify
import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
from video_processor import process_video
from werkzeug.utils import secure_filename
import os

# Define paths to your model and label files
MODEL_PATH = "custom_model_lite/detect.tflite"
LABEL_PATH = "custom_model_lite/labelmap.txt"
ALLOWED_EXTENSIONS = {'mp4', 'avi'}

# Function to load the TFLite model and labels


def load_model():
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    with open(LABEL_PATH, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    return interpreter, input_details, output_details, height, width, labels

# Function to preprocess the image for the model


def preprocess_image(image, input_details, height, width):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    if input_details[0]['dtype'] == np.float32:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    return input_data

# Function to perform object detection and draw bounding boxes


def detect_objects(image, interpreter, input_details, output_details, labels):
    height, width, _ = image.shape
    input_data = preprocess_image(image, input_details, height, width)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[3]['index'])[0]
    scores = interpreter.get_tensor(output_details[0]['index'])[0]

    for i in range(len(scores)):
        if scores[i] > 0.1:
            ymin, xmin, ymax, xmax = boxes[i]
            ymin = int(max(1, ymin * height))
            xmin = int(max(1, xmin * width))
            ymax = int(min(height, ymax * height))
            xmax = int(min(width, xmax * width))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = '%s: %.2f%%' % (labels[int(classes[i])], scores[i] * 100)
            cv2.putText(image, label, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return image


# Initialize the Flask app
app = Flask(__name__)

# Load the TFLite model and labels
interpreter, input_details, output_details, height, width, labels = load_model()


@app.route('/', methods=['GET', 'POST'])
def upload_and_detect():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        image = cv2.imdecode(np.frombuffer(
            file.read(), np.uint8), cv2.IMREAD_COLOR)
        processed_image = detect_objects(
            image, interpreter, input_details, output_details, labels)

        # Save processed image
        save_path = os.path.join('static', 'detected.jpg')
        cv2.imwrite(save_path, processed_image)

        # Send back the path to the processed image
        return jsonify({'image_url': url_for('static', filename='detected.jpg')})

    return render_template('upload.html')


app.config['UPLOAD_FOLDER'] = 'static/videos'

@app.route('/video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        video_file = request.files.get('file')
        if video_file:
            filename = secure_filename(video_file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video_file.save(video_path)
            processed_video_path = process_video(video_path)
            # Provide a way to access the processed video, e.g., redirect or return a link
            return jsonify({'video_url': url_for('static', filename=os.path.basename(processed_video_path))})
        return jsonify({"error": "No file part"}), 400
    # For GET requests, render a template or redirect
    return render_template('video.html')


if __name__ == '__main__':
    app.run(debug=True)
