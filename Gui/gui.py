from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
from live_video import live_video_app
from video_processor import process_video
from werkzeug.utils import secure_filename
import os

# Define default paths (these will be updated based on user selection)
MODEL_PATH = ""
LABEL_PATH = ""
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'jpg', 'jpeg', 'png'}

# Function to load the TFLite model and labels


def load_model(model_path, label_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    with open(label_path, 'r') as f:
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


@app.route('/', methods=['GET', 'POST'])
def upload_and_detect():
    global MODEL_PATH, LABEL_PATH
    if request.method == 'POST':
        model_type = request.form.get('modelType')
        if model_type == 'multi_class':
            MODEL_PATH = "ResNet50_640/custom_model_lite/detect.tflite"
            LABEL_PATH = "ResNet50_640/custom_model_lite/labelmap.txt"
        elif model_type == 'empty_detection':
            MODEL_PATH = "MobileNet_640/custom_model_lite/detect.tflite"
            LABEL_PATH = "MobileNet_640/custom_model_lite/labelmap.txt"
        elif model_type == 'unorganised_detection':
            MODEL_PATH = "MobileNet_320/custom_model_lite/detect.tflite"
            LABEL_PATH = "MobileNet_320/custom_model_lite/labelmap.txt"
        else:
            return jsonify({"error": "Invalid model type selected"}), 400

        interpreter, input_details, output_details, height, width, labels = load_model(
            MODEL_PATH, LABEL_PATH)
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
    global MODEL_PATH, LABEL_PATH
    if request.method == 'POST':
        model_type = request.form.get('modelType')
        if model_type == 'multi_class':
            MODEL_PATH = "ResNet50_640/custom_model_lite/detect.tflite"
            LABEL_PATH = "ResNet50_640/custom_model_lite/labelmap.txt"
        elif model_type == 'empty_detection':
            MODEL_PATH = "MobileNet_640/custom_model_lite/detect.tflite"
            LABEL_PATH = "MobileNet_640/custom_model_lite/labelmap.txt"
        elif model_type == 'unorganised_detection':
            MODEL_PATH = "MobileNet_320/custom_model_lite/detect.tflite"
            LABEL_PATH = "MobileNet_320/custom_model_lite/labelmap.txt"
        else:
            return jsonify({"error": "Invalid model type selected"}), 400

        video_file = request.files.get('file')
        if video_file:
            filename = secure_filename(video_file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video_file.save(video_path)
            print(f"Video file saved to {video_path}")  # Debug print
            processed_video_path = process_video(
                video_path, MODEL_PATH, LABEL_PATH)
            # Debug print
            print(f"Processed video path: {processed_video_path}")
            processed_video_url = url_for(
                'static', filename=f'videos/{os.path.basename(processed_video_path)}')
            return jsonify({'video_url': processed_video_url})
        return jsonify({"error": "No file part"}), 400
    return render_template('video.html')


# Register the live_video blueprint
app.register_blueprint(live_video_app, url_prefix='/live_video')


@app.route('/live_video')
def live_video():
    return render_template('live_video.html')


if __name__ == '__main__':
    app.run(debug=True)
