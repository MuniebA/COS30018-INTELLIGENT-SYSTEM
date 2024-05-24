import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
from flask import Blueprint, Response

live_video_app = Blueprint('live_video_app', __name__)

# Define paths to your model and label files
MODEL_PATH = "custom_model_lite/detect.tflite"
LABEL_PATH = "custom_model_lite/labelmap.txt"

# Load the TFLite model
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Load labels
with open(LABEL_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]


def preprocess_image(image, input_details, height, width):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    if input_details[0]['dtype'] == np.float32:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    return input_data


def detect_objects(image, interpreter, input_details, output_details, height, width, labels):
    input_data = preprocess_image(image, input_details, height, width)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[3]['index'])[0]
    scores = interpreter.get_tensor(output_details[0]['index'])[0]

    for i in range(len(scores)):
        if scores[i] > 0.1:
            ymin, xmin, ymax, xmax = boxes[i]
            ymin = int(max(1, ymin * image.shape[0]))
            xmin = int(max(1, xmin * image.shape[1]))
            ymax = int(min(image.shape[0], ymax * image.shape[0]))
            xmax = int(min(image.shape[1], xmax * image.shape[1]))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = f'{labels[int(classes[i])]}: {scores[i]*100:.2f}%'
            cv2.putText(image, label, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return image


def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_objects(
            frame, interpreter, input_details, output_details, height, width, labels)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@live_video_app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
