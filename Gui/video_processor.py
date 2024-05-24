import subprocess
import os
import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter


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


def preprocess_image(image, input_details, height, width):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    if input_details[0]['dtype'] == np.float32:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    return input_data


def detect_objects(image, interpreter, input_details, output_details, height, width):
    input_data = preprocess_image(image, input_details, height, width)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[3]['index'])[0]
    scores = interpreter.get_tensor(output_details[0]['index'])[0]

    detections = []
    for i in range(len(scores)):
        if scores[i] > 0.1:
            ymin = int(boxes[i][0] * image.shape[0])
            xmin = int(boxes[i][1] * image.shape[1])
            ymax = int(boxes[i][2] * image.shape[0])
            xmax = int(boxes[i][3] * image.shape[1])
            detections.append((xmin, ymin, xmax, ymax, classes[i], scores[i]))

    return detections


def draw_boxes(image, detections, labels):
    for (xmin, ymin, xmax, ymax, class_id, score) in detections:
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        label = f'{labels[int(class_id)]}: {score*100:.2f}%'
        cv2.putText(image, label, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image


def convert_avi_to_mp4(avi_path, mp4_path):
    ffmpeg_path = 'C:\\ffmpeg\\ffmpeg-7.0-essentials_build\\bin\\ffmpeg'

    command = [
        ffmpeg_path,
        '-i', avi_path,
        '-codec:v', 'libx264',
        '-crf', '23',
        '-preset', 'medium',
        mp4_path
    ]
    subprocess.run(command, check=True)


def process_video(video_path, model_path, label_path, process_fraction=0.05):
    print(f"Processing video: {video_path}")
    interpreter, input_details, output_details, height, width, labels = load_model(
        model_path, label_path)
    cap = cv2.VideoCapture(video_path)

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_avi_path = video_path.replace('.mp4', '_det.avi')
    output_mp4_path = video_path.replace('.mp4', '_det.mp4')
    out = cv2.VideoWriter(output_avi_path, cv2.VideoWriter_fourcc(
        *'XVID'), fps, (original_width, original_height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    process_frames = int(total_frames * process_fraction)
    frame_interval = max(1, total_frames // process_frames)

    frame_count = 0
    last_detections = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            try:
                detections = detect_objects(
                    frame, interpreter, input_details, output_details, height, width)
                last_detections = detections
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                detections = last_detections
        else:
            detections = last_detections

        processed_frame = draw_boxes(frame, detections, labels)
        out.write(processed_frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Processed video saved to {output_avi_path}")

    convert_avi_to_mp4(output_avi_path, output_mp4_path)
    print(f"Converted video saved to {output_mp4_path}")
    return output_mp4_path
