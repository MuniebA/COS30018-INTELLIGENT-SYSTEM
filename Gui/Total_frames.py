import cv2


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Get the frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get the total frame count
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the duration in seconds
    duration = frame_count / fps

    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration
    }


# Replace with the path to your video file
video_path = 'static/videos/test_2.mp4'
video_info = get_video_info(video_path)

if video_info:
    print(f"Frame rate: {video_info['fps']} fps")
    print(f"Total frames: {video_info['frame_count']}")
    print(f"Duration: {video_info['duration']} seconds")
