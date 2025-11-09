import cv2
import base64
import os

def encode_first_frame_to_base64(video_path):
    """
    Extracts the first frame of a video and encodes it to a base64 string.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    
    if success:
        _, buffer = cv2.imencode('.jpg', image)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        vidcap.release()
        return base64_image
    else:
        vidcap.release()
        raise Exception("Could not read the first frame of the video.")