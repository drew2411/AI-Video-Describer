import cv2
import base64
import os
import time
import json
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Define the video processing function
def process_frames_at_interval(video_path, frames_per_second_to_process=1):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise Exception("Could not open video file.")

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frames_per_second_to_process)
    
    frame_count = 0
    while True:
        success, image = vidcap.read()
        if not success:
            break

        if frame_count % frame_interval == 0:
            _, buffer = cv2.imencode('.jpg', image)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            yield base64_image
        
        frame_count += 1
    
    vidcap.release()

# Main script logic
load_dotenv()
video_file_path = r"C:\Users\nikhi\projects\AI-Video-Describer\sample videos\The Hunger Games with audio description_ Katniss hunting.mp4"

video_descriptions = []
total_start_time = time.time()  # Start timing the entire process

try:
    # Get total video length in seconds
    vidcap_info = cv2.VideoCapture(video_file_path)
    if not vidcap_info.isOpened():
        raise Exception("Could not get video information.")
    
    total_frames = int(vidcap_info.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap_info.get(cv2.CAP_PROP_FPS)
    video_length_seconds = round(total_frames / fps, 2)
    vidcap_info.release()
    
    llm = ChatOllama(model="moondream:1.8b") # Correcting the model to a multimodal one
    
    for frame_index, base64_frame in enumerate(process_frames_at_interval(video_file_path)):
        print(f"\n--- Analyzing frame at {frame_index} second mark ---")
        
        prompt_start_time = time.time()  # Start timing this specific prompt
        
        human_message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Analyze this video frame. Describe what is happening, the setting, and any prominent objects or characters.",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_frame}"
                    },
                },
            ]
        )

        analysis_result = llm.invoke([human_message])
        print(analysis_result)
        prompt_end_time = time.time() # End timing this specific prompt
        prompt_duration = round(prompt_end_time - prompt_start_time, 2)
        
        print(analysis_result.content)
        
        video_descriptions.append({
            "timestamp_seconds": frame_index,
            "prompt_duration_seconds": prompt_duration,
            "description": analysis_result.content
        })
        
        print(f"Analysis complete for frame {frame_index + 1}.")

    total_end_time = time.time()
    total_execution_time = round(total_end_time - total_start_time, 2)

    # Prepare final JSON data with summary information
    final_data = {
        "video_path": video_file_path,
        "video_length_seconds": video_length_seconds,
        "total_execution_time_seconds": total_execution_time,
        "analyses": video_descriptions
    }

    output_filename = "video_analysis.json"
    with open(output_filename, 'w') as f:
        json.dump(final_data, f, indent=4)
    
    print(f"\nSuccessfully saved all descriptions to {output_filename}")

except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")