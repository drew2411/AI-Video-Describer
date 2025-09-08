import cv2
import base64
import os
import time
import json
import chromadb
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import moondream as md
from PIL import Image

# Define the video processing function
def process_frames_at_interval(video_path, frames_per_second_to_process=1):
    """
    Processes video frames at a specified interval, yielding an image as a PIL object.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise Exception("Could not open video file.")

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frames_per_second_to_process)
    
    frame_count = 0
    while True:
        success, image_np = vidcap.read()
        if not success:
            break

        if frame_count % frame_interval == 0:
            # Convert the OpenCV image (numpy array) to a PIL Image object
            image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
            yield image_pil
        
        frame_count += 1
    
    vidcap.release()

# --- Main Script Logic ---
load_dotenv()
video_file_path = r"C:\Users\nikhi\projects\AI-Video-Describer\sample videos\The Hunger Games with audio description_ Katniss hunting.mp4"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MOONDREAM_API_KEY = os.getenv("MOONDREAM_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")
if not MOONDREAM_API_KEY:
    raise ValueError("MOONDREAM_API_KEY not found. Please set it in your .env file.")

# 1. Initialize ChromaDB client and collection
client = chromadb.PersistentClient(path="./video_captions_db")
# Using a default embedding function from ChromaDB, for simplicity
# You may want to use a more robust one like 'all-MiniLM-L6-v2' for better performance
collection = client.get_or_create_collection(name="video_captions")

# 2. Initialize the LLMs
groq_llm = ChatGroq(model_name="Llama-3-Vision-Preview", api_key=GROQ_API_KEY)
moondream_model = md.vl(api_key=MOONDREAM_API_KEY)

video_descriptions = []
total_start_time = time.time()

try:
    # Get total video length in seconds
    vidcap_info = cv2.VideoCapture(video_file_path)
    if not vidcap_info.isOpened():
        raise Exception("Could not get video information.")
    total_frames = int(vidcap_info.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap_info.get(cv2.CAP_PROP_FPS)
    video_length_seconds = round(total_frames / fps, 2)
    vidcap_info.release()
    
    for frame_index, image_pil in enumerate(process_frames_at_interval(video_file_path)):
        print(f"\n--- Analyzing frame at {frame_index} second mark ---")
        prompt_start_time = time.time()

        # Step 1: Get the initial caption from Moondream
        initial_caption = moondream_model.query(
            image=image_pil, 
            question="Describe this video frame as though you are providing a caption for an audience."
        )["answer"]
        print(f"Initial Caption: {initial_caption}")

        # Step 2: Retrieve context from ChromaDB (last 5 seconds)
        recent_timestamps = list(range(max(0, frame_index - 5), frame_index))
        if recent_timestamps:
            recent_context = collection.get(
                ids=[str(t) for t in recent_timestamps],
                include=["documents"]
            )['documents']
        else:
            recent_context = []

        # Step 3: Query ChromaDB for similar chunks based on the new caption
        # The query will use the initial caption to find similar past frames
        similar_chunks = collection.query(
            query_texts=[initial_caption],
            n_results=3, # Retrieve top 3 similar results
            where={"timestamp_seconds": {"$lte": frame_index}}, # Only look in past frames
            include=["documents"]
        )['documents'][0]

        # Step 4: Recontextualize using Groq with all the info
        recontext_prompt = (
            f"You are an expert video describer. Your task is to recontextualize a new caption "
            f"by using historical context. Here is the new caption for the current frame: '{initial_caption}'.\n\n"
            f"Here are the captions from the last 5 seconds of the video:\n{recent_context}\n\n"
            f"Additionally, here is some information from a database that MIGHT be relevant. "
            f"It is not guaranteed to be relevant, so use your best judgment. "
            f"The information is:\n{similar_chunks}\n\n"
            f"Based on all this information, write a final, comprehensive caption for an audience."
        )

        final_caption = groq_llm.invoke(recontext_prompt).content
        print(f"Final Caption: {final_caption}")

        # Step 5: Store the final recontextualized caption in ChromaDB
        collection.add(
            documents=[final_caption],
            metadatas=[{"timestamp_seconds": frame_index}],
            ids=[str(frame_index)]
        )

        prompt_end_time = time.time()
        prompt_duration = round(prompt_end_time - prompt_start_time, 2)
        
        video_descriptions.append({
            "timestamp_seconds": frame_index,
            "prompt_duration_seconds": prompt_duration,
            "description": final_caption
        })
        
        print(f"Analysis complete for frame {frame_index + 1}.")

    total_end_time = time.time()
    total_execution_time = round(total_end_time - total_start_time, 2)

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