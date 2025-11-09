import cv2
import os
import time
import json
import chromadb
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import moondream as md
from PIL import Image
from typing import List, Dict, Any
from gtts import gTTS


# Define the video processing function
def process_frames_at_interval(video_path: str, frames_per_second_to_process: int = 1):
    """
    Processes video frames at a specified interval, yielding an image as a PIL object
    and the corresponding timestamp in seconds.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise Exception("Could not open video file.")

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    print("Video FPS:", fps)

    # Calculate the number of frames to skip to get a frame every second
    frame_interval = int(round(fps / frames_per_second_to_process))

    frame_count = 0
    while True:
        success, image_np = vidcap.read()
        if not success:
            break

        if frame_count % frame_interval == 0:
            # Convert the OpenCV image (numpy array) to a PIL Image object
            image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
            # Calculate the timestamp in seconds based on the frame count
            current_time_seconds = int(frame_count // fps)
            yield image_pil, current_time_seconds

        frame_count += 1

    vidcap.release()


# --- New Summarization Function ---
def summarize_captions(llm: ChatGroq, captions: List[str], summary_type: str) -> str:
    """
    Uses Groq to summarize a list of captions.
    """
    captions_string = "\n".join(captions)
    prompt = f"You are creating a summary for an audio description for the visually impaired. Summarize the key events from the following descriptions into a brief, narrative paragraph. This is a {summary_type} summary.\n\nDescriptions:\n{captions_string}\n\nNarrative Summary:"
    try:
        summary = llm.invoke(prompt).content
        return summary
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "Failed to generate summary."


# --- New TTS Function ---
def generate_tts_audio(text: str, output_path: str):
    """
    Generates a TTS audio file from text using gTTS.
    """
    try:
        tts = gTTS(text=text, lang="en")
        tts.save(output_path)
        print(f"Successfully saved TTS audio to {output_path}")
    except Exception as e:
        print(f"Failed to generate TTS audio: {e}")


# --- Main Script Logic ---
def main():
    load_dotenv()
    video_file_path = "C:\\Users\\nikhi\\projects\\AI-Video-Describer\\sample videos\\Jack Jack Attack - The Replacement Sitter.mp4"
    video_name = os.path.basename(video_file_path)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY1")
    MOONDREAM_API_KEY = os.getenv("MOONDREAM_API_KEY")

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")
    if not MOONDREAM_API_KEY:
        raise ValueError(
            "MOONDREAM_API_KEY not found. Please set it in your .env file."
        )

    # 1. Initialize ChromaDB client and collections
    client = chromadb.PersistentClient(path="./video_captions_db")
    # Collection for detailed frame captions
    caption_collection = client.get_or_create_collection(name="video_captions")
    # Collection for 10-second summaries
    ten_second_summary_collection = client.get_or_create_collection(
        name="ten_second_summaries"
    )
    # Collection for 1-minute summaries
    minute_summary_collection = client.get_or_create_collection(name="minute_summaries")

    # 2. Initialize the LLMs
    groq_llm = ChatGroq(
        model_name="meta-llama/llama-4-scout-17b-16e-instruct", api_key=GROQ_API_KEY
    )
    moondream_model = md.vl(api_key=MOONDREAM_API_KEY)

    video_descriptions = []
    ten_second_captions_buffer = []
    ten_second_summaries_buffer = []
    total_start_time = time.time()

    # Create directory for audio output
    audio_output_dir = "./audio_output"
    os.makedirs(audio_output_dir, exist_ok=True)

    try:
        vidcap_info = cv2.VideoCapture(video_file_path)
        if not vidcap_info.isOpened():
            raise Exception("Could not get video information.")
        total_frames = int(vidcap_info.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vidcap_info.get(cv2.CAP_PROP_FPS)
        video_length_seconds = round(total_frames / fps, 2)
        vidcap_info.release()

        for image_pil, current_time_seconds in process_frames_at_interval(
            video_file_path
        ):
            print(f"\n--- Analyzing frame at {current_time_seconds} second mark ---")

            # Step 1: Get recent captions for Moondream context
            if current_time_seconds > 0:
                recent_captions = caption_collection.get(
                    ids=[
                        f"{video_name}_{t}"
                        for t in range(
                            max(0, current_time_seconds - 5), current_time_seconds
                        )
                    ],
                    include=["documents"],
                )["documents"]
                recent_context_text = " ".join(recent_captions)
            else:
                recent_context_text = ""

            moondream_prompt = f"You are an audio describer for the visually impaired. Your output will be read by a narrator. Describe this video frame briefly, focusing only on essential actions and characters. Your response MUST ONLY contain the description text. Do not add any extra conversational text. Previous context: '{recent_context_text}'"

            # --- Add retry logic for Moondream API call ---
            initial_caption = None
            for attempt in range(3):  # Retry up to 3 times
                try:
                    initial_caption = moondream_model.query(
                        image=image_pil, question=moondream_prompt
                    )["answer"]
                    break  # Success, exit loop
                except Exception as e:
                    print(f"Moondream API call failed (attempt {attempt + 1}/3): {e}")
                    if attempt < 2:
                        time.sleep(5 * (attempt + 1))  # Exponential backoff
                    else:
                        raise  # Re-raise the exception after the last attempt

            print(f"Initial Caption: {initial_caption}")

            # Step 2: Query ChromaDB for relevant context chunks
            if not initial_caption:
                print("Skipping frame due to failure in getting initial caption.")
                continue

            relevant_chunks_docs = caption_collection.query(
                query_texts=[initial_caption],
                n_results=5,
                where={
                    "$and": [
                        {"video_name": {"$eq": video_name}},
                        {"timestamp_seconds": {"$lte": current_time_seconds}},
                    ]
                },
                include=["documents", "metadatas"],
            )
            relevant_context = (
                " ".join(relevant_chunks_docs["documents"][0])
                if relevant_chunks_docs["documents"]
                and relevant_chunks_docs["documents"][0]
                else ""
            )

            # Step 3: Recontextualize using Groq
            recontext_prompt = (
                f"You are an audio description model for the visually impaired. Your output will be read by a narrator. "
                f"Synthesize the 'Initial Description' and 'Historical Context' into a single, brief, narrative sentence. "
                f"Your response MUST ONLY contain the final description text. Do not include conversational phrases or introductions like 'Here is the final audio description:'.\n\n"
                f"Initial Description: '{initial_caption}'\n"
                f"Historical Context: '{relevant_context}'\n\n"
                f"Final Description:"
            )

            # --- Add retry logic for Groq API call ---
            final_caption = None
            for attempt in range(3):  # Retry up to 3 times
                try:
                    final_caption = groq_llm.invoke(recontext_prompt).content
                    break  # Success, exit loop
                except Exception as e:
                    print(f"Groq API call failed (attempt {attempt + 1}/3): {e}")
                    if attempt < 2:
                        time.sleep(5 * (attempt + 1))  # Exponential backoff
                    else:
                        raise  # Re-raise the exception after the last attempt

            print(f"Final Caption: {final_caption}")

            # Step 4: Generate TTS audio for the final caption
            if final_caption:
                audio_file_path = os.path.join(
                    audio_output_dir, f"caption_{current_time_seconds}.mp3"
                )
                generate_tts_audio(final_caption, audio_file_path)

            # Step 5: Store the final caption
            caption_collection.add(
                documents=[final_caption],
                metadatas=[
                    {
                        "timestamp_seconds": current_time_seconds,
                        "video_name": video_name,
                    }
                ],
                ids=[f"{video_name}_{current_time_seconds}"],
            )
            video_descriptions.append(
                {
                    "timestamp_seconds": current_time_seconds,
                    "description": final_caption,
                }
            )
            ten_second_captions_buffer.append(final_caption)

            # --- Tiered Summarization Logic ---
            if (current_time_seconds > 0 and (current_time_seconds + 1) % 10 == 0) or (
                current_time_seconds + 1 >= video_length_seconds
            ):
                if ten_second_captions_buffer:
                    print("\n--- Generating 10-Second Summary ---")
                    summary_10s = summarize_captions(
                        groq_llm, ten_second_captions_buffer, "10-second"
                    )
                    ten_second_summary_collection.add(
                        documents=[summary_10s],
                        metadatas=[
                            {
                                "start_time": current_time_seconds
                                - len(ten_second_captions_buffer)
                                + 1,
                                "end_time": current_time_seconds,
                                "video_name": video_name,
                            }
                        ],
                        ids=[f"{video_name}_10s_{current_time_seconds}"],
                    )
                    ten_second_summaries_buffer.append(summary_10s)
                    ten_second_captions_buffer = []
                    print(f"10-second Summary: {summary_10s}")

            if (current_time_seconds > 0 and (current_time_seconds + 1) % 60 == 0) or (
                current_time_seconds + 1 >= video_length_seconds
            ):
                if ten_second_summaries_buffer:
                    print("\n--- Generating 1-Minute Summary ---")
                    summary_60s = summarize_captions(
                        groq_llm, ten_second_summaries_buffer, "1-minute"
                    )
                    minute_summary_collection.add(
                        documents=[summary_60s],
                        metadatas=[
                            {
                                "start_time": current_time_seconds - 59,
                                "end_time": current_time_seconds,
                                "video_name": video_name,
                            }
                        ],
                        ids=[f"{video_name}_60s_{current_time_seconds}"],
                    )
                    ten_second_summaries_buffer = []
                    print(f"1-minute Summary: {summary_60s}")

        total_end_time = time.time()
        total_execution_time = round(total_end_time - total_start_time, 2)

        final_data = {
            "video_path": video_file_path,
            "video_length_seconds": video_length_seconds,
            "total_execution_time_seconds": total_execution_time,
            "analyses": video_descriptions,
        }

        output_filename = "video_analysis.json"
        with open(output_filename, "w") as f:
            json.dump(final_data, f, indent=4)

        print(f"\nSuccessfully saved all descriptions to {output_filename}")
        print("\n--- Final Summaries ---")
        print("10-Second Summaries:")
        for doc in ten_second_summary_collection.get(
            where={"video_name": video_name}, include=["documents"]
        )["documents"]:
            print(f"- {doc}")
        print("\n1-Minute Summaries:")
        for doc in minute_summary_collection.get(
            where={"video_name": video_name}, include=["documents"]
        )["documents"]:
            print(f"- {doc}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
