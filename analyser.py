import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from video_processor import encode_first_frame_to_base64 # Import the function
from dotenv import load_dotenv
# Replace with the path to your video file
load_dotenv()
video_file_path = r"C:\Users\nikhi\projects\AI-Video-Describer\sample videos\The Hunger Games with audio description_ Katniss hunting.mp4"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

try:
    # STEP 1: Call the imported function
    base64_frame = encode_first_frame_to_base64(video_file_path)
    print("First frame successfully extracted and encoded.")

    # STEP 2: Use LangChain and Groq to analyze the encoded frame
    llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",api_key=GROQ_API_KEY)

    human_message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Analyze the first frame of this video. Describe what is happening, the setting, and any prominent objects or characters.",
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
    
    print("\n--- Analysis of the First Frame ---")
    print(analysis_result.content)

except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")