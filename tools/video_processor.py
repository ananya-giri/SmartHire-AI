import os
import time
from config import GOOGLE_API_KEY

def analyze_video(video_path: str, prompt: str) -> str:
    """
    Uploads a video to Google GenAI File API and analyzes it with a prompt.
    """
    if not os.path.exists(video_path):
        return "Video analysis error: File not found."

    if not GOOGLE_API_KEY or "PASTE_YOUR_KEY_HERE" in GOOGLE_API_KEY:
        return "Video analysis skipped: Valid GOOGLE_API_KEY required in .env."

    try:
        from google import genai
        client = genai.Client(api_key=GOOGLE_API_KEY)
        
        print(f"Uploading file: {video_path}...")
        video_file = client.files.upload(path=video_path)
        print(f"Completed upload: {getattr(video_file, 'uri', 'uploaded')}")

        # Wait for the file to be processed (up to 60 seconds)
        attempts = 0
        while attempts < 12:
            state_str = getattr(video_file.state, 'name', str(video_file.state)).upper()
            if "PROCESSING" not in state_str:
                break
            print('.', end='', flush=True)
            time.sleep(5)
            attempts += 1
            video_file = client.files.get(name=video_file.name)

        state_str = getattr(video_file.state, 'name', str(video_file.state)).upper()
        if "FAILED" in state_str:
            raise ValueError(f"Video processing failed: {state_str}")

        print("\nAnalyzing video...")
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[video_file, prompt]
            )
        except Exception:
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=[video_file, prompt]
            )
        
        return response.text
    except Exception as e:
        return f"Video analysis error: {str(e)}"

def deepfake_and_teleprompter_analysis(video_path: str) -> str:
    """
    Acts as a Multimodal Integrity Checker using the vision model to detect 
    candidates reading from ChatGPT off-screen or using deepfake avatars.
    """
    forensic_prompt = """
    You are an expert Forensic AI Audio-Visual Analyst. You are tasked with analyzing this async interview video submission.
    Please explicitly analyze the video for the following integrity flags:
    1. Teleprompter / Second Screen Reading: Are the candidate's eyes rhythmically tracking text off-screen? Is there an unnatural lack of direct eye contact?
    2. Deepfake / Avatar Usage: Are there unnatural lip-sync mismatches, robotic head movements, missing blinks, or synthetic facial rendering anomalies (e.g., blurring around the edges)?
    3. Voice Cloning: Does the voice sound unusually robotic, lack natural breathing pauses, or have synthetic artifacts?
    
    Provide a "Video Authenticity Score" from 0 to 100, and explicitly list any RED FLAGS found for deepfakes or teleprompters. Provide a detailed explanation of your visual findings.
    """
    return analyze_video(video_path, forensic_prompt)
