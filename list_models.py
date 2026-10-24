import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("Listing available Gemini models...")
try:
    models = [m.name for m in client.models.list()]
    for name in sorted(models):
        print(name)
except Exception as e:
    print(f"Error listing models: {e}")
