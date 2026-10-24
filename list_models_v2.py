import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

with open("models_list.txt", "w") as f:
    f.write("Available Models:\n")
    try:
        for m in client.models.list():
            f.write(f"{m.name}\n")
    except Exception as e:
        f.write(f"Error: {e}\n")
print("Done writing models to models_list.txt")
