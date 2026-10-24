import os
import sys
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# Disable CrewAI telemetry immediately
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

# pyrefly: ignore [missing-import]
from crewai import LLM

# Load environment variables from .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# Set standard keys
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GEMINI_API_KEY"] = GOOGLE_API_KEY

USE_GROQ = os.getenv("USE_GROQ", "false").lower() == "true"

is_valid_groq = bool(GROQ_API_KEY and not any(x in GROQ_API_KEY for x in ["your_groq_api_key_here", "PASTE_YOUR_KEY_HERE"]))
is_valid_google = bool(GOOGLE_API_KEY and not any(x in GOOGLE_API_KEY for x in ["your_gemini_api_key_here", "PASTE_YOUR_KEY_HERE"]))

if is_valid_google and not USE_GROQ:
    llm = LLM(
        model="gemini/gemini-flash-lite-latest",
        temperature=0.3,
        api_key=GOOGLE_API_KEY
    )
    print("[LLM CONFIG] Smart Hire configured to use Gemini LLM (gemini-flash-lite-latest).")
elif is_valid_groq:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        temperature=0.3,
        api_key=GROQ_API_KEY
    )
    print("[LLM CONFIG] Smart Hire configured to use Groq LLM (openai/gpt-oss-120b).")
else:
    print("\n[WARNING] Valid GOOGLE_API_KEY or GROQ_API_KEY not found in environment/.env.")
    print("Running with default Gemini Flash configuration. Please provide a valid key before executing live inferences.")
    llm = LLM(
        model="gemini/gemini-2.5-flash",
        temperature=0.4,
        api_key=GOOGLE_API_KEY or "dummy_key_for_initialization"
    )

# Configuration for CrewAI Agents
AGENT_CONFIG = {
    "llm": llm,
    "verbose": True,
    "allow_delegation": False
}
