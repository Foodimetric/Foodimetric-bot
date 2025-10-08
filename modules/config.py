import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='foodimetric_ai.log'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize the LLM model
def get_llm():
    """Initialize and return the LLM model"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0
    )