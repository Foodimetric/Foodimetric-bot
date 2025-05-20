from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging
from main import vector_store, prompt

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='foodimetric_ai_api.log'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Foodimetric AI API",
    description="API endpoint for Foodimetric's nutrition assistant powered by Gemini",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.foodimetric.com/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)

class Query(BaseModel):
    text: str
    user_id: Optional[str] = None

@app.post("/api/chat")
async def chat_endpoint(query: Query):
    try:
        if not query.text:
            raise HTTPException(status_code=400, detail="Query text cannot be empty")
            
        # Create retriever
        retriever = vector_store.as_retriever()
        
        # Get relevant context
        try:
            context = retriever.get_relevant_documents(query.text)
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            raise HTTPException(status_code=500, detail="Error retrieving context from knowledge base")
        
        # Process with LLM
        try:
            chain = (
                prompt 
                | llm 
                | StrOutputParser()
            )
            
            # Get response
            response = chain.invoke({
                "context": context,
                "input": query.text
            })
        except Exception as e:
            logger.error(f"Error processing with LLM: {str(e)}")
            raise HTTPException(status_code=500, detail="Error processing query with AI model")
        
        return {
            "status": "success",
            "response": response,
            "user_id": query.user_id
        }
        
    except HTTPException as he:
        # Re-raise HTTP exceptions as they're already properly formatted
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Foodimetric AI API is running"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 