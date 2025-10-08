from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import logging
from datetime import datetime, timedelta
import json
from modules.config import get_llm
from modules.prompts import get_prompt_template
from modules.vector_store import initialize_vector_store

api_logger = logging.getLogger(__name__)

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

# Initialize the model and vector store
llm = get_llm()

try:
    vector_store = initialize_vector_store()
    api_logger.info("Vector store initialized successfully")
except Exception as e:
    api_logger.error(f"Failed to initialize vector store: {str(e)}")
    import traceback
    api_logger.error(traceback.format_exc())
    raise

prompt = get_prompt_template()

# In-memory chat history storage with expiration
chat_histories: Dict[str, Dict] = {}

def cleanup_expired_sessions():
    """Remove expired chat sessions"""
    current_time = datetime.now()
    expired_sessions = [
        user_id for user_id, data in chat_histories.items()
        if current_time - data['last_activity'] > timedelta(hours=24)
    ]
    for user_id in expired_sessions:
        del chat_histories[user_id]
        api_logger.info(f"Cleaned up expired session for user: {user_id}")

def get_chat_history(user_id: str) -> List[str]:
    """Get chat history for a user"""
    if user_id not in chat_histories:
        chat_histories[user_id] = {
            'history': [],
            'last_activity': datetime.now()
        }
    return chat_histories[user_id]['history']

def update_chat_history(user_id: str, user_message: str, assistant_message: str):
    """Update chat history for a user"""
    if user_id not in chat_histories:
        chat_histories[user_id] = {
            'history': [],
            'last_activity': datetime.now()
        }
    
    chat_histories[user_id]['history'].append(f"User: {user_message}")
    chat_histories[user_id]['history'].append(f"Assistant: {assistant_message}")
    chat_histories[user_id]['last_activity'] = datetime.now()
    
    # Keep only last 10 messages
    if len(chat_histories[user_id]['history']) > 10:
        chat_histories[user_id]['history'] = chat_histories[user_id]['history'][-10:]

class Query(BaseModel):
    text: str
    user_id: Optional[str] = None

@app.post("/api/chat")
async def chat_endpoint(query: Query):
    try:
        if not query.text:
            raise HTTPException(status_code=400, detail="Query text cannot be empty")
        
        # Cleanup expired sessions
        cleanup_expired_sessions()
            
        # Create retriever
        retriever = vector_store.as_retriever()
        
        # Get relevant context
        try:
            context = retriever.invoke(query.text)
        except AssertionError as ae:
            api_logger.error(f"FATAL RETRIEVAL ERROR: Failed to invoke retriever.")
            import traceback
            api_logger.error(traceback.format_exc())
            api_logger.error(f"Exception details: {str(ae)}")
            api_logger.error(f"Error retrieving context: {str(ae)}")
            
            # This is likely an embedding dimension mismatch
            error_msg = (
                "Vector store embedding dimension mismatch. "
                "The vector store was created with a different embedding model. "
                "Please rebuild the vector store or ensure consistent embedding models."
            )
            api_logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        except Exception as e:
            api_logger.error(f"Error retrieving context: {str(e)}")
            import traceback
            api_logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error retrieving context: {str(e)}")
        
        # Get chat history
        chat_history = get_chat_history(query.user_id) if query.user_id else []
        
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
                "chat_history": "\n".join(chat_history),
                "input": query.text
            })
            
            # Update chat history if user_id is provided
            if query.user_id:
                update_chat_history(query.user_id, query.text, response)
            
        except Exception as e:
            api_logger.error(f"Error processing with LLM: {str(e)}")
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
        api_logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.delete("/api/chat/{user_id}")
async def clear_chat_history(user_id: str):
    """Clear chat history for a specific user"""
    try:
        if user_id in chat_histories:
            del chat_histories[user_id]
            api_logger.info(f"Cleared chat history for user: {user_id}")
        return {"status": "success", "message": "Chat history cleared"}
    except Exception as e:
        api_logger.error(f"Error clearing chat history: {str(e)}")
        raise HTTPException(status_code=500, detail="Error clearing chat history")

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Foodimetric AI API is running"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 