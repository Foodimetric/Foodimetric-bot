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
import socketio
from fastapi.staticfiles import StaticFiles

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='nutribot_api.log'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Socket.IO
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')

# Initialize FastAPI app
app = FastAPI(
    title="NutriBot API",
    description="API endpoint for Foodimetric's nutrition assistant powered by Gemini",
    version="1.0.0"
)

# Create ASGI app by combining FastAPI and Socket.IO
socket_app = socketio.ASGIApp(sio, app)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

class Query(BaseModel):
    text: str
    user_id: Optional[str] = None

@sio.event
async def connect(sid, environ):
    logger.info(f"Client connected: {sid}")
    await sio.emit('connected', {'status': 'Connected successfully'}, room=sid)

@sio.event
async def disconnect(sid):
    logger.info(f"Client disconnected: {sid}")

@sio.event
async def chat_message(sid, data):
    try:
        # Create retriever
        retriever = vector_store.as_retriever()
        
        # Get relevant context
        context = retriever.get_relevant_documents(data['text'])
        
        # Process with LLM
        chain = (
            prompt 
            | llm 
            | StrOutputParser()
        )
        
        # Get response
        response = chain.invoke({
            "context": context,
            "input": data['text']
        })
        
        await sio.emit('chat_response', {
            "status": "success",
            "response": response,
            "user_id": data.get('user_id')
        }, room=sid)
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        await sio.emit('error', {'error': str(e)}, room=sid)

@app.post("/api/chat")
async def chat_endpoint(query: Query):
    try:
        # Create retriever
        retriever = vector_store.as_retriever()
        
        # Get relevant context
        context = retriever.get_relevant_documents(query.text)
        
        # Process with LLM
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
        
        return {
            "status": "success",
            "response": response,
            "user_id": query.user_id
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "NutriBot API is running"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(socket_app, host="0.0.0.0", port=8080) 