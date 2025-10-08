import chainlit as cl
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from .config import logger, get_llm
from .prompts import get_prompt_template
from .vector_store import initialize_vector_store

# Initialize vector store before starting the app
logger.info("Initializing Foodimetric AI...")
vector_store = initialize_vector_store()
logger.info("Initialization complete! Starting the chat interface...")

@cl.on_chat_start
async def start():
    # Create retriever
    retriever = vector_store.as_retriever()
    
    # Initialize chat history in user session
    cl.user_session.set("chat_history", [])
    
    # Get LLM and prompt
    llm = get_llm()
    prompt = get_prompt_template()
    
    # Create RAG chain
    rag_chain = (
        {
            "context": retriever,
            "chat_history": lambda x: "\n".join(cl.user_session.get("chat_history")),
            "input": RunnablePassthrough()
        }
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    # Store chain in user session
    cl.user_session.set("chain", rag_chain)
    await cl.Message(content="👋 Hi! I'm ready to help you with your nutrition questions. What would you like to know?").send()

@cl.on_chat_end
async def end():
    # Clear chat history when session ends
    cl.user_session.set("chat_history", [])
    cl.user_session.set("chain", None)
    logger.info("Chat session ended - history cleared")

@cl.on_message
async def main(message: cl.Message):
    try:
        # Get the chain from user session
        chain = cl.user_session.get("chain")
        
        if chain is None:
            await cl.Message(content="Please wait for document processing to complete.").send()
            return
            
        chat_history = cl.user_session.get("chat_history", [])
        response = await chain.ainvoke(message.content)
        chat_history.append(f"User: {message.content}")
        chat_history.append(f"Assistant: {response}")
        
        # Keep only last 10 messages to prevent context window issues
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]
        
        # Update chat history in session
        cl.user_session.set("chat_history", chat_history)
        await cl.Message(content=response).send()
        
    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}").send()