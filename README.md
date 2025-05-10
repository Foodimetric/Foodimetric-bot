# 🥗 Foodimetric AI - Academic Nutrition Assistant

Foodimetric AI is a Chainlit-powered conversational AI that provides evidence-based nutrition information. Built with Google's Gemini model and advanced RAG (Retrieval Augmented Generation) capabilities, it helps users understand nutrition concepts through an intuitive chat interface and REST API.

## 🔧 Prerequisites

- Python 3.8+
- Google Gemini API key

## 📦 Installation

1. Clone the repository:

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your Google API key:
   ```bash
   GEMINI_API_KEY=your_api_key_here
   ```

## 📚 Usage

### Chainlit Web Interface

1. Run the web application:
   ```bash
   chainlit run main.py --port 8080
   ```

2. Open your web browser and navigate to the provided URL (typically `http://localhost:8080`)

### REST API

1. Start the API server:
   ```bash
   python api.py
   ```

2. The API will be available at `http://localhost:8080`

#### Testing the API

You can test the API in several ways:

1. Using curl:
   ```bash
   # Test the health check endpoint
   curl http://localhost:8080/

   # Test the chat endpoint
   curl -X POST "http://localhost:8080/api/chat" \
        -H "Content-Type: application/json" \
        -d '{"text": "What are good sources of protein?", "user_id": "optional-user-id"}'

   # In windows powershell
   curl -Method POST "http://localhost:8080/api/chat" -Headers @{ "Content-Type" = "application/json" } -Body '{"text": "What are good sources of protein?", "user_id": "user123"}'

   ```

2. Using Python requests:
   ```python
   import requests

   # Test the chat endpoint
   response = requests.post(
       "http://localhost:8080/api/chat",
       json={
           "text": "What are good sources of protein?",
           "user_id": "user123"
       }
   )
   print(response.json())
   ```

3. Using WebSocket connection (with Python):
   ```python
   import socketio
   import asyncio

   async def test_socket():
       sio = socketio.AsyncClient()

       @sio.event
       async def connect():
           print("Connected!")
           await sio.emit('chat_message', {
               'text': 'What are good sources of protein?',
               'user_id': 'user123'
           })

       @sio.event
       async def chat_response(data):
           print("Received response:", data)
           await sio.disconnect()

       await sio.connect('http://localhost:8080')
       await sio.wait()

   asyncio.run(test_socket())
   ```

4. Using WebSocket connection (with JavaScript):
   ```javascript
   // In your HTML file, include Socket.IO client
   // <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>

   const socket = io('http://localhost:8080');

   socket.on('connect', () => {
       console.log('Connected to server');
       
       // Send a test message
       socket.emit('chat_message', {
           text: "What are good sources of protein?",
           user_id: "user123"
       });
   });

   socket.on('chat_response', (data) => {
       console.log('Received response:', data);
   });

   socket.on('error', (error) => {
       console.error('Error:', error);
   });
   ```

Expected Response Format:
```json
{
    "status": "success",
    "response": "...",
    "user_id": "user123"
}
```

## 📖 API Documentation

Once the API is running, you can access the interactive API documentation at:
- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`

Replace `"*"` in CORS settings with your specific frontend domain

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

1. Make sure you have Docker and Docker Compose installed on your system.

2. Create a `.env` file with your Gemini API key:
   ```bash
   GEMINI_API_KEY=your_api_key_here
   ```

3. Build and start the container:
   ```bash
   docker-compose up -d
   ```

4. The API will be available at `http://localhost:8080`

### Stopping the Container

- If using Docker Compose:
  ```bash
  docker-compose down
  ```