from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_system import chatbot, simple_chat_manager

# FastAPI app
app = FastAPI(
    title="AllOfTech Chatbot API",
    description="AI-powered chatbot API for AllOfTech agency.",
    version="1.0.0"
)

# Enable CORS (so frontend or Thunder Client can call it)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body model
class ChatRequest(BaseModel):
    message: str
    product: str | None = "AllOfTech"  # optional, default to AllOfTech

@app.get("/")
def root():
    """Health check endpoint"""
    return {"message": "🚀 AllOfTech Chatbot API is running!"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    POST /chat
    JSON Body:
    {
        "message": "Your message",
        "product": "AllOfTech"   # optional
    }
    """
    message = request.message
    product = request.product or "AllOfTech"

    if not message:
        return {"error": "Message is required."}

    # Search for relevant chunks in FAISS
    relevant_chunks = simple_chat_manager.search_similar_chunks(message, product)

    # Generate response
    response_text = simple_chat_manager.generate_response(message, relevant_chunks, product)

    return {"response": response_text, "product": product}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chat_server:app", host="0.0.0.0", port=8000)
