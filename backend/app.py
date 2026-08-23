from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import asyncio
from model.transformer import UltiTransformer
from inference.engine import InferenceEngine
from integrations.internet import InternetSearch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ulti-AI", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models
transformer_model = None
inference_engine = None
internet_search = None

class CodeRequest(BaseModel):
    prompt: str
    language: str = "auto"
    max_tokens: int = 2048
    temperature: float = 0.7
    search_context: bool = True

class CodeResponse(BaseModel):
    code: str
    language: str
    explanation: str
    confidence: float
    search_results: list = []

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global transformer_model, inference_engine, internet_search
    
    logger.info("Loading Ulti-AI Transformer Model...")
    transformer_model = UltiTransformer.load_pretrained("ulti-ai-v1")
    inference_engine = InferenceEngine(transformer_model)
    internet_search = InternetSearch()
    logger.info("✅ Ulti-AI Ready!")

@app.get("/")
async def root():
    return {
        "status": "online",
        "model": "Ulti-AI Transformer v1",
        "capabilities": ["code generation", "debugging", "optimization", "documentation"],
        "languages_supported": "ALL"
    }

@app.post("/generate", response_model=CodeResponse)
async def generate_code(request: CodeRequest):
    """Generate code for any programming language"""
    try:
        # Search context if enabled
        search_results = []
        if request.search_context:
            search_results = await internet_search.search(request.prompt)
        
        # Generate code with transformer
        result = await inference_engine.generate(
            prompt=request.prompt,
            language=request.language,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            context=search_results
        )
        
        return CodeResponse(
            code=result["code"],
            language=result["language"],
            explanation=result["explanation"],
            confidence=result["confidence"],
            search_results=search_results
        )
    
    except Exception as e:
        logger.error(f"Error generating code: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debug")
async def debug_code(code: str, language: str = "auto"):
    """Debug code and suggest fixes"""
    try:
        result = await inference_engine.debug(code, language)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize")
async def optimize_code(code: str, language: str = "auto"):
    """Optimize code for performance"""
    try:
        result = await inference_engine.optimize(code, language)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
