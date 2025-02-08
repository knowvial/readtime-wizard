from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from enum import Enum
import uvicorn
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our ML model
from readtime_model import ReadTimeModel

class Genre(str, Enum):
    FICTION = "fiction"
    NON_FICTION = "non-fiction"
    TEXTBOOK = "textbook"
    TECHNICAL = "technical"

class ReadingLevel(int, Enum):
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3

class ReadingSpeed(int, Enum):
    SLOW = 1
    MEDIUM = 2
    FAST = 3

class ReadTimeRequest(BaseModel):
    book_title: str = Field(..., min_length=1, max_length=200)
    pages: int = Field(..., gt=0, le=5000)
    reading_level: ReadingLevel
    genre: Genre
    available_time: int = Field(..., gt=0, le=1440)  # Max 24 hours in minutes
    reading_speed: ReadingSpeed
    
    class Config:
        schema_extra = {
            "example": {
                "book_title": "The Great Gatsby",
                "pages": 180,
                "reading_level": 2,
                "genre": "fiction",
                "available_time": 60,
                "reading_speed": 2
            }
        }

class ReadTimeResponse(BaseModel):
    book_title: str
    total_hours: float
    days_to_complete: int
    confidence: float
    reading_tips: List[str]
    schedule: List[dict]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

app = FastAPI(
    title="ReadTime Wizard API",
    description="API for estimating book reading time and generating reading schedules",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None

@app.on_event("startup")
async def load_model():
    """Load the ML model on startup."""
    global model
    try:
        model = ReadTimeModel.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError("Failed to initialize the model")

def get_model():
    """Dependency to get the model instance."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return model

def generate_reading_schedule(days: int, available_time: int, total_hours: float) -> List[dict]:
    """Generate a daily reading schedule."""
    total_minutes = total_hours * 60
    daily_pages = total_minutes / days
    
    schedule = []
    for day in range(1, days + 1):
        schedule.append({
            "day": day,
            "minutes": available_time,
            "cumulative_progress": round(min((day * available_time / total_minutes) * 100, 100), 1)
        })
    
    return schedule

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "ReadTime Wizard API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/estimate", response_model=ReadTimeResponse, responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
async def estimate_reading_time(request: ReadTimeRequest, model: ReadTimeModel = Depends(get_model)):
    """
    Estimate reading time for a book based on various factors.
    """
    try:
        # Prepare features for prediction
        features = {
            'pages': request.pages,
            'reading_level': request.reading_level.value,
            'genre': request.genre.value,
            'available_time': request.available_time,
            'reading_speed': request.reading_speed.value
        }
        
        # Get model predictions
        prediction = model.predict(features)
        
        # Get reading tips
        reading_tips = model.get_reading_tips(request.genre.value)
        
        # Generate reading schedule
        schedule = generate_reading_schedule(
            prediction['days_to_complete'],
            request.available_time,
            prediction['total_hours']
        )
        
        return ReadTimeResponse(
            book_title=request.book_title,
            total_hours=prediction['total_hours'],
            days_to_complete=prediction['days_to_complete'],
            confidence=prediction['confidence'],
            reading_tips=reading_tips,
            schedule=schedule
        )
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Error processing request: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)