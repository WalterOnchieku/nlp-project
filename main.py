from fastapi import FastAPI
from pydantic import BaseModel
from analysis_engine import engine

# ========1. Create the FastAPI application ===========

app = FastAPI()


# ========2. Request model ===========

class ReviewRequest(BaseModel):
    text: str


# ========3. Response model ===========

class ReviewResponse(BaseModel): 
    text: str
    sentiment: str
    confidence: float 
    entities: list 
    topics: list

# ======== 4. Health check endpoint ===========

@app.get("/health") 
def health(): 
    return { "status": "ok" }


# ========5. Analysis Endpoint ===========

@app.post("/analyse", response_model=ReviewResponse)
def analyse_review(request: ReviewRequest):

    # temporary response while building nlp engine
    # return{
    #     "text": request.text,
    #     "sentiment": "positive",
    #     "confidence": 0.95,
    #     "entities": [],
    #     "topics": []
    # }

    # sent user's text to nlp engine
    result = engine.analyse(request.text)

    return result