from fastapi import FastAPI
from pydantic import BaseModel
from analysis_engine import engine

import pandas as pd

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

@app.get("/evaluate-real-reviews")
def evaluate_real_reviews():

    #load the real reviews
    df = pd.read_csv("real_customer_reviews_20.csv")

    results = []
    correct = 0

    for _, row in df.iterrows():

        #send the review through nlp engine
        prediction = engine.analyse(row["review"])

        #compare model prediction with expected sentiment
        predicted = prediction["sentiment"]
        expected = row["expected_sentiment"]

        is_correct = predicted == expected

        if is_correct:
            correct += 1

        
        results.append({
            "id": int(row["eval_id"]),
            "review": row["review"],
            "expected": expected,
            "predicted": predicted,
            "confidence": prediction["confidence"],
            "correct": is_correct,
            "entities": prediction["entities"],
            "topics": prediction["topics"]
        })

    #calculate accuracy
    accuracy = correct / len(df)

    return{
        "total_reviews": len(df),
        "correct_predictions": correct,
        "incorrect_predictions": len(df) - correct,
        "acuracy": round(accuracy, 3),
        "accuracy_percentage": round(accuracy * 100, 2),
        "results": results
    }