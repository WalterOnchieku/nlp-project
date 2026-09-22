# Customer Review Text Analysis System

A Natural Language Processing (NLP) application for analyzing customer reviews using a fine-tuned DistilBERT sentiment classifier, Named Entity Recognition (NER), and topic detection.

The system provides a FastAPI interface through which users can submit customer reviews and receive structured NLP analysis.

---

## 1. Project Overview

Customer reviews contain useful information about how customers perceive products and services. Manually analyzing large numbers of reviews can be time-consuming, particularly when the reviews contain different sentiments, product references, and discussion topics.

This project implements an NLP-based text analysis system that automatically analyzes customer reviews and extracts:

* Sentiment
* Sentiment confidence
* Named entities
* Relevant topics

The sentiment classifier is based on **DistilBERT** and is fine-tuned for three sentiment classes:

* Negative
* Neutral
* Positive

The application exposes the trained model through a **FastAPI REST API**.

---

## 2. Assignment Requirements

The project was developed according to the following NLP application requirements.

| Requirement                               | Implementation                                             | Status                                        |
| ----------------------------------------- | ---------------------------------------------------------- | --------------------------------------------- |
| Select an NLP domain                      | Customer reviews                                           | Completed                                     |
| Sentiment analysis                        | Fine-tuned DistilBERT sequence-classification model        | Implemented                                   |
| Use a pretrained/fine-tuned model         | `distilbert-base-uncased` fine-tuned on the review dataset | Implemented                                   |
| Additional NLP capability                 | Named Entity Recognition and topic detection               | Implemented                                   |
| Prediction interface                      | FastAPI REST API                                           | Implemented                                   |
| Test at least 20 real examples            | Separate evaluation using genuine customer reviews         | To be completed                               |
| Report accuracy/quality                   | Accuracy, precision, recall and weighted F1                | Implemented; final results pending evaluation |
| Short report on architecture and findings | Documented in this README and project report               | In progress                                   |

---

## 3. NLP Capabilities

### 3.1 Sentiment Analysis

The primary NLP task is three-class sentiment classification.

The model predicts:

```text
0 → Negative
1 → Neutral
2 → Positive
```

The project uses the pretrained:

```text
distilbert-base-uncased
```

model from Hugging Face Transformers and fine-tunes it using the customer review dataset.

The model is evaluated using:

* Accuracy
* Precision
* Recall
* Weighted F1-score

---

### 3.2 Named Entity Recognition

The system also performs Named Entity Recognition (NER).

NER identifies named entities appearing in customer reviews, such as:

* Products
* Organizations
* People
* Locations
* Other recognized entities

The NER pipeline is implemented using the Hugging Face Transformers pipeline API.

---

### 3.3 Topic Detection

The application uses zero-shot classification to identify topics discussed in a review.

The current candidate topics are:

```text
battery life
display
performance
price
build quality
customer service
```

The zero-shot classifier uses:

```text
facebook/bart-large-mnli
```

A review can therefore be analyzed for multiple relevant topics rather than being restricted to a single category.

---

## 4. System Architecture

The application follows the following processing pipeline:

```text
                    Customer Review
                           |
                           v
                    FastAPI Endpoint
                           |
                           v
                    ReviewEngine
                    /      |      \
                   /       |       \
                  v        v        v
          Sentiment       NER     Topic Detection
           DistilBERT              BART MNLI
              |           |            |
              +-----------+------------+
                          |
                          v
                   Structured JSON
                          |
                          v
                       Client
```

### Main Components

#### `train_model.py`

Responsible for:

1. Loading the review dataset.
2. Creating training, validation, and test sets.
3. Loading the DistilBERT tokenizer.
4. Tokenizing the reviews.
5. Loading the pretrained DistilBERT model.
6. Fine-tuning the model.
7. Calculating evaluation metrics.
8. Saving the trained model.

#### `analysis_engine.py`

Contains the `ReviewEngine` class.

It loads the trained sentiment model and combines it with:

* Sentiment analysis
* Named Entity Recognition
* Zero-shot topic detection

It returns the complete analysis for a review.

#### `main.py`

Defines the FastAPI application.

It provides:

```text
GET  /health
POST /analyse
```

The `/analyse` endpoint accepts a customer review and returns the NLP analysis.

---

## 5. Dataset

The project uses a local CSV dataset containing **1,200 customer-review examples**.

The dataset contains three sentiment classes:

```text
Negative    400
Neutral     400
Positive    400
```

This provides a balanced dataset for the three-class classification task.

### Dataset fields

| Field       | Description                      |
| ----------- | -------------------------------- |
| `review_id` | Unique identifier for the review |
| `review`    | Customer review text             |
| `sentiment` | Sentiment label as text          |
| `label`     | Numerical sentiment label        |
| `category`  | Product or service category      |

The numerical labels are:

```text
0 = Negative
1 = Neutral
2 = Positive
```

### Dataset Split

The dataset is divided into:

```text
Training:    960 examples
Validation:  120 examples
Testing:     120 examples
```

The split is stratified so that the three sentiment classes remain represented proportionally across the datasets.

> Note: The 1,200-row dataset currently used for model development consists of constructed training examples. It should not be presented as a collection of genuine customer reviews. The final requirement for testing at least 20 real examples is handled separately.

---

## 6. Technologies Used

### Programming Language

* Python 3.12

### NLP and Machine Learning

* Hugging Face Transformers
* Hugging Face Datasets
* PyTorch
* NumPy
* Scikit-learn

### NLP Models

* `distilbert-base-uncased`
* `facebook/bart-large-mnli`
* Hugging Face NER pipeline

### API

* FastAPI
* Uvicorn
* Pydantic

### Data

* Pandas
* CSV

---

## 7. Project Structure

```text
nlp_project/
│
├── customer_reviews_1200.csv
│
├── train_model.py
│
├── analysis_engine.py
│
├── main.py
│
├── sentiment_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── ...
│
├── venv/
│
└── README.md
```

The `sentiment_model/` directory is generated after training and contains the fine-tuned DistilBERT model.

---

## 8. Installation

Clone or copy the project and create a Python virtual environment.

### Create virtual environment

```bash
python3 -m venv venv
```

### Activate the environment

Linux/macOS:

```bash
source venv/bin/activate
```

Windows:

```powershell
venv\Scripts\activate
```

### Install dependencies

```bash
pip install pandas numpy scikit-learn transformers datasets torch accelerate fastapi uvicorn
```

---

## 9. Training the Model

Before starting the API, the sentiment model must be trained.

Run:

```bash
python3 train_model.py
```

The training process:

```text
Load CSV
   |
   v
Create train/validation/test splits
   |
   v
Load DistilBERT tokenizer
   |
   v
Tokenize reviews
   |
   v
Fine-tune DistilBERT
   |
   v
Evaluate model
   |
   v
Save model
```

After successful training, the model will be saved to:

```text
./sentiment_model/
```

The API uses this directory when loading the sentiment model.

---

## 10. Evaluation Metrics

The sentiment classifier is evaluated using four metrics.

### Accuracy

Measures the proportion of predictions that are correct.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Measures how many predictions for a particular class were actually correct.

### Recall

Measures how many examples belonging to a particular class were successfully identified.

### Weighted F1-score

Combines precision and recall while accounting for the number of examples in each class.

The training script reports:

```text
accuracy
precision
recall
f1
```

Final test results will be recorded here after model training:

```text
Accuracy:  [to be completed]
Precision: [to be completed]
Recall:    [to be completed]
F1-score:  [to be completed]
```

---

## 11. Running the API

Once the model has finished training and `sentiment_model/` has been created, start the FastAPI application:

```bash
uvicorn main:app --reload
```

The API will normally be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

---

## 12. API Endpoints

### Health Check

```http
GET /health
```

Example response:

```json
{
    "status": "ok"
}
```

---

### Analyze a Review

```http
POST /analyse
```

Request:

```json
{
    "text": "The battery life is excellent and the headphones sound great."
}
```

Example response structure:

```json
{
    "text": "The battery life is excellent and the headphones sound great.",
    "sentiment": "positive",
    "confidence": 0.95,
    "entities": [],
    "topics": [
        "battery life"
    ]
}
```

The exact sentiment, confidence, entities, and topics depend on the model's prediction.

---

### Evaluate Real Customer Reviews

```http
GET /evaluate-real-reviews
```

This endpoint evaluates the trained NLP application against **20 real customer reviews** stored in:

```text
real_customer_reviews_20.csv
```

For each review, the endpoint:

1. Loads the review and expected sentiment from the CSV file.
2. Sends the review through the NLP engine.
3. Retrieves the predicted sentiment, confidence, entities, and topics.
4. Compares the predicted sentiment with the expected sentiment.
5. Records whether the prediction was correct.
6. Calculates the overall evaluation accuracy.

The endpoint returns both an overall evaluation summary and the detailed results for each review.

Example response structure:

```json
{
    "total_reviews": 20,
    "correct_predictions": 17,
    "incorrect_predictions": 3,
    "accuracy": 0.85,
    "accuracy_percentage": 85.0,
    "results": [
        {
            "id": 1,
            "review": "The product works perfectly and I am very happy with it.",
            "expected": "positive",
            "predicted": "positive",
            "confidence": 0.94,
            "correct": true,
            "entities": [],
            "topics": [
                "product"
            ]
        }
    ]
}
```

The actual values returned depend on the model's predictions.

---

## 13. Testing the API

The API can be tested through the automatically generated Swagger interface.

Open:

```text
http://127.0.0.1:8000/docs
```

### Test Individual Reviews

Select:

```text
POST /analyse
```

Click:

```text
Try it out
```

Enter a review:

```json
{
    "text": "The laptop is fast and the battery lasts all day."
}
```

Click:

```text
Execute
```

The API will return the sentiment classification, confidence score, detected entities, and detected topics.

### Test Real Customer Reviews

Select:

```text
GET /evaluate-real-reviews
```

Click:

```text
Try it out
```

Click:

```text
Execute
```

The endpoint will load the reviews from:

```text
real_customer_reviews_20.csv
```

and evaluate the model against all 20 real customer reviews.

The response includes:

* Total number of reviews
* Correct predictions
* Incorrect predictions
* Overall accuracy
* Accuracy percentage
* Detailed results for each review

---

## 14. Real-World Evaluation

The application is evaluated using **20 real customer reviews** stored in:

```text
real_customer_reviews_20.csv
```

Each record contains an evaluation ID, the customer review, and its expected sentiment.

The reviews are passed through the same NLP analysis engine used by the `/analyse` endpoint. The predicted sentiment is then compared with the expected sentiment stored in the evaluation dataset.

For each review, the evaluation records:

* Review ID
* Review text
* Expected sentiment
* Predicted sentiment
* Confidence score
* Whether the prediction was correct
* Detected entities
* Detected topics

The evaluation endpoint is:

```http
GET /evaluate-real-reviews
```

### Accuracy Calculation

The overall accuracy is calculated using:

```text
Accuracy =
Correct Predictions / Total Reviews
```

The API also returns the accuracy as a percentage:

```text
Accuracy Percentage =
Correct Predictions / Total Reviews × 100
```

For example, if the model correctly classifies 17 out of 20 reviews:

```text
Accuracy = 17 / 20
         = 0.85

Accuracy Percentage = 85%
```

This evaluation provides a more realistic indication of how the NLP application performs on previously unseen, real-world customer reviews.

---


## 15. Key Findings

The final findings will be documented after training and real-example evaluation.

The analysis will consider:

1. Overall sentiment classification accuracy.
2. Precision, recall, and F1-score.
3. Which sentiment classes are most frequently confused.
4. Whether the model performs differently on short and long reviews.
5. The usefulness of NER for identifying entities in customer reviews.
6. Whether zero-shot topic detection correctly identifies review topics.
7. Limitations caused by the size and composition of the training dataset.

---

## 16. Limitations

Several limitations should be considered when interpreting the results.

### Dataset Size

The training dataset contains 1,200 constructed examples. This is considerably smaller and less diverse than a large production customer-review dataset.

### Synthetic Training Data

The training examples are constructed for this academic project. Consequently, they may not capture all of the language patterns, spelling variations, slang, ambiguity, and product-specific terminology found in genuine customer reviews.

### Domain Generalization

The sentiment model is trained specifically for customer-review style text. Its performance may differ when applied to other domains such as news articles, academic papers, or social media posts.

### Zero-Shot Topic Detection

Topic detection does not use a task-specific training dataset. It relies on a pretrained zero-shot classification model and a predefined list of candidate topics.

### NER

Named Entity Recognition is performed using a general pretrained NER model. Its entity extraction performance may vary depending on the product names and terminology appearing in customer reviews.

---

## 17. Future Improvements

Possible improvements include:

* Training on a larger collection of genuine customer reviews.
* Using domain-specific pretrained language models.
* Expanding the sentiment dataset with more diverse examples.
* Adding more product and service categories.
* Improving topic detection with supervised topic classification.
* Adding aspect-based sentiment analysis.
* Adding a Streamlit web interface.
* Storing analyzed reviews and predictions in a database.
* Adding automated API tests.
* Deploying the FastAPI application to a cloud server.

---

## 18. Assignment Requirement Summary

The completed system addresses the required components as follows:

### Domain

**Customer reviews**

### Sentiment Analysis

**DistilBERT fine-tuned for three-class sentiment classification**

```text
Negative
Neutral
Positive
```

### Additional NLP Capabilities

**Named Entity Recognition**

Identifies entities contained within customer reviews.

**Topic Detection**

Uses zero-shot classification to identify topics such as:

```text
Battery life
Display
Performance
Price
Build quality
Customer service
```

### Prediction Interface

**FastAPI REST API**

The primary prediction endpoint is:

```text
POST /analyse
```

### Model Evaluation

The project calculates:

```text
Accuracy
Precision
Recall
F1-score
```

A separate evaluation using at least 20 genuine customer reviews will provide the final real-world performance measurement.

### Documentation

The architecture, dataset, technologies, API, evaluation methodology, limitations, and future improvements are documented in this README.

---

## 19. Conclusion

This project demonstrates an end-to-end NLP application for customer review analysis.

The system combines a fine-tuned transformer-based sentiment classifier with Named Entity Recognition and zero-shot topic detection. The resulting NLP pipeline is exposed through a FastAPI REST interface, allowing customer reviews to be submitted and analyzed programmatically.

The project also demonstrates the complete machine-learning workflow from dataset preparation and model fine-tuning through evaluation and deployment as an API.
