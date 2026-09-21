from transformers import pipeline

class ReviewEngine:
    def __init__(self, model_path="./sentiment_model"):

        # ======== 1. sentiment analysis =========

        self.sentiment = pipeline(
            "text-classification",
            model = model_path,
            tokenizer = model_path
        )


        # ======== 2. Named Entity Recognition (NER) =========

        self.ner = pipeline(
            "ner",
            aggregation_strategy = "simple"
        )

        # ======== 3. Zero-shot topic classification =========

        self.zero_shot = pipeline(
            "zero-shot-classification",
            model = "facebook/bart-large-mnli"
        )

        #topics the application understands
        self.topics = [ 
            "battery life", 
            "display", 
            "performance", 
            "price", 
            "build quality", 
            "customer service" 
            ]

    def analyse(self, text):

        # Sentiment
        sentiment_result = self.sentiment(text)[0]

        # Named Entity Recognition
        entity_results = self.ner(text)

        entities = [
            entity["word"]
            for entities in entity_results
        ]

        # Topic detection
        topic_result = self.zero_shot(
            text,
            candidate_labels = self.topics,
            multi_label = True
        )

        topics = [
            label
            for label, score in zip(
                topic_result["labels"],
                topic_result["scores"]
            )
            if score > 0.4
        ]

        # Return combined result
        return{
            "text": text,
            "sentiment": sentiment_result["label"],
            "confidence": round(
                sentiment_result["score"], 3
            ),
            "entities": list(set(entities)),
            "topics": topics[:3]
        }

# create engine instance

engine = ReviewEngine()