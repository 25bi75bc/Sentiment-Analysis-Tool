def get_model_evaluation():
    import pandas as pd
    from transformers import pipeline
    from sklearn.metrics import classification_report, confusion_matrix

    # Manually labeled comments
    data = pd.DataFrame({
        "comment": [
            "That movie was absolutely incredible.",
            "Worst sequel I've ever seen.",
            "I haven't watched it yet, but it's on my list.",
            "Blade 2 blew my mind! Loved the action scenes.",
            "Meh. I felt nothing about the plot.",
            "What a complete waste of time.",
            "Honestly, it was fine. Nothing amazing.",
            "Ugh. So boring.",
            "Not bad! I might rewatch it sometime.",
            "I don’t know what to think about this one."
        ],
        "label": [
            "POSITIVE", "NEGATIVE", "NEUTRAL", "POSITIVE", "NEUTRAL",
            "NEGATIVE", "NEUTRAL", "NEGATIVE", "POSITIVE", "NEUTRAL"
        ]
    })

    sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

    def convert_label(star_label):
        if star_label in ["5 stars", "4 stars"]:
            return "POSITIVE"
        elif star_label in ["1 star", "2 stars"]:
            return "NEGATIVE"
        return "NEUTRAL"

    comments = list(data["comment"].astype(str))
    predictions = sentiment_model(comments)
    data["predicted"] = [convert_label(r["label"]) for r in predictions]

    report = classification_report(data["label"], data["predicted"], output_dict=True, zero_division=0)
    matrix = confusion_matrix(data["label"], data["predicted"], labels=["NEGATIVE", "NEUTRAL", "POSITIVE"])

    return {
        "accuracy": round(report["accuracy"] * 100, 2),
        "macro_avg": {k: round(v * 100, 1) for k, v in report["macro avg"].items()},
        "matrix": matrix.tolist(),
        "labels": ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    }
