import os, re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spacy
from tqdm import tqdm
from datetime import datetime
from nltk.corpus import stopwords
from transformers import pipeline
from wordcloud import WordCloud
from analysis.text_utils import mask_swearwords


# === GLOBALS ===
stopwords_set = set(stopwords.words("english"))
excluded_words = {"http", "https", "www", "com", "score", "rank", "ps3", "x", "1", "2", "3", "85"}
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])
    return _nlp

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\W", " ", text)
    doc = get_nlp()(text)
    return " ".join([t.text for t in doc if t.text not in stopwords_set and t.text not in excluded_words])

def get_sentiments(texts):
    sentiments = []
    for i in tqdm(range(0, len(texts), 32), desc="🔍 BERT Sentiment"):
        batch = texts[i:i+32]
        results = sentiment_pipeline(batch, truncation=True, max_length=512)
        for res in results:
            label = res["label"]
            if label in ["5 stars", "4 stars"]:
                sentiments.append("POSITIVE")
            elif label in ["1 star", "2 stars"]:
                sentiments.append("NEGATIVE")
            else:
                sentiments.append("NEUTRAL")
    return sentiments

def is_likely_sarcastic(text, sentiment):
    markers = ["sure", "obviously", "as if", "yeah right", "totally", "...", "🙄", "brilliant", "great job"]
    return sentiment == "POSITIVE" and any(m in text.lower() for m in markers)

def make_wordcloud(texts, cmap):
    blob = " ".join(texts)
    return WordCloud(width=800, height=400, background_color="white", colormap=cmap).generate(blob)

def analyze_excel_file(file_path, output_dir="static/results"):
    os.makedirs(output_dir, exist_ok=True)
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".xlsx":
        df = pd.read_excel(file_path)
    elif ext == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload .xlsx or .csv")

    df.columns = df.columns.str.strip().str.lower()
    if "comment" not in df.columns:
        raise ValueError("Input file must contain a 'comment' column.")

    texts_raw = df["comment"].astype(str).tolist()
    texts_cleaned = [clean_text(t) for t in texts_raw]
    sentiments = get_sentiments(texts_cleaned)
    sarcasm_flags = [is_likely_sarcastic(t, s) for t, s in zip(texts_cleaned, sentiments)]

    pos = [mask_swearwords(t) for t, s in zip(texts_cleaned, sentiments) if s == "POSITIVE"]
    neg = [mask_swearwords(t) for t, s in zip(texts_cleaned, sentiments) if s == "NEGATIVE"]
    neu = [mask_swearwords(t) for t, s in zip(texts_cleaned, sentiments) if s == "NEUTRAL"]

    base = os.path.splitext(os.path.basename(file_path))[0]
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    label = f"{base}_{timestamp_str}"

    has_time = "timestamp" in df.columns
    df_time = pd.DataFrame()
    time_bins = pd.DataFrame()

    if has_time:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", dayfirst=True)
        df_time = pd.DataFrame({"timestamp": df["timestamp"], "sentiment": sentiments}).dropna()
        df_time.set_index("timestamp", inplace=True)
        time_bins = df_time.groupby([pd.Grouper(freq="30min"), "sentiment"]).size().unstack(fill_value=0)

    # === Plotting ===
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"Sentiment Summary: {base}\n{timestamp_str}", fontsize=16, y=0.99)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2])

    if has_time and not df_time.empty:
        ax0 = fig.add_subplot(gs[0, 0])
        time_bins.plot(ax=ax0, linewidth=2, color={"POSITIVE": "green", "NEGATIVE": "red", "NEUTRAL": "gray"})
        ax0.set_title("Sentiment Over Time")
        ax0.set_xlabel("Time")
        ax0.set_ylabel("Comment Count")
        ax0.legend(title="Sentiment")
    else:
        sentiment_series = pd.Series(sentiments)
        ax0 = fig.add_subplot(gs[0, 0])
        sentiment_series.value_counts().plot.bar(
            ax=ax0,
            color=["green", "red", "gray"],
            edgecolor="black"
        )
        ax0.set_title("Sentiment Distribution")
        ax0.set_xlabel("Sentiment")
        ax0.set_ylabel("Number of Comments")

    ax_pie = fig.add_subplot(gs[0, 1])
    pie_counts = [len(pos), len(neg), len(neu)]
    ax_pie.pie(pie_counts, labels=["Positive", "Negative", "Neutral"], colors=["green", "red", "gray"],
               autopct="%1.1f%%", startangle=90)
    ax_pie.set_title("Overall Sentiment")

    if pos:
        ax1 = fig.add_subplot(gs[1, 0])
        ax1.imshow(make_wordcloud(pos, "Greens"), interpolation="bilinear")
        ax1.axis("off")
        ax1.set_title("Positive Word Cloud")

    if neg:
        ax2 = fig.add_subplot(gs[1, 1])
        ax2.imshow(make_wordcloud(neg, "Reds"), interpolation="bilinear")
        ax2.axis("off")
        ax2.set_title("Negative Word Cloud")

    summary_path = os.path.join(output_dir, f"{label}_Summary.png")
    fig.savefig(summary_path, bbox_inches="tight")
    plt.close(fig)

    df["clean_text"] = texts_cleaned
    df["predicted_sentiment"] = sentiments
    df["sarcastic_flag"] = sarcasm_flags
    df.to_excel(f"data/{label}_labeled.xlsx", index=False)

    stats = {
        "total_comments": len(texts_cleaned),
        "positive": len(pos),
        "negative": len(neg),
        "neutral": len(neu),
        "sarcastic": sum(sarcasm_flags)
    }

    return summary_path, stats, sentiments
