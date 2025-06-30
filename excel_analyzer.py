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

# === NLP Setup ===
nlp = spacy.load("en_core_web_sm")
stopwords_set = set(stopwords.words("english"))
excluded_words = {"http", "https", "www", "com", "score", "rank", "ps3", "x", "1", "2", "3", "85"}
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\W", " ", text)
    doc = nlp(text)
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

    if "comment" not in df.columns:
        raise ValueError("Input file must contain a 'comment' column.")

    texts_raw = df["comment"].astype(str).tolist()
    texts_cleaned = [clean_text(t) for t in texts_raw]
    sentiments = get_sentiments(texts_cleaned)
    sarcasm_flags = [is_likely_sarcastic(t, s) for t, s in zip(texts_cleaned, sentiments)]

    pos = [t for t, s in zip(texts_cleaned, sentiments) if s == "POSITIVE"]
    neg = [t for t, s in zip(texts_cleaned, sentiments) if s == "NEGATIVE"]
    neu = [t for t, s in zip(texts_cleaned, sentiments) if s == "NEUTRAL"]

    base = os.path.splitext(os.path.basename(file_path))[0]
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    label = f"{base}_{timestamp_str}"

    # === Determine if timestamps exist
    has_time = "timestamp" in df.columns
    if has_time:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df_time = pd.DataFrame({"timestamp": df["timestamp"], "sentiment": sentiments})
        df_time.dropna(inplace=True)
        df_time.set_index("timestamp", inplace=True)
        time_bins = df_time.groupby([pd.Grouper(freq="30min"), "sentiment"]).size().unstack(fill_value=0)

    # === Plot ===
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"Sentiment Summary: {base}\n{timestamp_str}", fontsize=16, y=0.99)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2])

    # Top-Left Plot: Time-Series or Fallback Histogram
    if has_time and not df_time.empty:
        ax0 = fig.add_subplot(gs[0, 0])
        time_bins.plot(ax=ax0, linewidth=2, color={"POSITIVE": "green", "NEGATIVE": "red", "NEUTRAL": "gray"})
        ax0.set_title("Sentiment Over Time")
        ax0.set_xlabel("Time")
        ax0.set_ylabel("Comment Count")
        ax0.legend(title="Sentiment")
    else:
        lengths = [len(t.split()) for t in texts_raw]
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.hist(lengths, bins=20, color="skyblue", edgecolor="gray")
        ax0.set_title("Comment Length Distribution")
        ax0.set_xlabel("Words per Comment")
        ax0.set_ylabel("Frequency")

    # Top-Right: Pie Chart
    pie_counts = [len(pos), len(neg), len(neu)]
    ax_pie = fig.add_subplot(gs[0, 1])
    ax_pie.pie(pie_counts, labels=["Positive", "Negative", "Neutral"],
               autopct="%1.1f%%", colors=["green", "red", "gray"])
    ax_pie.set_title("Overall Sentiment")

    # Bottom: Word Clouds
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

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    summary_path = os.path.join(output_dir, f"{label}_Summary.png")
    plt.savefig(summary_path)
    plt.close()

    df["clean_text"] = texts_cleaned
    df["predicted_sentiment"] = sentiments
    df["sarcastic_flag"] = sarcasm_flags

    df.to_excel(f"data/{label}_labeled.xlsx", index=False)  # Still saved locally for dev access

    stats = {
        "total_comments": len(texts_cleaned),
        "positive": len(pos),
        "negative": len(neg),
        "neutral": len(neu),
        "sarcastic": sum(sarcasm_flags)
    }

    return summary_path, stats, sentiments
