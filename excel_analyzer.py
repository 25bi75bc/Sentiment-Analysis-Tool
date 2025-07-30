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


def analyze_excel_file(file_path, output_dir="static/results"):
    os.makedirs(output_dir, exist_ok=True)
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".xlsx":
        df = pd.read_excel(file_path)
    elif ext == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload .xlsx or .csv")

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

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

    # === Time Parsing Block ===
    has_time = "timestamp" in df.columns
    df_time = pd.DataFrame()
    time_bins = pd.DataFrame()

    if has_time:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", dayfirst=True)
        valid_timestamps = df["timestamp"].notna().sum()
        print(f"✅ Timestamp column detected, {valid_timestamps} valid entries")

        df_time = pd.DataFrame({
            "timestamp": df["timestamp"],
            "sentiment": sentiments
        }).dropna()

        if not df_time.empty:
            df_time.set_index("timestamp", inplace=True)
            time_bins = df_time.groupby([pd.Grouper(freq="30min"), "sentiment"]).size().unstack(fill_value=0)
            print("📊 Time bins constructed:\n", time_bins.head())
        else:
            print("⚠️ All timestamps were invalid — falling back to distribution chart")
    else:
        print("⛔ No 'timestamp' column detected — fallback activated")

    # === Plotting ===
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"Sentiment Summary: {base}\n{timestamp_str}", fontsize=16, y=0.99)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2])

    if has_time and not df_time.empty and not time_bins.empty:
        print("📈 Plotting Sentiment Over Time")
        ax0 = fig.add_subplot(gs[0, 0])
        time_bins.plot(ax=ax0, linewidth=2, color={"POSITIVE": "green", "NEGATIVE": "red", "NEUTRAL": "gray"})
        ax0.set_title("Sentiment Over Time")
        ax0.set_xlabel("Time")
        ax0.set_ylabel("Comment Count")
        ax0.legend(title="Sentiment")
    else:
        print("📉 Plotting Fallback Sentiment Distribution")
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

    # === Pie and Word Clouds ===
    ax_pie = fig.add_subplot(gs[0, 1])
    pie_counts = [len(pos), len(neg), len(neu)]
    ax_pie.pie(pie_counts, labels=["Positive", "Negative", "Neutral"],
               colors=["green", "red", "gray"], autopct="%1.1f%%", startangle=90)
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

    # === Save Annotated Excel ===
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


