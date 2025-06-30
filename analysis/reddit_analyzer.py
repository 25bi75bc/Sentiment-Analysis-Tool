import praw, spacy, re, os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from wordcloud import WordCloud
from nltk.corpus import stopwords
from tqdm import tqdm
from datetime import datetime

# === REDDIT SETUP ===
reddit = praw.Reddit(
    client_id="FFnd2gd4-aD-qfRunLhWDw",
    client_secret="jBvwcD_jjkYuuOa-FY8GdgPDH54XcQ",
    user_agent="sentimentality by u/Natural_Mix6280"
)

# === NLP & SENTIMENT SETUP ===
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
    sarcasm_markers = ["sure", "obviously", "as if", "yeah right", "totally", "...", "🙄", "brilliant", "great job"]
    return sentiment == "POSITIVE" and any(mark in text.lower() for mark in sarcasm_markers)

def make_wordcloud(texts, cmap):
    blob = " ".join(texts)
    return WordCloud(width=800, height=400, background_color="white", colormap=cmap).generate(blob)

def analyze_reddit_post(post_url, output_dir):
    submission = reddit.submission(url=post_url)
    subreddit_name = submission.subreddit.display_name
    submission.comments.replace_more(limit=0)
    comments_raw = [
        (comment.body, datetime.utcfromtimestamp(comment.created_utc))
        for comment in submission.comments if isinstance(comment.body, str)
    ]

    texts_cleaned = [clean_text(c) for c, _ in comments_raw]
    timestamps = [t for _, t in comments_raw]
    sentiments = get_sentiments(texts_cleaned)
    sarcasm_flags = [is_likely_sarcastic(text, sentiment) for text, sentiment in zip(texts_cleaned, sentiments)]

    pos = [t for t, s in zip(texts_cleaned, sentiments) if s == "POSITIVE"]
    neg = [t for t, s in zip(texts_cleaned, sentiments) if s == "NEGATIVE"]
    neu = [t for t, s in zip(texts_cleaned, sentiments) if s == "NEUTRAL"]

    df_time = pd.DataFrame({"timestamp": timestamps, "sentiment": sentiments})
    df_time["timestamp"] = pd.to_datetime(df_time["timestamp"])
    df_time.set_index("timestamp", inplace=True)
    time_bins = df_time.groupby([pd.Grouper(freq="30min"), "sentiment"]).size().unstack(fill_value=0)

    title_clean = re.sub(r'[\\/*?:"<>|]', "", submission.title).replace(" ", "_")[:50]
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    base = f"{title_clean}_{timestamp_str}"

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"r/{subreddit_name} | {submission.title}\n{timestamp_str}", fontsize=16, y=0.99)

    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2])

    ax0 = fig.add_subplot(gs[0, 0])
    time_bins.plot(ax=ax0, linewidth=2, color={"POSITIVE": "green", "NEGATIVE": "red", "NEUTRAL": "gray"})
    ax0.set_title("Sentiment Over Time")
    ax0.set_ylabel("Comment Count")
    ax0.set_xlabel("Time")
    ax0.legend(title="Sentiment")

    ax_pie = fig.add_subplot(gs[0, 1])
    pie_counts = [len(pos), len(neg), len(neu)]
    ax_pie.pie(pie_counts, labels=["Positive", "Negative", "Neutral"], colors=["green", "red", "gray"], autopct="%1.1f%%", startangle=90)
    ax_pie.set_title("Overall Sentiment Proportion")

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
    summary_path = os.path.join(output_dir, f"{base}_Summary.png")
    plt.savefig(summary_path)
    plt.close()

    stats = {
        "total_comments": len(texts_cleaned),
        "positive": len(pos),
        "negative": len(neg),
        "neutral": len(neu),
        "sarcastic": sum(sarcasm_flags)
    }

    return summary_path, stats, sentiments

