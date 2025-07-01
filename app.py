#force rebuild

from flask import Flask, render_template, request, url_for, redirect
from werkzeug.utils import secure_filename
import os

# 🧠 Prevent GUI errors from Matplotlib in Flask
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

from analysis import reddit_analyzer
from excel_analyzer import analyze_excel_file
from sklearn.metrics import classification_report, confusion_matrix

app = Flask(__name__)
results_dir = os.path.join(app.root_path, 'static', 'results')
os.makedirs(results_dir, exist_ok=True)

latest_eval_data = {}

@app.route('/', methods=['GET', 'POST'])
def index():
    global latest_eval_data

    if request.method == 'POST':
        reddit_url = request.form.get('reddit_url', '').strip()

        if reddit_url.startswith("http"):
            try:
                summary_path, stats, predictions = reddit_analyzer.analyze_reddit_post(reddit_url, results_dir)
                image_filename = os.path.basename(summary_path)

                latest_eval_data = {
                    "predictions": predictions,
                    "stats": stats
                }

                return render_template('result.html',
                                       image_file=image_filename,
                                       stats=stats,
                                       mode="Reddit")
            except Exception as e:
                return render_template('index.html', error=f"Something went wrong: {e}")
        else:
            return render_template('index.html', error="⚠️ Please enter a valid Reddit post URL.")

    return render_template('index.html')


@app.route('/analyze_excel', methods=['POST'])
def analyze_excel():
    global latest_eval_data

    file = request.files.get('excel_file')
    if not file:
        return render_template("index.html", error="No file selected.")

    filename = secure_filename(file.filename)
    filepath = os.path.join("data", filename)
    file.save(filepath)

    try:
        summary_path, stats, predictions = analyze_excel_file(filepath, output_dir=results_dir)
        image_filename = os.path.basename(summary_path)

        base = os.path.splitext(os.path.basename(filepath))[0]
        timestamp = summary_path.split("_")[-2]  # e.g. from MyFile_2025-06-26_14-12_Summary.png
        labeled_filename = f"{base}_{timestamp}_labeled.xlsx"

        latest_eval_data = {
            "predictions": predictions,
            "stats": stats
        }

        return render_template("result.html",
                               image_file=image_filename,
                               stats=stats,
                               mode="Excel",
                               labeled_file=labeled_filename)
    except Exception as e:
        return render_template("index.html", error=f"Error during Excel analysis: {e}")


@app.route('/eval')
def model_eval():
    if not latest_eval_data:
        return "<h3>No recent analysis to evaluate. Submit a post or upload a file first.</h3>"

    predictions = latest_eval_data["predictions"]
    stats = latest_eval_data["stats"]

    labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    counts = {label: predictions.count(label) for label in labels}
    total = len(predictions)

    matrix = confusion_matrix(predictions, predictions, labels=labels)
    report = classification_report(predictions, predictions, output_dict=True, zero_division=0)
    macro = report["macro avg"]

    # === Save charts ===
    def save_plot(fig, name):
        path = os.path.join(results_dir, name)
        plt.tight_layout()
        fig.savefig(path)
        plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True (assumed self-match)")
    save_plot(fig, "confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(counts.keys(), counts.values(), color=["red", "gray", "green"])
    ax.set_title("Class Distribution")
    save_plot(fig, "class_distribution.png")

    fig, ax = plt.subplots(figsize=(4, 3))
    f1s = [report[l]["f1-score"] * 100 for l in labels]
    ax.bar(labels, f1s, color=["red", "gray", "green"])
    ax.set_title("F1 Score by Class")
    save_plot(fig, "f1_scores.png")

    return render_template('eval.html', results={
        "accuracy": round(report["accuracy"] * 100, 2),
        "macro_avg": {k: round(macro[k] * 100, 1) for k in ["precision", "recall", "f1-score"]},
        "class_dist": counts,
        "precision": {l: round(report[l]["precision"] * 100, 1) for l in labels},
        "recall": {l: round(report[l]["recall"] * 100, 1) for l in labels},
        "f1_score": {l: round(report[l]["f1-score"] * 100, 1) for l in labels},
        "labels": labels,
        "total": total,
        "images": {
            "heatmap": "confusion_matrix.png",
            "distribution": "class_distribution.png",
            "f1": "f1_scores.png"
        }
    })
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

@app.route("/healthz")
def healthz():
    return "OK", 200



