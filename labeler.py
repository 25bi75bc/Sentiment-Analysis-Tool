from flask import Flask, render_template, request, redirect, url_for
import csv
import os
import json

app = Flask(__name__)

COMMENTS_FILE = "data/comments_to_label.json"
OUTPUT_FILE = "data/true_labels.csv"
os.makedirs("data", exist_ok=True)

# Load comments once (they won't change during session)
with open(COMMENTS_FILE, "r", encoding="utf-8") as f:
    comments = json.load(f)

@app.route("/")
def index():
    labeled_indices = set()

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row:
                    labeled_indices.add(row[2])  # Use ID column to track

    for i, entry in enumerate(comments):
        if entry["id"] not in labeled_indices:
            return render_template("label.html", index=i, comment=entry["text"])
    
    return "<h3>✅ All comments have been labeled.</h3>"

@app.route("/label", methods=["POST"])
def label():
    index = int(request.form["index"])
    label = request.form["label"]
    comment = comments[index]

    # Append label to CSV with ID for tracking
    with open(OUTPUT_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["comment", "true_label", "id"])
        writer.writerow([comment["text"], label, comment["id"]])

    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
