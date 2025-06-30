import json

# Path to your messy JSON file
with open("data/comments_to_label.json", "r", encoding="utf-8") as f:
    raw = f.read()

# Extract lines that look like valid comment objects
lines = [line.strip().rstrip(',') for line in raw.splitlines() if line.strip().startswith('{')]

# Parse each line as a standalone JSON object
objects = [json.loads(line) for line in lines]

# Save cleaned JSON array
with open("data/comments_to_label.json", "w", encoding="utf-8") as f:
    json.dump(objects, f, indent=2)

print("✅ comments_to_label.json has been cleaned and fixed.")
