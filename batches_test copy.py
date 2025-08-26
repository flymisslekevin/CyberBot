import json

# Load results from batch_results.json
with open("batch_results_5.json", "r", encoding="utf-8") as f:
    results = json.load(f)

print(len(results))

unknown = []

for key, value in results.items():
    if "the context does not" in value.lower() or "the passage does not" in value.lower() or "i don't know" in value.lower():
        unknown.append(key)

print(f"Number of unknown questions: {len(unknown)}")
print(f"Unknown questions: {unknown}")

print(f"Accuracy: {100 - (len(unknown) / len(results) * 100):.2f}%")
