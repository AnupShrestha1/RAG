from datasets import load_dataset
import os

RAW_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")

def save_wikipeqa_raw():
    # Load dataset
    ds = load_dataset("teilomillet/wikipeqa", split="sample")  # use 'train' later for full dataset

    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH)

    print(f"Saving articles to {RAW_DATA_PATH}")

    for i, item in enumerate(ds):
        article_text = item["source"]  # correct field containing Wikipedia text
        filename = os.path.join(RAW_DATA_PATH, f"article_{i}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(article_text)

    print(f"Saved {len(ds)} articles.")


if __name__ == "__main__":
    save_wikipeqa_raw()
