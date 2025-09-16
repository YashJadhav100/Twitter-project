# Twitter Data Analysis Project

This repository contains code and outputs for analyzing Twitter data using machine learning models. The workflow covers preprocessing, prediction, summarization, and visualization.

---

## 📂 Repository Structure

```
twitter_data_starter/
│
├── 01_preprocess.py         # Script for cleaning and preprocessing raw JSON Twitter data
├── 02_train_model.py        # (Optional) Training script for fine-tuning classification models
├── 03_predict.py            # Script for running predictions on new Twitter data
├── 04_eval.py               # Evaluation script for classification outputs
├── 05_visualize.py          # Visualization of predictions and evaluation metrics
├── 06_summarize_preds.py    # Script to summarize predictions (counts, plots, examples)
│
├── preds_im1_20k_debug.csv  # Sample predictions on 20k tweets
├── summary_table.csv        # Emotion counts table (generated output)
├── summary_plot.png         # Bar chart visualization of predicted emotions
├── examples_by_label.txt    # Sample tweets per predicted emotion
│
├── requirements.txt         # Python dependencies
├── README.md                # Documentation (this file)
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/YashJadhav100/twitter_data_starter.git
   cd twitter_data_starter
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Mac/Linux
   .venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Running the Scripts

### 1. Preprocess Twitter JSON

Convert raw JSONL files into a clean DataFrame:

```bash
python 01_preprocess.py --input data/all_tweets.json --output clean_tweets.csv
```

### 2. Run Predictions

Classify emotions on preprocessed data:

```bash
python 03_predict.py --input clean_tweets.csv --output preds.csv
```

### 3. Summarize Predictions

Generate tables, plots, and example tweets by label:

```bash
python 06_summarize_preds.py --preds preds.csv --per_label 2
```

This produces:

* `summary_table.csv` → Counts of predicted emotions
* `summary_plot.png` → Bar chart of predictions
* `examples_by_label.txt` → Example tweets per label

---

## 📊 Key Outputs

* **`preds_im1_20k_debug.csv`** → Predictions on \~20k tweets (labels + probabilities)
* **`summary_table.csv`** → Counts of each predicted emotion
* **`summary_plot.png`** → Bar chart of predicted emotions
* **`examples_by_label.txt`** → Two sample tweets per predicted emotion

---

## 📥 Access

* The full **Twitter Project folder** (including scripts and outputs) has been uploaded to Google Drive:
  👉 https://drive.google.com/drive/folders/1PCAOvHJjCDrKwPNgqPzoz8ASFHwQbPOi?usp=drive_link

---
