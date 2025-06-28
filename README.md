# Collections Scoring Model

This project fine-tunes sentence transformer models to better score how well a category name matches a group of 4 words in the Collections game (inverse of NYT Connections). The model learns from the NYT Connections dataset to understand semantic relationships between category names and word groups.

## Installation

```bash
uv sync
```

## Usage

### Interactive Scoring

Test how well a category matches a group of words:

```bash
python main.py score
```

This will prompt you to enter a category and words, then output similarity scores.

### Single Score

Score a specific category/words combination:

```bash
python main.py score --group "dog breeds" --words "poodle, beagle, husky, terrier"
```

### Training

Fine-tune a new model:

```bash
python main.py train --base-model "Qwen/Qwen3-Embedding-0.6B"
```

### Generate Random Words

Get 4 random words for testing:

```bash
python main.py random-words
```

## How It Works

1. **Dataset Creation**: Converts NYT Connections data into positive/negative pairs
   - Positive: Correct category name + its 4 words
   - Negative: Correct category name + 4 random words from other groups

2. **Training**: Uses contrastive loss to learn semantic similarity between categories and word groups

3. **Scoring**: Computes embedding similarity between category names and word combinations, optionally normalized against random baselines

## Model

The default fine-tuned model is `samsartor/connections-categories-qwen3-0.6B`, based on Qwen3-Embedding-0.6B and trained on NYT Connections data.
