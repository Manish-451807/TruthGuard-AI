# Getting the LIAR dataset

LIAR is a public benchmark of short political statements labeled by
PolitiFact fact-checkers. It was introduced in:

> William Yang Wang, "Liar, Liar Pants on Fire": A New Benchmark Dataset for
> Fake News Detection, ACL 2017. Paper: https://arxiv.org/abs/1705.00648

**This project does not include or redistribute the dataset.**

## Download
1. Open the paper page above and follow its link to the dataset. The author
   has hosted it as `liar_dataset.zip` on his university page
   (https://www.cs.ucsb.edu/~william/data/liar_dataset.zip).
   I could not verify that link from my environment, so if it fails, search
   for "LIAR dataset William Yang Wang" and use the author's official page.
2. Check the dataset's terms of use on that page before using it.
3. Unzip and place these files in `data/raw/liar/`:
   - `train.tsv`
   - `valid.tsv`
   - `test.tsv`
   (To use another folder, set `LIAR_DIR` in `.env`.)

## Expected format
Tab-separated, NO header row, 14 columns:

| # | Column |
|---|--------|
| 1 | id |
| 2 | label |
| 3 | statement |
| 4 | subject |
| 5 | speaker |
| 6 | speaker_job |
| 7 | state |
| 8 | party |
| 9-13 | barely_true / false / half_true / mostly_true / pants_on_fire counts |
| 14 | context |

Labels: `pants-fire`, `false`, `barely-true`, `half-true`, `mostly-true`, `true`.

## What they mean (and what we do with them)
These are fact-checker ratings of the original statements. By default we keep
all SIX labels. `LABEL_MODE=grouped_two_class` merges them into two groups
(pants-fire/false/barely-true vs half-true/mostly-true/true). That grouping is
our own modelling choice, never merged silently, and recorded in
`data/processed/dataset_info.json`.

Our model learns word patterns linked to those ratings. It does NOT verify
whether any new claim is true.

## Then run
```
python scripts/prepare_dataset.py
```
No download yet? Use the demo fallback: `python scripts/prepare_dataset.py --use-sample`

---

# Sentiment dataset (TweetEval)

The credibility data (LIAR) has no sentiment labels, so the sentiment model
uses a separate public dataset: the **sentiment** task of TweetEval
(Barbieri et al., Findings of EMNLP 2020, https://arxiv.org/abs/2010.12421).
It contains social media posts labeled negative, neutral or positive.

1. Open the project repository: https://github.com/cardiffnlp/tweeteval
   and find the `datasets/sentiment/` folder. (I could not open it from my
   environment, so check the folder layout and the license on that page.)
2. Download these six files into `data/raw/tweeteval/sentiment/`:
   `train_text.txt`, `train_labels.txt`, `val_text.txt`, `val_labels.txt`,
   `test_text.txt`, `test_labels.txt`
3. Label codes: `0` = negative, `1` = neutral, `2` = positive.
4. Train: `python scripts/train_sentiment_model.py`
   (To use another folder: `--data-dir path\to\folder`.)

No download yet? `python scripts/train_sentiment_model.py --use-sample` trains
on the 24 synthetic demo rows. Those results are NOT meaningful.
