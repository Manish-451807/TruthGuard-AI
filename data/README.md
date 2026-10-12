# Data folder

| Path | What it is |
|------|------------|
| `sample_news.csv` | **Synthetic, demonstration-only** data (24 made-up statements about a fictional town). It exists so the app can start before you download the real dataset. Its labels are arbitrary and prove nothing. Never cite it as research data. |
| `raw/liar/` | You place the real LIAR files here (`train.tsv`, `valid.tsv`, `test.tsv`). Not included in this project. |
| `processed/` | Created by `scripts/prepare_dataset.py`. Cleaned CSVs plus `dataset_info.json`, which records whether the data is real or demo. |

`raw/` and `processed/` are git-ignored. See `scripts/download_instructions.md`.

## sample_news.csv columns
`id, statement, label, sentiment`
- `label`: one of pants-fire, false, barely-true, half-true, mostly-true, true
- `sentiment`: positive, negative or neutral (hand-assigned for the demo)
