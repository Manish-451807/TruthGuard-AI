# Models folder

Trained models are saved here by the scripts in `scripts/` (added in Part 4):
`*.joblib` pipelines and `*.json` evaluation reports.
They are git-ignored. Until you train, the app runs in demonstration mode.

Only load `.joblib` files that YOU created. Joblib files use pickle, which can
run code when loaded, so never load one downloaded from an unknown source.
