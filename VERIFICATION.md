# Verification — 3 October 2026

- `python -m unittest -v`: **5 tests passed**.
- `python classifier.py`: holdout evaluation completed; 18 train and 6 test rows.
- `python classifier.py --text "مجھے مدد چاہیے"`: executed, predicted `help`.
- Data is hand-authored **synthetic demo data**; no external research dataset or benchmark has been evaluated. The tiny holdout result is not evidence of generalization.
- No Transformers or BERT model is implemented in this repository.
