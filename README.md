# Stylometric Analysis for Insider Threat Detection — ML + Flask

This package contains two separately trained traditional-ML pipelines and a Flask dashboard. The included training dataset is **synthetic** (`data/emails_labeled_SYNTHETIC.csv`, 12,000 rows); it is not a verified CERT R4.2 email corpus.

## Models

- **Temporal analysis:** class-weighted Logistic Regression and a normal-only Isolation Forest. Inputs include hour/day/weekend/after-hours indicators and activity/message metadata. The primary supervised model intentionally excludes `external_recipient` and `personal_email_recipient` because those fields almost perfectly encode the label in this synthetic dataset.
- **Non-temporal stylometry:** subject/content word and character TF-IDF plus hand-crafted linguistic features, with class-weighted Logistic Regression. Recipient metadata, `label`, `split`, and `threat_scenario` are not model inputs.
- **Anomaly detectors:** Isolation Forest models trained on the normal class only. Their scores are uncalibrated anomaly rankings, not probabilities.

## Run the dashboard

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Windows cmd: .venv\Scripts\activate.bat
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000.

## Retrain

```bash
python train.py --csv data/emails_labeled_SYNTHETIC.csv
```

The script normalizes `normal`/`suspicious` and `0`/`1` labels, then creates train/validation/test partitions using grouped splitting by normalized email body. Repeated body content is kept in a single partition to reduce duplicate leakage. The CSV's original `split` column is reported as a diagnostic only; it is not used for model training or final metrics.

## Current development results

See `artifacts/metrics.json` for full metrics and split counts. The held-out grouped test partition contains 1,799 rows. Current scores:

- Temporal supervised model: precision 0.794, recall 0.786, F1 0.790, PR-AUC 0.886.
- Non-temporal stylometry model: precision/recall/F1 and PR-AUC all 1.000.

**Critical interpretation:** the perfect stylometry score is a red flag, not proof of real-world performance. Synthetic suspicious messages appear to have highly distinctive, label-correlated text patterns. Inspect `generate_synthetic.py`, templates, phrases, and the split before using these results in a report. The split prevents identical normalized bodies from crossing partitions, but it cannot prevent broader template/scenario leakage. Use real, appropriately labeled data and preferably user/time-separated evaluation before claiming real-world effectiveness.

## Dataset and ground-truth limitations

- The included 12,000-row dataset is synthetic and balanced at 6,000 normal / 6,000 suspicious.
- `threat_scenario` is metadata only and must not be passed as a feature.
- The CERT R4.2 logs have not been joined to these synthetic email records. R4.2 log ingestion remains a separate workflow; actual logs and defensible ground-truth mapping are required before claiming the temporal model uses CERT R4.2.
- Risk scores are research indicators, not proof of malicious intent. Require human review.
