from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, flash

# These helpers belong to the existing non-temporal/stylometry training pipeline.
from train_np import stylometric_frame, text_join

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
app = Flask(__name__)
app.secret_key = "change-this-development-secret-key"


def load_artifact(name):
    path = ART / name
    return joblib.load(path) if path.exists() else None


def probability_label(score, threshold=0.5):
    suspicious = score >= threshold
    return (
        "Suspicious-pattern indicator" if suspicious else "No strong suspicious-pattern signal",
        "high" if suspicious else "low",
    )


def _int_value(row, key, default=0):
    value = row.get(key, default)
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def temporal_feature_frame(row, feature_columns):
    """Build exactly the feature schema expected by the saved temporal model.

    Single-email inference cannot calculate genuine rolling history by itself.
    History features may be supplied in the form/request row; otherwise they
    default to zero and are disclosed in the result.
    """
    timestamp = pd.to_datetime(row.get("timestamp"), errors="coerce")
    if pd.isna(timestamp):
        timestamp = pd.Timestamp.now()

    hour = _int_value(row, "hour", int(timestamp.hour))
    values = {
        "hour": hour,
        "day_of_week_num": int(timestamp.day_of_week),
        "is_weekend": int(timestamp.day_of_week >= 5),
        "is_after_hours": int(hour < 8 or hour >= 18),
        "recipient_count": _int_value(row, "recipient_count", 1),
        "external_recipient": _int_value(row, "external_recipient", 0),
        "attachment_count": _int_value(row, "attachment_count", 0),
        "size": _int_value(row, "size", 0),
        "word_count": _int_value(row, "word_count", 0),
        "char_count": _int_value(row, "char_count", 0),
        "sentence_count": _int_value(row, "sentence_count", 0),
        "activity_frequency_1h": _int_value(row, "activity_frequency_1h", 0),
        "activity_frequency_24h": _int_value(row, "activity_frequency_24h", 0),
        "unique_recipients_24h": _int_value(row, "unique_recipients_24h", 0),
        "off_hours_events_7d": _int_value(row, "off_hours_events_7d", 0),
        "external_recipient_events_7d": _int_value(row, "external_recipient_events_7d", 0),
    }
    frame = pd.DataFrame([values])
    for column in feature_columns:
        if column not in frame.columns:
            frame[column] = 0
    return frame[list(feature_columns)], timestamp


def run_temporal(row):
    artifact = load_artifact("temporal_model.joblib")
    if artifact is None:
        raise RuntimeError("Temporal model not found. Run train_temporal.py first.")

    feature_columns = artifact["feature_columns"]
    X, timestamp = temporal_feature_frame(row, feature_columns)

    score = float(artifact["model"].predict_proba(X)[0, 1])
    threshold = float(artifact.get("threshold", 0.5))
    label, level = probability_label(score, threshold)

    # The Isolation Forest was trained on imputed feature arrays. Reuse its
    # saved imputer and use the same score definition as training.
    anomaly_X = artifact["anomaly_imputer"].transform(X)
    anomaly_score = float((-artifact["anomaly_model"].score_samples(anomaly_X))[0])
    anomaly_threshold = float(artifact.get("anomaly_threshold", float("inf")))
    anomaly_flag = anomaly_score >= anomaly_threshold

    history_columns = [
        "activity_frequency_1h", "activity_frequency_24h",
        "unique_recipients_24h", "off_hours_events_7d",
        "external_recipient_events_7d",
    ]
    history_missing = not any(row.get(column) not in (None, "") for column in history_columns)

    details = [
        f"Activity timestamp: {timestamp.isoformat()}",
        f"Activity hour: {int(X.iloc[0]['hour'])}",
        f"External-recipient indicator: {int(X.iloc[0]['external_recipient'])}",
        f"After-hours indicator: {int(X.iloc[0]['is_after_hours'])}",
        f"Recipient count: {int(X.iloc[0]['recipient_count'])}",
        f"Supervised probability threshold: {threshold:.3f}",
        f"Isolation Forest anomaly threshold: {anomaly_threshold:.4f}",
        f"Anomaly flag: {'Yes' if anomaly_flag else 'No'}",
    ]
    if history_missing:
        details.append(
            "History note: rolling activity features were unavailable and set to 0; "
            "provide employee-history values for meaningful behavioral analysis."
        )

    return {
        "score": score,
        "anomaly": anomaly_score,
        "anomaly_flag": bool(anomaly_flag),
        "label": label,
        "level": level,
        "details": details,
    }


def run_stylometry(subject, content):
    artifact = load_artifact("non_temporal_text_plus_stylometry_v2.joblib")
    if artifact is None:
        raise RuntimeError(
            "Non-temporal model not found. Train the non-temporal model first."
        )

    model = artifact["model"]

    # Match the exact input schema used when the model was trained.
    expected = list(model.feature_names_in_)

    # Initialize a single row with all expected columns.
    row = {column: "" for column in expected}

    # Populate the fields available from the submitted email.
    row.update({
        "subject": subject,
        "content": content,
        "word_count": len(content.split()),
        "char_count": len(content),
        "sentence_count": max(
            1, sum(content.count(p) for p in ".!?")
        ),
        "recipient_count": 1,
        "attachment_count": 0,
        "size": len(content.encode("utf-8")),
    })

    df = pd.DataFrame([row], columns=expected)

    # Generate text and stylometric features using the training helpers.
    df["_text_for_model"] = text_join(df)

    style = stylometric_frame(df)

    for column in artifact["style_columns"]:
        if column not in style.columns:
            raise ValueError(
                f"Stylometric feature missing from feature builder: {column}"
            )
        df[column] = style[column].to_numpy()

    # Reorder columns exactly as expected by the fitted pipeline.
    df = df[expected]

    score = float(model.predict_proba(df)[0, 1])

    # The anomaly detector uses only the saved stylometric feature columns.
    style_input = df[artifact["style_columns"]]
    anomaly = float(
        -artifact["anomaly_model"].decision_function(style_input)[0]
    )

    threshold = float(artifact.get("threshold", 0.5))
    label, level = probability_label(score, threshold)

    details = [
        f"Words: {int(style.iloc[0]['word_count_calc'])}",
        f"Average word length: {style.iloc[0]['avg_word_len']:.2f}",
        f"Vocabulary type-token ratio: {style.iloc[0]['type_token_ratio']:.2f}",
        f"Punctuation count: {int(style.iloc[0]['punctuation_count'])}",
        f"Uppercase ratio: {style.iloc[0]['uppercase_ratio']:.2f}",
    ]

    return {
        "score": score,
        "anomaly": anomaly,
        "label": label,
        "level": level,
        "details": details,
    }

@app.route("/")
def index():
    metrics_path = ART / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
    trained = (ART / "temporal_model.joblib").exists() and (ART / "non_temporal_model.joblib").exists()
    return render_template("index.html", metrics=metrics, trained=trained)


@app.route("/temporal", methods=["GET", "POST"])
def temporal():
    result = None
    if request.method == "POST":
        try:
            content = request.form.get("content") or ""
            row = {
                "timestamp": request.form.get("timestamp") or pd.Timestamp.now().isoformat(),
                "hour": request.form.get("hour") or 12,
                "recipient_count": request.form.get("recipient_count") or 1,
                "external_recipient": request.form.get("external_recipient") or 0,
                "attachment_count": request.form.get("attachment_count") or 0,
                "size": request.form.get("size") or len(content.encode("utf-8")),
                "word_count": request.form.get("word_count") or len(content.split()),
                "char_count": request.form.get("char_count") or len(content),
                "sentence_count": request.form.get("sentence_count") or max(1, sum(content.count(p) for p in ".!?")),
                "username": request.form.get("username") or "single-input",
                "content": content,
            }
            # Optional history values: if the HTML form does not provide them,
            # the temporal builder will disclose that they defaulted to zero.
            for column in [
                "activity_frequency_1h", "activity_frequency_24h",
                "unique_recipients_24h", "off_hours_events_7d",
                "external_recipient_events_7d",
            ]:
                row[column] = request.form.get(column, "")
            result = run_temporal(row)
        except Exception as exc:
            flash(str(exc), "error")
    return render_template("temporal.html", result=result)


@app.route("/stylometry", methods=["GET", "POST"])
def stylometry():
    result = None
    if request.method == "POST":
        try:
            result = run_stylometry(request.form.get("subject", ""), request.form.get("content", ""))
        except Exception as exc:
            flash(str(exc), "error")
    return render_template("stylometry.html", result=result)


@app.route("/batch", methods=["GET", "POST"])
def batch():
    rows = None
    if request.method == "POST":
        try:
            uploaded = request.files.get("file")
            if not uploaded or not uploaded.filename.lower().endswith(".csv"):
                raise ValueError("Upload a CSV file.")
            df = pd.read_csv(uploaded)
            if "content" not in df.columns:
                raise ValueError("CSV needs a content column. A subject column is optional.")
            outputs = []
            for _, record in df.iterrows():
                res = run_stylometry(str(record.get("subject", "")), str(record.get("content", "")))
                outputs.append({
                    "subject": str(record.get("subject", ""))[:70],
                    "label": res["label"],
                    "score": round(res["score"], 4),
                    "anomaly": round(res["anomaly"], 4),
                })
            rows = outputs
        except Exception as exc:
            flash(str(exc), "error")
    return render_template("batch.html", rows=rows)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
