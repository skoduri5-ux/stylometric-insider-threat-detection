from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, flash

# Use the exact feature builder used by train_np_v2.py.
from train_np_v2 import stylometric_frame

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
NON_TEMPORAL_ARTIFACT = "non_temporal_text_plus_stylometry_v2.joblib"
LEGACY_ARTIFACT = "non_temporal_model.joblib"

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
    """Build the feature schema expected by the existing temporal artifact."""
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

    anomaly_X = artifact["anomaly_imputer"].transform(X)
    anomaly_score = float(-artifact["anomaly_model"].score_samples(anomaly_X)[0])
    anomaly_threshold = float(artifact.get("anomaly_threshold", float("inf")))
    anomaly_flag = anomaly_score >= anomaly_threshold

    history_columns = [
        "activity_frequency_1h", "activity_frequency_24h",
        "unique_recipients_24h", "off_hours_events_7d",
        "external_recipient_events_7d",
    ]
    history_missing = not any(row.get(c) not in (None, "") for c in history_columns)
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
        "score": score, "anomaly": anomaly_score,
        "anomaly_flag": bool(anomaly_flag), "label": label,
        "level": level, "details": details,
    }


def run_stylometry(subject, content):
    """Run the v2 text-plus-stylometry classifier and legacy IF as a separate score."""
    artifact = load_artifact(NON_TEMPORAL_ARTIFACT)
    if artifact is None:
        raise RuntimeError(
            f"V2 model not found: artifacts/{NON_TEMPORAL_ARTIFACT}. "
            "Run train_np_v2.py first."
        )

    model = artifact["model"]
    style_columns = list(artifact.get("style_columns", []))
    if artifact.get("model_type") != "text_plus_stylometry":
        raise RuntimeError("The selected artifact is not the text-plus-stylometry v2 model.")
    print("\n===== NON-TEMPORAL DEBUG =====")
    print("Model path:", ART / "non_temporal_text_plus_stylometry_v2.joblib")
    print("Model type:", artifact.get("model_type"))
    print("Threshold:", artifact.get("threshold"))
    print("Input content:", repr(content))
    print("Input length:", len(content))

    # This mirrors prepare() in train_np_v2.py. Subject is retained for the UI,
    # but the trained classifier uses email body only.
    df = pd.DataFrame({"subject": [subject], "content": [content]})
    df["_text_for_model"] = df["content"].fillna("").astype(str)
    style = stylometric_frame(df)
    for column in style_columns:
        if column not in style.columns:
            raise ValueError(f"V2 stylometric feature missing: {column}")
        df[column] = style[column].to_numpy()

    score = float(model.predict_proba(df)[0, 1])
    print("Flask classifier score:", score)
    print("Expected columns:", list(df.columns))
    print("Stylometric features:")
    print(df[artifact["style_columns"]].to_dict(orient="records"))
    print("===== END DEBUG =====\n")
    threshold = float(artifact["threshold"])
    label, level = probability_label(score, threshold)

    # The v2 artifact does not contain an Isolation Forest. To preserve the
    # existing UI's anomaly field, optionally calculate it with the legacy IF
    # only when its feature schema exactly matches. It remains a separate,
    # uncalibrated anomaly ranking, not a probability or classification.
    anomaly = None
    legacy = load_artifact(LEGACY_ARTIFACT)
    if legacy is not None and "anomaly_model" in legacy:
        legacy_cols = list(legacy.get("style_columns", []))
        if legacy_cols == style_columns and all(c in df.columns for c in legacy_cols):
            try:
                anomaly = float(-legacy["anomaly_model"].decision_function(df[legacy_cols])[0])
            except Exception:
                anomaly = None

    details = [
        f"Classifier: text + stylometry v2",
        f"Email subject is not used by the classifier; body text only.",
        f"Suspicious-class score: {score:.4f}",
        f"Validation-selected threshold: {threshold:.4f}",
        f"Words: {int(style.iloc[0]['word_count_calc'])}",
        f"Average word length: {style.iloc[0]['avg_word_len']:.2f}",
        f"Vocabulary type-token ratio: {style.iloc[0]['type_token_ratio']:.2f}",
        f"Punctuation count: {int(style.iloc[0]['punctuation_count'])}",
        f"Uppercase ratio: {style.iloc[0]['uppercase_ratio']:.2f}",
    ]
    if anomaly is None:
        details.append("Anomaly score: unavailable for the v2 artifact; no v2 Isolation Forest was saved.")
    else:
        details.append("Anomaly score: from the legacy Isolation Forest; uncalibrated and separate from the classifier.")

    return {
        "score": score, "anomaly": anomaly, "label": label,
        "level": level, "details": details,
    }


@app.route("/")
def index():
    metrics_path = ART / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
    trained = (ART / "temporal_model.joblib").exists() and (ART / NON_TEMPORAL_ARTIFACT).exists()
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
            result = run_stylometry(
                request.form.get("subject", ""),
                request.form.get("content", ""),
            )
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
                subject = "" if pd.isna(record.get("subject", "")) else str(record.get("subject", ""))
                content = "" if pd.isna(record.get("content", "")) else str(record.get("content", ""))
                res = run_stylometry(subject, content)
                outputs.append({
                    "subject": subject[:70],
                    "label": res["label"],
                    "score": round(res["score"], 4),
                    "anomaly": res["anomaly"],
                })
            rows = outputs
        except Exception as exc:
            flash(str(exc), "error")
    return render_template("batch.html", rows=rows)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
