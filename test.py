
import joblib
import pandas as pd
from train_np_v2 import prepare

emails = [
    {
        "name": "Project Update",
        "content": "Dear team, Please send the updation mail on the current project. And also send further work to be completed in scheduled time."
    },
    {
        "name": "Meeting Reminder",
        "content": "Hello everyone, This is a reminder about our project meeting tomorrow at 10 AM. Please review the agenda before joining."
    },
    {
        "name": "Leave Request",
        "content": "Dear Manager, I would like to request leave for two days next week due to personal work. Please approve my request."
    },
    {
        "name": "Confidential File Transfer",
        "content": "I will copy the confidential project files and transfer them to an external storage device before leaving the organization."
    },
    {
        "name": "Personal Account Transfer",
        "content": "Send the confidential customer database to my personal email account so I can access it outside the company network."
    }
]

for model_name in ["text_only", "text_plus_stylometry"]:
    artifact = joblib.load(
        f"artifacts/non_temporal_{model_name}_v2.joblib"
    )

    model = artifact["model"]
    threshold = artifact["threshold"]
    style_cols = artifact["style_columns"]

    print("\nMODEL:", model_name)

    for email in emails:
        df = pd.DataFrame({
            "subject": [email["name"]],
            "content": [email["content"]]
        })

        df = prepare(df, style_cols)
        score = model.predict_proba(df)[0, 1]

        prediction = (
            "SUSPICIOUS" if score >= threshold else "NORMAL"
        )

        print(
            f'{email["name"]}: '
            f'Score={score:.4f}, '
            f'Prediction={prediction}'
        )
