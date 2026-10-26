"""Module 5: SAD content - requirements, actors, use cases, workflow."""
from __future__ import annotations

FUNCTIONAL = [
    ("FR1", "Upload a reservation CSV and validate its schema, types, nulls and duplicates.",
     "Dataset page"),
    ("FR2", "Show descriptive statistics, class distribution and interactive charts.",
     "Data Analysis page"),
    ("FR3", "Predict cancellation/no-show risk for a single reservation using the saved pipeline.",
     "Prediction page"),
    ("FR4", "Score a batch file and rank reservations into a High/Medium/Low review queue.",
     "Dashboard page"),
    ("FR5", "Let the facility administrator record a final decision for each flagged reservation.",
     "Dashboard page (decision log)"),
]
NON_FUNCTIONAL = [
    ("NFR1", "Performance", "Model is loaded once (cached); pages respond without retraining."),
    ("NFR2", "Privacy/Security", "Only synthetic or authorised data; no credentials committed; leakage fields dropped."),
    ("NFR3", "Transparency", "Model version, timestamp, limitations and human-review notice shown on every prediction."),
]
USE_CASES = [
    "Upload & validate dataset", "Explore data", "Predict single reservation",
    "Score batch & view risk queue", "Record final decision", "Maintain model/data",
]
WORKFLOW_STEPS = [
    "1. Data intake: CSV uploaded and validated (schema, nulls, duplicates, leakage columns dropped).",
    "2. Feature preparation: columns ordered/typed exactly as in training.",
    "3. Prediction: saved pipeline returns class + risk probability.",
    "4. Risk banding: High / Medium / Low using documented thresholds.",
    "5. Review queue: sorted by risk with recommended human follow-up.",
    "6. Human decision: Facility Administrator confirms, reminds, reconfirms or releases the slot.",
    "7. Audit: decision, user, timestamp and model version logged.",
]
DECISIONS = ["Confirm as is", "Send reminder", "Contact to reconfirm",
             "Add waitlist backup", "Release slot (after contact)"]
ACTOR_ACCESS = [
    ("Requester", "Submit reservation", "No", "No"),
    ("Facility Administrator", "Review queue", "Yes", "Yes (final decision)"),
    ("Registrar/IT Admin", "Upload data / maintain model", "Yes", "No"),
]
