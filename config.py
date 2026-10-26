"""Shared configuration: schema, paths, and constants.

SWAP POINT: replace FEATURES / TARGET / category lists with the exact ones
from your midterm notebook so the app matches your saved pipeline.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "trained_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
SAMPLE_DATA_PATH = ROOT / "data" / "sample_reservations.csv"
DIAGRAM_PATH = ROOT / "assets" / "use_case_diagram.png"

TARGET = "cancelled_or_no_show"  # 1 = cancelled / no-show, 0 = honoured

CATEGORICAL = [
    "facility_type", "building", "day_of_week", "requester_type", "semester_period",
]
NUMERIC = [
    "start_hour", "duration_hours", "lead_time_days", "attendees",
    "prior_reservations", "prior_cancellations", "is_recurring",
]
FEATURES = CATEGORICAL + NUMERIC

# Columns known only AFTER the reservation outcome -> must never reach inference.
LEAKAGE_COLUMNS = ["actual_checkin_time", "cancellation_reason", "cancelled_at", "checked_in"]

CATEGORY_VALUES = {
    "facility_type": ["Classroom", "Computer Lab", "Auditorium", "Gymnasium", "Meeting Room"],
    "building": ["Main", "Science", "Engineering", "Library", "Sports Complex"],
    "day_of_week": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
    "requester_type": ["Student Org", "Faculty", "Staff", "Student"],
    "semester_period": ["Start", "Midterm", "Finals", "Break"],
}

NUMERIC_BOUNDS = {
    "start_hour": (7, 20),
    "duration_hours": (1, 8),
    "lead_time_days": (0, 60),
    "attendees": (1, 500),
    "prior_reservations": (0, 100),
    "prior_cancellations": (0, 100),
    "is_recurring": (0, 1),
}

RISK_THRESHOLDS = {"high": 0.60, "medium": 0.35}  # probability cut-offs for review queue

ACTORS = {
    "Requester (student/faculty/staff)": "Submits reservations; cannot see risk scores.",
    "Facility Administrator": "Views risk queue, records the FINAL decision (confirm, remind, reconfirm, release).",
    "Registrar/IT Admin": "Uploads data, maintains model, manages access.",
}
