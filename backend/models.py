"""
NeuroGuard Clinic – Data Models
Structured data classes for sessions and sensor readings.
"""
from datetime import datetime


QUESTION_OPTIONS = [
    {"value": 0, "label": "Not at all"},
    {"value": 1, "label": "Several days"},
    {"value": 2, "label": "More than half the days"},
    {"value": 3, "label": "Nearly every day"},
]

OPTION_LABELS = {0: "Not at all", 1: "Several days", 2: "More than half the days", 3: "Nearly every day"}

PHQ9_QUESTIONS = [
    "Little interest or pleasure in doing things?",
    "Feeling down, depressed, or hopeless?",
    "Trouble falling or staying asleep, or sleeping too much?",
    "Feeling tired or having little energy?",
    "Poor appetite or overeating?",
    "Feeling bad about yourself — or that you are a failure or have let yourself or your family down?",
    "Trouble concentrating on things, such as reading the newspaper or watching television?",
    "Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless?",
    "Thoughts that you would be better off dead, or of hurting yourself in some way?",
]


def get_severity(score):
    """PHQ-9 severity classification."""
    if score <= 4:
        return 'Normal'
    elif score <= 9:
        return 'Mild'
    elif score <= 14:
        return 'Moderate'
    elif score <= 19:
        return 'Moderately Severe'
    else:
        return 'Severe'


def get_severity_color(severity):
    """Return CSS color class for severity."""
    colors = {
        'Normal': 'success',
        'Mild': 'warning',
        'Moderate': 'orange',
        'Moderately Severe': 'danger',
        'Severe': 'danger',
    }
    return colors.get(severity, 'secondary')
