import os
import json

from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient, ContentSettings

load_dotenv()

connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

if not connection_string:
    raise ValueError("AZURE_STORAGE_CONNECTION_STRING not found")

blob_service_client = BlobServiceClient.from_connection_string(
    connection_string
)

# -----------------------------
# CONFIG
# -----------------------------
RAW_CONTAINER = "raw-data"
PROCESSED_CONTAINER = "processed-data"

session_blob = "sessions/NG-20260908-35432.json"


# -----------------------------
# EXTRACT
# -----------------------------
raw_blob = blob_service_client.get_blob_client(
    container=RAW_CONTAINER,
    blob=session_blob
)

data = raw_blob.download_blob().readall()
raw_data = json.loads(data)

print("✓ Raw data extracted")


# -----------------------------
# VALIDATION
# -----------------------------
errors = []

if not raw_data.get("session_id"):
    errors.append("Missing session_id")

score = raw_data.get("score")
if score is not None and not (0 <= score <= 27):
    errors.append("Invalid score")

confidence = raw_data.get("confidence")
if confidence is not None and not (0 <= confidence <= 100):
    errors.append("Invalid confidence")

duration = raw_data.get("duration_seconds")
if duration is not None and duration < 0:
    errors.append("Invalid duration")

if errors:
    print("\n⚠ Validation failed:")
    for error in errors:
        print(" -", error)
    raise ValueError("Data validation failed")

print("✓ Validation passed")


# -----------------------------
# TRANSFORMATION
# -----------------------------
camera = raw_data.get("sensor_summary", {}).get("camera", {})

processed_data = {
    "session_id": raw_data.get("session_id"),
    "patient_id": raw_data.get("patient_id"),
    "severity": raw_data.get("severity"),
    "score": score,
    "confidence": confidence,
    "duration_seconds": duration,
    "data_source": raw_data.get("sensors_used", []),
    "created_at": raw_data.get("created_at"),

    "camera_active": camera.get("active", False),
    "face_detected": camera.get("face_detected", False),
    "avg_blink_rate": camera.get("avg_blink_rate", 0),
    "avg_posture_score": camera.get("avg_posture_score", 0),
    "dominant_expression": camera.get(
        "dominant_expression", "unknown"
    ),
    "total_frames_analyzed": camera.get(
        "total_frames_analyzed", 0
    )
}

print("✓ Data transformed")


# -----------------------------
# LOAD
# -----------------------------
container_client = blob_service_client.get_container_client(
    PROCESSED_CONTAINER
)

if not container_client.exists():
    container_client.create_container()
    print("✓ Created processed-data container")
else:
    print("✓ processed-data container exists")


processed_blob_name = (
    f"sessions/{processed_data['session_id']}_processed.json"
)

processed_blob = blob_service_client.get_blob_client(
    container=PROCESSED_CONTAINER,
    blob=processed_blob_name
)

processed_blob.upload_blob(
    json.dumps(processed_data, indent=2),
    overwrite=True,
    content_settings=ContentSettings(
        content_type="application/json"
    )
)

print(
    f"✓ Processed data uploaded: "
    f"{PROCESSED_CONTAINER}/{processed_blob_name}"
)


# -----------------------------
# FINAL OUTPUT
# -----------------------------
print("\n========== ETL COMPLETE ==========")
print("Extract  : ✓")
print("Validate : ✓")
print("Transform : ✓")
print("Load     : ✓")