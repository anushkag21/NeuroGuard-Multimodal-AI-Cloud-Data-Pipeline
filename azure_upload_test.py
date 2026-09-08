import os
import json
from datetime import datetime
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

load_dotenv()

connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

if not connection_string:
    raise ValueError("AZURE_STORAGE_CONNECTION_STRING not found")

# Connect to Azure
blob_service_client = BlobServiceClient.from_connection_string(
    connection_string
)

container_name = "raw-data"
container_client = blob_service_client.get_container_client(container_name)

# Sample NeuroGuard real-time session data
data = {
    "session_id": "NG_TEST_001",
    "timestamp": datetime.utcnow().isoformat(),

    "heart_rate_bpm": 82,
    "spo2": 97,
    "hrv_rmssd": 38,

    "blink_rate": 24,
    "sad_expression": 42,
    "posture_score": 68,

    "phq9_base_score": 11
}

# Convert JSON to bytes
json_data = json.dumps(data, indent=2)

# File name
blob_name = f"session_{data['session_id']}.json"

# Upload
blob_client = container_client.get_blob_client(blob_name)

blob_client.upload_blob(
    json_data,
    overwrite=True
)

print("✓ Data uploaded successfully!")
print(f"✓ Container: {container_name}")
print(f"✓ Blob: {blob_name}")
print("\nUploaded data:")
print(json_data)