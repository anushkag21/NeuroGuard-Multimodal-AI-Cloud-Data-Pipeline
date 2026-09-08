import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

load_dotenv()

connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

if not connection_string:
    raise ValueError("AZURE_STORAGE_CONNECTION_STRING not found in .env")

blob_service_client = BlobServiceClient.from_connection_string(
    connection_string
)

container_name = "raw-data"
container_client = blob_service_client.get_container_client(container_name)

print("Connecting to Azure Blob Storage...")

blobs = list(container_client.list_blobs())

print("✓ Azure Blob Storage connection successful!")
print(f"✓ Storage Account: neuroguarddata2026")
print(f"✓ Container: {container_name}")
print(f"✓ Existing blobs: {len(blobs)}")