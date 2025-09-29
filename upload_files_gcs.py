import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import storage
from tqdm import tqdm
import time
def upload_blob(bucket, source_file_name, destination_blob_name):
    blob = bucket.blob(destination_blob_name)
    try:
        blob.upload_from_filename(source_file_name)
        return True  # Indicate success
    except Exception as e:
        print(f"Error uploading {source_file_name} to gs://{bucket.name}/{destination_blob_name}: {e}")
        return False  # Indicate failure
def upload_folder_with_progress(bucket_name, source_folder, prefix="", max_workers=8, credentials_path=None):
    print(f"Starting upload of folder '{source_folder}' to bucket '{bucket_name}' with prefix '{prefix}'...")
    # Initialize GCS client with credentials
    try:
        if credentials_path:
            storage_client = storage.Client.from_service_account_json(credentials_path)
            print(f"Using credentials from: {credentials_path}")
        else:
            storage_client = storage.Client()
            print("Using default Google Cloud credentials.")
    except Exception as e:
        print(f"Error initializing Google Cloud Storage client: {e}")
        print("Please ensure your credentials are correctly set up.")
        return
    try:
        bucket = storage_client.bucket(bucket_name)
        # Verify if the bucket exists and is accessible
        bucket.exists()
    except Exception as e:
        print(f"Error accessing bucket '{bucket_name}': {e}")
        print("Please check if the bucket exists and your credentials have permission to access it.")
        return
    files_to_upload = []
    if not os.path.isdir(source_folder):
        print(f"Error: Source folder '{source_folder}' does not exist or is not a directory.")
        return
    for root, _, files in os.walk(source_folder):
        for file in files:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, source_folder)
            destination_blob_name = os.path.join(prefix, relative_path).replace("\\", "/")
            files_to_upload.append((full_path, destination_blob_name))
    num_files = len(files_to_upload)
    if num_files == 0:
        print(f"No files found in '{source_folder}' to upload.")
        return
    print(f"Found {num_files} files to upload.")
    successful_uploads = 0
    failed_uploads = 0
    # Use ThreadPoolExecutor to upload files concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for full_path, dest_blob in files_to_upload:
            futures.append(executor.submit(upload_blob, bucket, full_path, dest_blob))
        for future in tqdm(as_completed(futures), total=num_files, desc="Uploading files", unit="file"):
            try:
                if future.result():  # True if upload was successful
                    successful_uploads += 1
                else:
                    failed_uploads += 1
            except Exception as e:

                print(f"An unexpected error occurred during file processing: {e}")
                failed_uploads += 1
    print("\nUpload process finished.")
    print(f"Summary: {successful_uploads} files uploaded successfully, {failed_uploads} files failed.")
    if failed_uploads > 0:
        print("Please review the logs above for details on failed uploads.")
        
if __name__ == "__main__":
    BUCKET_NAME = "similar_dataset"  #GCS bucket name
    SOURCE_FOLDER = 'C:/Users/Nick/Desktop/finalset'  # Path to local folder
    PREFIX = "cnn_set"  # GCS folder
    CREDENTIALS_PATH = "C:/Users/Nick/Desktop/AI_Resource/GCS/tokyo-bird-463814-q4-7a1f113af459.json"
    MAX_WORKERS = 10
    upload_folder_with_progress(
        bucket_name=BUCKET_NAME,
        source_folder=SOURCE_FOLDER,
        prefix=PREFIX,
        max_workers=MAX_WORKERS,
        credentials_path=CREDENTIALS_PATH
    )












