import os, uuid
from fastapi import UploadFile

async def save_upload_file(uploadFile: UploadFile, dest_dir: str) -> str:
    """
    Save an uploaded file to the specified directory with a unique name.
    Returns the path to the saved file.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
    
    extension = os.path.splitext(uploadFile.filename)[-1].lower()
    unique_filename = f"{uuid.uuid4().hex}{extension}"
    file_path = os.path.join(dest_dir, unique_filename)
    
    with open(file_path, "wb") as buffer:
        while True:
            chunk = await uploadFile.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)
    
    return file_path