import os

def scan_directory(folder_path):
    files_data = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()

            files_data.append({
                "file_name": file,
                "file_path": file_path,
                "extension": file_ext
            })

    return files_data
