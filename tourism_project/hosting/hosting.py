from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

# Create the Space if it does not exist
api.create_repo(
    repo_id="samuelrego/tourism-package-prediction",
    repo_type="space",
    space_sdk="docker",
    exist_ok=True,
)

api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id="samuelrego/tourism-package-prediction",
    repo_type="space",
    path_in_repo="",
)
print("Deployment files pushed to Hugging Face Space successfully.")
