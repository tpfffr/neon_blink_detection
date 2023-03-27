import json
import os
import zipfile
from pathlib import Path

# from pupilcloud import Api, ApiException


# def download(recording_dir, recording_ids):
#     recording_dir = Path(recording_dir)
#     recording_dir.mkdir(parents=True, exist_ok=True)

#     api_token = get_api_token()
#     host = "https://api.cloud-staging.pupil-labs.com/"
#     api = Api(api_key=api_token, host=host)

#     for recording_id in recording_ids:
#         print(f"Downloading {recording_id}")
#         _download_zip(api, recording_dir, recording_id)
#     rename_folders(recording_dir)


# def _download_zip(api, recording_dir, recording_id):
#     try:
#         response = api.download_recording_zip(recording_id, _preload_content=False)
#         recording_zip = recording_dir / f"recording_{recording_id}.zip"
#         with open(recording_zip, "wb") as fp:
#             fp.write(response.read())

#     except ApiException as e:
#         print("Exception when calling RecordingsApi->download_recording_zip: %s\n" % e)
#         return

#     with zipfile.ZipFile(recording_zip, "r") as zip_ref:
#         zip_ref.extractall(recording_dir)
#     os.remove(recording_zip)


# def rename_folders(download_path):
#     folders = filter(lambda f: f.is_dir(), download_path.iterdir())
#     for folder in folders:
#         recording_id = get_recording_id(folder)
#         if recording_id:
#             new_path = download_path / recording_id
#             if new_path != folder:
#                 print(f"renamed {folder} to {new_path}")
#                 folder.rename(new_path)


# def get_recording_id(folder: Path):
#     info_path = folder / "info.json"
#     if info_path.is_file():
#         info = json.load(info_path.open())
#         return info.get("recording_id")


# def get_api_token():
#     api_token_path = Path.home() / "api_token"
#     api_token = api_token_path.read_text().rstrip("\n")
#     print(api_token)
#     return api_token


# if __name__ == "__main__":
#     _recording_dir = Path("/users/Ching/datasets/blink_detection/staging")
#     _recording_id = "abe6ec68-d0c5-4bf1-b3ec-83523e518b93"
#     download(_recording_dir, [_recording_id])
