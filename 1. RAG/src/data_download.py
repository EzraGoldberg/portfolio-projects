from dotenv import load_dotenv
import kagglehub
import shutil
import os

load_dotenv()

DATA_DIR = "data/1_source"

def download():
    datasets = {
        "lgbt_EU"            : "ruslankl/european-union-lgbt-survey-2012",
        "retractions"        : "kanchana1990/global-scientific-retractions-19272026",
        "HIV AIDS data"      : "imdevskp/hiv-aids-dataset",
        "UNICEF Immunization": "fahimvj/immunization-data-unicef"
    }

    for name, ds in datasets.items():
        path = kagglehub.dataset_download(ds)
        target = os.path.join(DATA_DIR, name)
        shutil.copytree(path, target, dirs_exist_ok=True)
        print(f"Downloaded {name} → {target}")

if __name__ == "__main__":
    download()

    