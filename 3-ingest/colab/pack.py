"""Bundle the code the Colab embed notebook imports -> 3-ingest/colab/dist/r2ai_ingest.zip.
Upload: gcloud storage cp 3-ingest/colab/dist/r2ai_ingest.zip gs://aiguru_crawler/code/r2ai_ingest.zip"""
import glob
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # main-pipeline/
FILES = ["2-extract/extract/__init__.py", "2-extract/extract/storage.py", "2-extract/extract/clean.py"]
FILES += [os.path.relpath(p, ROOT).replace("\\", "/") for p in glob.glob(os.path.join(ROOT, "3-ingest", "ingest", "*.py"))]

if __name__ == "__main__":
    dst = os.path.join(ROOT, "3-ingest", "colab", "dist", "r2ai_ingest.zip")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES:  # extract/... and ingest/... at the zip root, as in the Docker image
            z.write(os.path.join(ROOT, f), f.split("/", 1)[1])
    print(dst, len(FILES), "files")
