"""
upload_corpus_to_s3.py — Upload local corpus folders to S3.

Resumable: skips files already on S3 with matching size.
Mirrors local folder structure as S3 prefixes.

Usage:
    python upload_corpus_to_s3.py
"""

import boto3
import os
import sys
from pathlib import Path

from tqdm import tqdm

# ── CONFIGURE THESE ───────────────────────────────────────────────────────────

BUCKET   = "beke-corpus-prod"
REGION   = "us-west-1"
PROFILE  = "beke"          # aws configure --profile beke

# Add your local folder paths here, one per line.
# Each folder will appear in S3 under its folder name, e.g.:
#   /Users/you/papers  →  s3://beke-corpus-prod/papers/
# CORPUS_FOLDERS = [
#     "/Users/yanghanghuang/Desktop/PhD Related/2024 Summer Poster/",
#     "/Users/yanghanghuang/Desktop/PhD Related/2024.10 Manuscript Revision/",
#     "/Users/yanghanghuang/Desktop/PhD Related/2024.11_Manuscript_Revision/",
#     "/Users/yanghanghuang/Desktop/PhD Related/2025.03.JMM_Revision/",
#     "/Users/yanghanghuang/Desktop/PhD Related/Alumina Air Battery Literature/",
#     "/Users/yanghanghuang/Desktop/PhD Related/Chenhao File/",
#     "/Users/yanghanghuang/Desktop/PhD Related/Haoxuan File/",
#     "/Users/yanghanghuang/Desktop/PhD Related/Ice MEMS/",
#     "/Users/yanghanghuang/Desktop/PhD Related/IoT4Ag Archieve/",
#     "/Users/yanghanghuang/Desktop/PhD Related/MEMS materials/",
#     "/Users/yanghanghuang/Desktop/PhD Related/MEMS structures/",
#     "/Users/yanghanghuang/Desktop/PhD Related/Publication Archieve/",
#     "/Users/yanghanghuang/Desktop/PhD Related/Reference - Drone Battery/",
#     "/Users/yanghanghuang/Desktop/PhD Related/Weekly Plan/",
#     "/Users/yanghanghuang/Desktop/PhD Related/Yanghang PhD Milestones/",
#     "/Users/yanghanghuang/Desktop/PhD Related/无人机电池Literature Reseach/"
# ]
CORPUS_FOLDERS = [
    "/Users/yanghanghuang/Desktop/PhD Related/Chenhao File/",
    "/Users/yanghanghuang/Desktop/PhD Related/Haoxuan File/",
]

# File extensions to upload (lowercase). Everything else is skipped.
INCLUDE_EXTENSIONS = {".docx", ".pptx", ".pdf"}

# ── END CONFIG ────────────────────────────────────────────────────────────────


def collect_files(folders: list[str]) -> list[tuple[Path, str]]:
    """Return list of (local_path, s3_key) for all matching files."""
    pairs = []
    for folder_str in folders:
        folder = Path(folder_str)
        if not folder.exists():
            print(f"  WARNING: folder not found, skipping — {folder}")
            continue
        folder_name = folder.name
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in INCLUDE_EXTENSIONS:
                continue
            if path.name.startswith("."):
                continue
            # Mirror structure: folder_name/relative/path/file.ext
            relative = path.relative_to(folder.parent)
            s3_key = relative.as_posix()
            pairs.append((path, s3_key))
    return pairs


def already_uploaded(s3, bucket: str, key: str, local_size: int) -> bool:
    """Return True if S3 object exists with same size — skip re-upload."""
    try:
        obj = s3.head_object(Bucket=bucket, Key=key)
        return obj["ContentLength"] == local_size
    except Exception:
        return False


def human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def main():
    if not CORPUS_FOLDERS:
        print("ERROR: CORPUS_FOLDERS is empty — add your folder paths at the top of the script.")
        sys.exit(1)

    print(f"Collecting files from {len(CORPUS_FOLDERS)} folder(s)…")
    files = collect_files(CORPUS_FOLDERS)
    if not files:
        print("No matching files found. Check your paths and INCLUDE_EXTENSIONS.")
        sys.exit(1)

    total_files = len(files)
    total_bytes = sum(p.stat().st_size for p, _ in files)
    print(f"Found {total_files} files ({human_size(total_bytes)} total)\n")

    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    s3 = session.client("s3")

    uploaded = 0
    skipped  = 0
    failed   = 0
    uploaded_bytes = 0

    with tqdm(total=total_files, unit="file", desc="Overall", position=0) as overall_bar:
        for i, (local_path, s3_key) in enumerate(files, 1):
            local_size = local_path.stat().st_size
            prefix = f"[{i}/{total_files}]"

            if already_uploaded(s3, BUCKET, s3_key, local_size):
                tqdm.write(f"  {prefix} SKIP   {s3_key}  ({human_size(local_size)})")
                skipped += 1
                overall_bar.update(1)
                continue

            try:
                with tqdm(
                    total=local_size,
                    unit="B",
                    unit_scale=True,
                    desc=local_path.name[:40],
                    position=1,
                    leave=False,
                ) as file_bar:
                    s3.upload_file(
                        str(local_path), BUCKET, s3_key,
                        Callback=lambda n: file_bar.update(n),
                    )
                tqdm.write(f"  {prefix} OK     {s3_key}  ({human_size(local_size)})")
                uploaded += 1
                uploaded_bytes += local_size
            except Exception as e:
                tqdm.write(f"  {prefix} FAIL   {s3_key}  — {e}")
                failed += 1

            overall_bar.update(1)

    print(f"""
── Summary ──────────────────────────────────────
  Uploaded : {uploaded} files  ({human_size(uploaded_bytes)})
  Skipped  : {skipped} files  (already on S3)
  Failed   : {failed} files
  Bucket   : s3://{BUCKET}/
─────────────────────────────────────────────────
""")

    if failed:
        print("Re-run the script to retry failed files — successful uploads are skipped automatically.")
        sys.exit(1)


if __name__ == "__main__":
    main()
