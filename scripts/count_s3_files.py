"""Utility: count files by type in s3://beke-corpus-prod.
Run: python count_s3_files.py
"""
from collections import defaultdict
from pathlib import Path
import boto3

BUCKET     = "beke-corpus-prod"
AWS_PROFILE = "beke"
AWS_REGION  = "us-west-1"

session   = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
client    = session.client("s3")
paginator = client.get_paginator("list_objects_v2")

counts     = defaultdict(int)
total_size = defaultdict(int)

for page in paginator.paginate(Bucket=BUCKET):
    for obj in page.get("Contents", []):
        ext = Path(obj["Key"]).suffix.lower()
        counts[ext] += 1
        total_size[ext] += obj["Size"]

print(f"\n{'Extension':<12} {'Count':>8} {'Size (MB)':>12}")
print("-" * 35)
for ext in sorted(counts):
    mb = total_size[ext] / 1_048_576
    print(f"{ext:<12} {counts[ext]:>8} {mb:>12.1f}")
print("-" * 35)
total = sum(counts.values())
total_mb = sum(total_size.values()) / 1_048_576
print(f"{'TOTAL':<12} {total:>8} {total_mb:>12.1f}")
