"""BEKE — S3 source file download service.

Functions (bottom-up):
  generate_presigned_url()  — create a temporary download link for an S3 object [COMPLETED]
  render_download_button()  — Streamlit download link, hidden from guests
"""

from __future__ import annotations

import boto3                         # pip install boto3
from botocore.exceptions import ClientError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUCKET    = "beke-corpus-prod"
REGION    = "us-west-1"
EXPIRY_S  = 300  # 5 minutes


# ---------------------------------------------------------------------------
# generate_presigned_url (COMPLETED by user)
# ---------------------------------------------------------------------------

def generate_presigned_url(s3_key: str) -> str | None:
    """Return a presigned GET URL for s3_key, or None if an error occurs."""
    s3 = boto3.client("s3", region_name=REGION)
    try:
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": BUCKET, "Key": s3_key},
            ExpiresIn=EXPIRY_S,
        )
    except ClientError as e:
        print(f"Error generating presigned URL for {s3_key}: {e}")
        return None
    return url


# ---------------------------------------------------------------------------
# Scaffold 2 of 2 — render_download_button
# render_download_button: show a download link for a cited source file.
# Input:  s3_key — object key from chunk metadata (e.g. "/corpus/AAB_Manuscript.docx")
# Output: None (renders a Streamlit link, or nothing for guests)
# ---------------------------------------------------------------------------

def render_download_button(s3_key: str) -> None:
    """Render a download link for the source file. Hidden from guests."""
    import streamlit as st
    from src.auth import is_guest

    # Step 1: gate — guests don't get download access
    if is_guest():                                                          # (easy)
    # Answer: is_guest()
        return

    # Step 2: strip leading slash from metadata key (S3 keys don't start with /)
    clean_key = s3_key.lstrip("/")                                                  # (think)
    # Answer: s3_key.lstrip("/")

    # Step 3: generate the presigned URL
    url = generate_presigned_url(clean_key)                                                        # (easy)
    # Answer: generate_presigned_url(clean_key)

    # Step 4: render as a clickable link (only if URL generation succeeded)
    if url:
        filename = clean_key.split("/")[-1]
        st.markdown(
            f'<a href="{url}" target="_blank" download="{filename}"'
            f' style="color: #00d4ff; text-decoration: none;">'
            f'⬇ Download {filename}</a>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_key = "corpus/AAB_Manuscript.docx"
    url = generate_presigned_url(test_key)
    if url:
        print(f"Presigned URL (expires in {EXPIRY_S}s):")
        print(url[:80] + "...")
    else:
        print("Failed — check AWS credentials / network")


# ---- HINTS (uncover only if stuck > 5 min) ----
# Step 1: what does the auth module expose for checking the current role?
# Step 2: metadata stores "/corpus/file.docx" but S3 keys are "corpus/file.docx"
# Step 3: you already built the function above — call it with the cleaned key


# ---- Reflection question ----
# Q: generate_presigned_url() creates a new boto3 client on every call.
#    If render_download_button() runs for 5 sources per query, that's 5 client
#    instantiations. What caching strategy would you use, and does Streamlit's
#    @st.cache_resource help here?