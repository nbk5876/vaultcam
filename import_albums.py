"""
VaultCam — Bulk Album Cover Import Script
==========================================
Imports existing JPEG album cover photos into VaultCam.

For each image it will:
  1. Compress the image to 800x800px max (saves ~93% database space)
  2. Call GPT-4o Vision to extract artist, title, label, year, genre,
     condition, pressing type, and estimated collector value
  3. Insert an Item record into the vaultcam database

Usage:
  cd C:\\Users\\gb105\\projects\\vaultcam
  python import_albums.py

The script reads DATABASE_URL and OPENAI_API_KEY from your .env file.
It is safe to re-run — already-imported images are skipped.

Requirements:
  pip install pillow openai python-dotenv psycopg2-binary sqlalchemy
"""

import os
import sys
import base64
import json
import io
import time
from pathlib import Path
from datetime import datetime, timezone

from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# Configuration — edit these two lines before running
# ---------------------------------------------------------------------------
IMAGES_FOLDER = r"C:\Users\gb105\Projects\vaultcam-album-covers"
OWNER_EMAIL   = "test@example.com"   # the VaultCam user account to import under
# ---------------------------------------------------------------------------

load_dotenv()

DATABASE_URL  = os.environ.get("DATABASE_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not DATABASE_URL:
    print("ERROR: DATABASE_URL not found in .env")
    sys.exit(1)
if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY not found in .env")
    sys.exit(1)

# SQLAlchemy engine
engine = create_engine(DATABASE_URL)
client = OpenAI(api_key=OPENAI_API_KEY)

ALBUM_PROMPT = """Analyze this vinyl record album cover. Return ONLY a JSON object, no other text:
{
  "artist": "artist or band name or null",
  "title": "album title or null",
  "label": "record label name or null",
  "year": "release year as 4-digit string or null",
  "genre": "one of: rock/pop/jazz/blues/classical/country/soul/funk/folk/electronic/other or null",
  "condition": "one of: mint/near_mint/very_good_plus/very_good/good/fair/poor or null",
  "pressing": "one of: original/reissue/repress/unknown or null",
  "color_variant": "black/colored/picture_disc or null",
  "estimated_value": estimated collector market value in USD as a number or null,
  "confidence": "high/medium/low"
}"""

MAX_IMAGE_PX = 800
JPEG_QUALITY = 75


def compress_image(image_path: Path) -> tuple[bytes, str]:
    """Resize and compress a JPEG. Returns (bytes, mime_type)."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((MAX_IMAGE_PX, MAX_IMAGE_PX))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        return buffer.getvalue(), "image/jpeg"


def analyze_with_gpt4o(image_bytes: bytes, mime_type: str) -> dict:
    """Send image to GPT-4o Vision. Returns extracted fields as dict."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                {"type": "text", "text": ALBUM_PROMPT}
            ]
        }],
        max_tokens=400
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def get_or_create_user(conn, email: str) -> int:
    """Return user_id for the given email. Exits if user not found."""
    row = conn.execute(
        text("SELECT id FROM users WHERE email = :email"),
        {"email": email}
    ).fetchone()
    if not row:
        print(f"\nERROR: No user found with email '{email}'")
        print("Please sign up at http://127.0.0.1:5000/signup first, then re-run this script.")
        sys.exit(1)
    return row[0]


def get_vinyl_category_id(conn) -> int:
    """Return category_id for vinyl_album. Exits if category not seeded yet."""
    row = conn.execute(
        text("SELECT id FROM categories WHERE slug = 'vinyl_album'")
    ).fetchone()
    if not row:
        print("\nERROR: vinyl_album category not found in database.")
        print("Start the Flask app once (python vaultcam.py) to seed the categories, then re-run.")
        sys.exit(1)
    return row[0]


def already_imported(conn, user_id: int, filename: str) -> bool:
    """Check if this filename was already imported (stored in notes field)."""
    row = conn.execute(
        text("SELECT id FROM items WHERE user_id = :uid AND notes LIKE :fname"),
        {"uid": user_id, "fname": f"%{filename}%"}
    ).fetchone()
    return row is not None


def insert_item(conn, user_id: int, category_id: int,
                extracted: dict, image_b64: str, mime_type: str, filename: str):
    """Insert a single item record."""
    properties = {
        "condition":       extracted.get("condition"),
        "pressing":        extracted.get("pressing"),
        "color_variant":   extracted.get("color_variant"),
        "estimated_value": extracted.get("estimated_value"),
        "genre":           extracted.get("genre"),
        "year":            extracted.get("year"),
        "label":           extracted.get("label"),
    }
    conn.execute(text("""
        INSERT INTO items
            (user_id, category_id, name, brand, properties, status,
             image_data, ai_confidence, notes, created_at)
        VALUES
            (:user_id, :category_id, :name, :brand, :properties::jsonb, :status,
             :image_data, :confidence, :notes, :created_at)
    """), {
        "user_id":     user_id,
        "category_id": category_id,
        "name":        extracted.get("title") or "Unknown Title",
        "brand":       extracted.get("artist") or "Unknown Artist",
        "properties":  json.dumps(properties),
        "status":      "owned",
        "image_data":  f"data:{mime_type};base64,{image_b64}",
        "confidence":  extracted.get("confidence", "low"),
        "notes":       f"Bulk imported from {filename}",
        "created_at":  datetime.now(timezone.utc),
    })
    conn.commit()


def main():
    folder = Path(IMAGES_FOLDER)
    if not folder.exists():
        print(f"ERROR: Folder not found: {IMAGES_FOLDER}")
        sys.exit(1)

    images = sorted(folder.glob("*.JPEG")) + sorted(folder.glob("*.jpeg")) + \
             sorted(folder.glob("*.jpg"))  + sorted(folder.glob("*.JPG"))

    if not images:
        print(f"No JPEG files found in {IMAGES_FOLDER}")
        sys.exit(1)

    print(f"\nVaultCam Bulk Album Import")
    print(f"==========================")
    print(f"Folder : {IMAGES_FOLDER}")
    print(f"Images : {len(images)} files found")
    print(f"User   : {OWNER_EMAIL}")
    print(f"Model  : gpt-4o  |  Max size: {MAX_IMAGE_PX}px  |  Quality: {JPEG_QUALITY}")
    print()

    with engine.connect() as conn:
        user_id     = get_or_create_user(conn, OWNER_EMAIL)
        category_id = get_vinyl_category_id(conn)

        skipped  = 0
        imported = 0
        failed   = 0

        for i, image_path in enumerate(images, 1):
            filename = image_path.name
            prefix   = f"[{i:>3}/{len(images)}] {filename}"

            # Skip already-imported
            if already_imported(conn, user_id, filename):
                print(f"{prefix}  →  already imported, skipping")
                skipped += 1
                continue

            try:
                # 1. Compress
                original_kb  = image_path.stat().st_size // 1024
                image_bytes, mime_type = compress_image(image_path)
                compressed_kb = len(image_bytes) // 1024
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")

                # 2. GPT-4o Vision
                extracted = analyze_with_gpt4o(image_bytes, mime_type)

                artist = extracted.get("artist") or "Unknown"
                title  = extracted.get("title")  or "Unknown"
                value  = extracted.get("estimated_value")
                conf   = extracted.get("confidence", "?")
                value_str = f"  ${value}" if value else ""

                print(f"{prefix}  →  {artist} — {title}{value_str}  [{conf}]  "
                      f"({original_kb}KB → {compressed_kb}KB)")

                # 3. Insert
                insert_item(conn, user_id, category_id,
                            extracted, image_b64, mime_type, filename)
                imported += 1

                # Polite pause between API calls
                time.sleep(0.5)

            except json.JSONDecodeError:
                print(f"{prefix}  →  WARNING: GPT-4o returned non-JSON, skipping")
                failed += 1
            except Exception as e:
                print(f"{prefix}  →  ERROR: {e}")
                failed += 1

    print()
    print(f"Done. Imported: {imported}  |  Skipped: {skipped}  |  Failed: {failed}")
    print()
    if imported > 0:
        print(f"Open http://127.0.0.1:5000/dashboard?category=vinyl_album to review your albums.")


if __name__ == "__main__":
    main()
