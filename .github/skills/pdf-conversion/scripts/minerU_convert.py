#!/usr/bin/env python3
"""Convert a PDF to GeneralExplorer paper artifacts via MinerU online API."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import requests

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None


API_FILE_URLS_BATCH = "https://mineru.net/api/v4/file-urls/batch"
API_EXTRACT_RESULTS_BATCH = "https://mineru.net/api/v4/extract-results/batch/{batch_id}"
DEFAULT_POLL_INTERVAL = 10
DEFAULT_MAX_POLLS = 180
JPEG_QUALITY = 75
WEBP_QUALITY = 75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PDF into paper.raw.md / paper.md / paper.references.md using MinerU online API."
    )
    parser.add_argument("--pdf-path", required=True, help="Input PDF path")
    parser.add_argument("--output-dir", required=True, help="Paper directory or temp output directory")
    parser.add_argument("--token", default="", help="MinerU API token; defaults to assets/mineru.key or env")
    parser.add_argument("--lang", default="en", help="Document language")
    parser.add_argument(
        "--model-version",
        default="vlm",
        choices=["vlm", "pipeline", "MinerU-HTML"],
        help="MinerU model version",
    )
    parser.add_argument("--page-ranges", default="", help="Optional page ranges, e.g. 1-20")
    parser.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL)
    parser.add_argument("--max-polls", type=int, default=DEFAULT_MAX_POLLS)
    parser.add_argument("--disable-image-compression", action="store_true")
    return parser.parse_args()


def resolve_paths(pdf_path: str, output_dir: str) -> tuple[Path, Path]:
    pdf = Path(pdf_path).resolve()
    if not pdf.exists() or not pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf}")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    return pdf, output


def load_token(explicit_token: str) -> str:
    if explicit_token.strip():
        return explicit_token.strip()

    env_token = os.environ.get("MINERU_API_TOKEN", "").strip()
    if env_token:
        return env_token

    key_path = Path(__file__).resolve().parent.parent / "assets" / "mineru.key"
    if key_path.exists():
        token = key_path.read_text(encoding="utf-8").strip()
        if token:
            return token

    raise RuntimeError(
        "MinerU token not found. Provide --token, set MINERU_API_TOKEN, or place it in .github/skills/paper-recommend/assets/mineru.key"
    )


def auth_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "*/*",
    }


def apply_upload_url(pdf: Path, token: str, args: argparse.Namespace) -> str:
    payload: dict[str, object] = {
        "files": [{"name": pdf.name}],
        "model_version": args.model_version,
        "language": args.lang,
        "enable_formula": True,
        "enable_table": True,
    }
    if args.page_ranges:
        payload["files"][0]["page_ranges"] = args.page_ranges

    response = requests.post(API_FILE_URLS_BATCH, headers=auth_headers(token), json=payload, timeout=60)
    response.raise_for_status()
    result = response.json()
    if result.get("code") != 0:
        raise RuntimeError(f"Failed to apply upload URL: {json.dumps(result, ensure_ascii=False)}")

    batch_id = result["data"]["batch_id"]
    upload_url = result["data"]["file_urls"][0]

    with pdf.open("rb") as handle:
        upload_response = requests.put(upload_url, data=handle, timeout=600)
        upload_response.raise_for_status()

    return batch_id


def poll_result(batch_id: str, token: str, args: argparse.Namespace) -> dict[str, object]:
    url = API_EXTRACT_RESULTS_BATCH.format(batch_id=batch_id)
    headers = {"Authorization": f"Bearer {token}", "Accept": "*/*"}

    for attempt in range(1, args.max_polls + 1):
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        result = response.json()
        if result.get("code") != 0:
            raise RuntimeError(f"Polling failed: {json.dumps(result, ensure_ascii=False)}")

        extract_result = result.get("data", {}).get("extract_result", [])
        if extract_result:
            first = extract_result[0]
            state = str(first.get("state", "unknown"))
            print(f"[online] poll {attempt}/{args.max_polls}: {state}")
            if state == "done":
                return first
            if state == "failed":
                raise RuntimeError(f"MinerU task failed: {first.get('err_msg', '')}")

        time.sleep(args.poll_interval)

    raise TimeoutError("MinerU polling timed out")


def download_zip_bytes(url: str) -> bytes:
    command = ["curl.exe", "--http1.1", "-L", url]
    result = subprocess.run(command, capture_output=True, timeout=600)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"curl download failed: {stderr[:400]}")
    return result.stdout


def split_references(raw_text: str) -> tuple[str, str]:
    reference_markers = [
        r"\n#{1,6}\s*References\b",
        r"\nREFERENCES\b",
        r"\n参考文献\b",
    ]

    reference_match = None
    for pattern in reference_markers:
        reference_match = re.search(pattern, raw_text, flags=re.IGNORECASE)
        if reference_match:
            break

    if reference_match:
        body = raw_text[: reference_match.start()].strip()
        refs = raw_text[reference_match.start() :].strip()
    else:
        body = raw_text.strip()
        refs = ""

    appendix_match = re.search(r"\n#{1,6}\s*Appendix\b|\nAPPENDIX\b", body, flags=re.IGNORECASE)
    if appendix_match:
        body = body[: appendix_match.start()].strip()

    return body, refs


def compress_images_in_dir(images_dir: Path) -> tuple[int, int, int]:
    if Image is None or not images_dir.exists():
        return 0, 0, 0

    handled = 0
    compressed = 0
    saved_bytes = 0

    for image_path in images_dir.rglob("*"):
        if not image_path.is_file():
            continue

        ext = image_path.suffix.lower()
        if ext not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue

        handled += 1
        original_size = image_path.stat().st_size
        tmp_path = image_path.with_suffix(image_path.suffix + ".tmp")

        try:
            with Image.open(image_path) as image:
                if ext == ".png":
                    image.save(tmp_path, optimize=True, compress_level=9)
                elif ext in {".jpg", ".jpeg"}:
                    rgb_image = image.convert("RGB")
                    rgb_image.save(tmp_path, "JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)
                else:
                    image.save(tmp_path, "WEBP", quality=WEBP_QUALITY, method=6)

            tmp_size = tmp_path.stat().st_size
            if tmp_size < original_size:
                tmp_path.replace(image_path)
                compressed += 1
                saved_bytes += original_size - tmp_size
            else:
                tmp_path.unlink(missing_ok=True)
        except Exception:
            tmp_path.unlink(missing_ok=True)

    return handled, compressed, saved_bytes


def write_outputs(zip_bytes: bytes, output_dir: Path, pdf: Path, compress_images: bool) -> None:
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as archive:
        names = archive.namelist()
        full_md_name = next((name for name in names if name.endswith("/full.md") or name == "full.md"), None)
        if not full_md_name:
            raise FileNotFoundError("full.md not found in MinerU result ZIP")

        raw_text = archive.read(full_md_name).decode("utf-8", errors="ignore")
        paper_root = output_dir
        raw_path = paper_root / "paper.raw.md"
        paper_path = paper_root / "paper.md"
        refs_path = paper_root / "paper.references.md"

        body_text, refs_text = split_references(raw_text)
        raw_path.write_text(raw_text.strip() + "\n", encoding="utf-8")
        paper_path.write_text(body_text + "\n", encoding="utf-8")
        refs_path.write_text((refs_text + "\n") if refs_text else "", encoding="utf-8")

        images_prefixes = sorted({name.split("images/")[0] + "images/" for name in names if "images/" in name})
        if images_prefixes:
            target_images_dir = paper_root / "images"
            target_images_dir.mkdir(parents=True, exist_ok=True)
            for name in names:
                if "images/" not in name or name.endswith("/"):
                    continue
                relative_name = name.split("images/", 1)[1]
                destination = target_images_dir / relative_name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(name))

            if compress_images:
                handled, compressed, saved = compress_images_in_dir(target_images_dir)
                print(f"image compression: handled={handled} compressed={compressed} saved_bytes={saved}")

    pdf_target = output_dir / "paper.pdf"
    if pdf.resolve() != pdf_target.resolve():
        pdf_target.write_bytes(pdf.read_bytes())


def main() -> int:
    args = parse_args()
    pdf, output_dir = resolve_paths(args.pdf_path, args.output_dir)
    token = load_token(args.token)

    print("[1/4] applying upload URL")
    batch_id = apply_upload_url(pdf, token, args)
    print(f"batch_id: {batch_id}")

    print("[2/4] polling MinerU result")
    result = poll_result(batch_id, token, args)

    print("[3/4] downloading ZIP")
    zip_bytes = download_zip_bytes(result["full_zip_url"])

    print("[4/4] writing outputs")
    write_outputs(zip_bytes, output_dir, pdf, compress_images=not args.disable_image_compression)

    print("done")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"conversion failed: {exc}", file=sys.stderr)
        raise SystemExit(1)