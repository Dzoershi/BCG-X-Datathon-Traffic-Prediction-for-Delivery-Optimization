"""Discover, download, and validate historical Paris traffic archives."""

import argparse
import json
import re
from dataclasses import dataclass
from urllib.parse import quote
from urllib.request import Request, urlopen
import sys
import time
import zipfile
import zlib
from http.client import HTTPException
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError

DATASET_URL = (
    "https://opendata.paris.fr/api/datasets/1.0/"
    "comptages-routiers-permanents-historique/"
)


@dataclass(frozen=True)
class Archive:
    year: int
    attachment_id: str
    url: str


def parse_archives(metadata: dict) -> list[Archive]:
    """Read annual ZIP attachments, preserving multiple archives per year."""
    if not isinstance(metadata, dict):
        raise ValueError("Dataset metadata must be an object.")

    attachments = metadata.get("attachments")
    if not isinstance(attachments, list):
        raise ValueError("Dataset metadata is missing an attachments list.")

    archives = []
    seen_ids = set()

    for item in attachments:
        if not isinstance(item, dict):
            raise ValueError("Invalid attachment metadata.")

        title = item.get("title", "")
        if not isinstance(title, str):
            raise ValueError("Invalid attachment title.")

        match = re.fullmatch(r"opendata_txt_([0-9]{4})\.zip", title)
        if match is None:
            continue

        attachment_id = item.get("id")
        if not isinstance(attachment_id, str) or not attachment_id:
            raise ValueError(f"Missing attachment ID for {title}.")

        if attachment_id in seen_ids:
            raise ValueError(f"Duplicate attachment ID: {attachment_id}.")
        seen_ids.add(attachment_id)

        archives.append(
            Archive(
                year=int(match.group(1)),
                attachment_id=attachment_id,
                url=f"{DATASET_URL}attachments/{quote(attachment_id, safe='')}/",
            )
        )

    if not archives:
        raise ValueError("No recognized annual traffic ZIP archives found.")

    return sorted(
        archives,
        key=lambda archive: (archive.year, archive.attachment_id),
    )


def discover_archives(*, attempts: int = 3) -> list[Archive]:
    """Fetch metadata, retrying temporary request or response failures."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1.")

    request = Request(
        DATASET_URL,
        headers={
            "User-Agent": "traffic-prediction/0.1",
            "Accept-Encoding": "identity",
        },
    )

    for attempt in range(1, attempts + 1):
        try:
            with urlopen(request, timeout=30) as response:
                if response.status != 200:
                    raise ValueError(
                        f"Unexpected metadata HTTP status: {response.status}"
                    )
                metadata = json.load(response)

            return parse_archives(metadata)

        except (
            OSError,
            HTTPException,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ) as error:
            permanent_http_error = (
                isinstance(error, HTTPError)
                and 400 <= error.code < 500
                and error.code not in (408, 429)
            )

            if attempt == attempts or permanent_http_error:
                raise

            delay = 2 ** (attempt - 1)
            print(
                f"Metadata request failed (attempt {attempt}/{attempts}): "
                f"{error}; retrying in {delay}s...",
                file=sys.stderr,
            )
            time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument(
        "--list",
        action="store_true",
        help="List archive URLs.",
    )
    actions.add_argument(
        "--download",
        action="store_true",
        help="Download and validate archives.",
    )
    parser.add_argument(
        "--years",
        type=parse_years,
        metavar="YYYY[-YYYY]",
        help="One year or an inclusive range; required for downloads.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parents[3]
            / "data"
            / "raw"
            / "historical"
        ),
        help="Archive destination; default: repository data/raw/historical.",
    )

    args = parser.parse_args()

    if args.download and args.years is None:
        parser.error("--download requires --years.")

    try:
        archives = select_archives(discover_archives(), args.years)

        for archive in archives:
            if args.list:
                print(
                    f"{archive.year}\t{archive.attachment_id}\t{archive.url}"
                )
            else:
                download_archive(archive, args.output_dir)

    except (OSError, ValueError, HTTPException) as error:
        parser.exit(1, f"Historical ingestion failed: {error}\n")
    except KeyboardInterrupt:
        parser.exit(
            130,
            "Historical ingestion interrupted; rerun to retry.\n",
        )


def parse_years(value: str) -> range:
    """Parse one year or an inclusive range, e.g. 2024 or 2022-2024."""
    match = re.fullmatch(r"([0-9]{4})(?:-([0-9]{4}))?", value)
    if match is None:
        raise argparse.ArgumentTypeError("Use YYYY or YYYY-YYYY.")

    start = int(match.group(1))
    end = int(match.group(2) or match.group(1))
    if start > end:
        raise argparse.ArgumentTypeError(
            "The start year must not exceed the end year."
        )

    return range(start, end + 1)


def select_archives(
    archives: list[Archive], years: range | None
) -> list[Archive]:
    """Select all attachments for the requested years; reject missing years."""
    if years is None:
        return list(archives)

    requested = set(years)
    available = {archive.year for archive in archives}
    missing = sorted(requested - available)

    if missing:
        missing_text = ", ".join(str(year) for year in missing)
        raise ValueError(f"No archives available for: {missing_text}.")

    return [archive for archive in archives if archive.year in requested]


def validate_zip(path: Path) -> None:
    """Check archive contents and CRCs without extracting files."""
    try:
        with zipfile.ZipFile(path) as archive:
            if not any(
                not item.is_dir() and item.filename.lower().endswith(".txt")
                for item in archive.infolist()
            ):
                raise ValueError("Archive contains no traffic text files.")

            bad_file = archive.testzip()
            if bad_file is not None:
                raise ValueError(f"Corrupted ZIP member: {bad_file}")

    except (
        zipfile.BadZipFile,
        zlib.error,
        EOFError,
        RuntimeError,
        NotImplementedError,
    ) as error:
        raise ValueError(f"Invalid or unsupported ZIP: {error}") from error


def download_archive(
    archive: Archive,
    output_dir: Path,
    *,
    attempts: int = 3,
) -> Path:
    """Stream one ZIP to disk; retry failures from the beginning."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1.")

    if re.fullmatch(r"[A-Za-z0-9_-]+", archive.attachment_id) is None:
        raise ValueError("Unsafe attachment ID.")

    directory = Path(output_dir) / str(archive.year)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{archive.attachment_id}.zip"

    if destination.exists():
        print(f"Checking existing archive: {destination}", flush=True)
        try:
            validate_zip(destination)
        except ValueError as error:
            print(f"Existing archive is invalid: {error}", file=sys.stderr)
        else:
            print(f"Already valid; skipping: {destination}", flush=True)
            return destination

    for attempt in range(1, attempts + 1):
        temporary_path = None

        try:
            print(
                f"Downloading {archive.attachment_id} "
                f"(attempt {attempt}/{attempts})...",
                flush=True,
            )
            request = Request(
                archive.url,
                headers={
                    "User-Agent": "traffic-prediction/0.1",
                    "Accept-Encoding": "identity",
                },
            )

            with urlopen(request, timeout=60) as response:
                if response.status != 200:
                    raise ValueError(
                        f"Unexpected HTTP status: {response.status}"
                    )

                length = response.headers.get("Content-Length")
                expected = int(length) if length is not None else None
                received = 0

                with NamedTemporaryFile(
                    mode="wb",
                    dir=directory,
                    prefix=f".{archive.attachment_id}.",
                    suffix=".part",
                    delete=False,
                ) as output:
                    temporary_path = Path(output.name)

                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        received += len(chunk)

                if expected is not None and received != expected:
                    raise ValueError(
                        f"Incomplete download: expected {expected} bytes, "
                        f"received {received}."
                    )

            print(
                "Checking ZIP integrity; this may take a while...",
                flush=True,
            )
            validate_zip(temporary_path)
            temporary_path.replace(destination)

            print(f"Saved: {destination}", flush=True)
            return destination

        except (OSError, ValueError, HTTPException) as error:
            permanent_http_error = (
                isinstance(error, HTTPError)
                and 400 <= error.code < 500
                and error.code not in (408, 429)
            )
            if attempt == attempts or permanent_http_error:
                raise

            print(
                f"Attempt failed: {error}; retrying...",
                file=sys.stderr,
            )
            time.sleep(2 ** (attempt - 1))

        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()