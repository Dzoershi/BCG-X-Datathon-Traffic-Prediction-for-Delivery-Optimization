"""Retrieve recent traffic observations from Paris Open Data."""

import argparse
import json
from http.client import HTTPException
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from datetime import datetime, timezone
import sys
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError

RECORDS_URL = (
    "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/"
    "comptages-routiers-permanents/records"
)


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO timestamp with an explicit timezone and convert to UTC."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.utcoffset() is None:
            raise ValueError("Timezone is missing.")
        return parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError) as error:
        raise argparse.ArgumentTypeError(
            "Use an ISO timestamp with Z or an offset, "
            "for example 2026-10-29T00:00:00Z."
        ) from error


def build_date_filter(
    start: datetime | None, end: datetime | None
) -> str | None:
    """Filter observation timestamps using inclusive start and exclusive end."""
    if start is None and end is None:
        return None

    if start is None or end is None:
        raise ValueError("Provide both --start and --end.")

    if start.utcoffset() is None or end.utcoffset() is None:
        raise ValueError("Date boundaries must include a timezone.")

    if start >= end:
        raise ValueError("--start must be earlier than --end.")

    lower = start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    upper = end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    return f"t_1h >= date'{lower}' AND t_1h < date'{upper}'"


def fetch_sample(
    limit: int = 3,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> dict:
    """Preview recent raw observations without writing files."""
    if not 1 <= limit <= 10:
        raise ValueError("Sample limit must be between 1 and 10.")

    parameters = {"limit": limit, "order_by": "t_1h DESC"}

    date_filter = build_date_filter(start, end)
    if date_filter is not None:
        parameters["where"] = date_filter

    query = urlencode(parameters)
    request = Request(
        f"{RECORDS_URL}?{query}",
        headers={
            "User-Agent": "traffic-prediction/0.1",
            "Accept-Encoding": "identity",
        },
    )

    with urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise ValueError(
                f"Unexpected HTTP status: {response.status}"
            )
        payload = json.load(response)

    if not isinstance(payload, dict):
        raise ValueError("API response must be an object.")

    total = payload.get("total_count")
    records = payload.get("results")

    if (
        type(total) is not int
        or total < 0
        or not isinstance(records, list)
    ):
        raise ValueError(
            "API response has invalid total_count or results."
        )

    if len(records) != min(limit, total):
        raise ValueError("API returned an incomplete sample.")

    if any(not isinstance(record, dict) for record in records):
        raise ValueError("API returned an invalid observation.")

    return payload


def validate_observation(
    record: object, start: datetime, end: datetime
) -> None:
    """Validate record structure and requested bounds without changing values."""
    required = {"iu_ac", "t_1h", "q", "k"}
    if not isinstance(record, dict) or not required.issubset(record):
        raise ValueError("Export contains an invalid traffic observation.")

    if not isinstance(record["t_1h"], str):
        raise ValueError("Export contains an invalid timestamp.")

    try:
        timestamp = parse_timestamp(record["t_1h"])
    except argparse.ArgumentTypeError as error:
        raise ValueError("Export contains an invalid timestamp.") from error

    if not start <= timestamp < end:
        raise ValueError(
            "Export contains an observation outside the requested period."
        )


def download_observations(
    start: datetime,
    end: datetime,
    output_dir: Path,
    *,
    attempts: int = 3,
) -> Path:
    """Save a complete raw JSONL snapshot, retrying temporary failures."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1.")

    date_filter = build_date_filter(start, end)
    if date_filter is None:
        raise ValueError("Downloads require start and end timestamps.")

    start = start.astimezone(timezone.utc)
    end = end.astimezone(timezone.utc)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = (
        f"traffic_{start:%Y%m%dT%H%M%S%fZ}_"
        f"{end:%Y%m%dT%H%M%S%fZ}_"
    )

    parameters = {
        "where": date_filter,
        "limit": -1,
        "timezone": "UTC",
    }
    export_url = RECORDS_URL.rsplit("/", 1)[0] + "/exports/jsonl"
    request = Request(
        f"{export_url}?{urlencode(parameters)}",
        headers={
            "User-Agent": "traffic-prediction/0.1",
            "Accept-Encoding": "identity",
        },
    )

    for attempt in range(1, attempts + 1):
        temporary_path = None

        try:
            expected = fetch_sample(
                1, start=start, end=end
            )["total_count"]

            print(
                f"Exporting {expected} observations "
                f"(attempt {attempt}/{attempts})...",
                flush=True,
            )

            with urlopen(request, timeout=60) as response:
                if response.status != 200:
                    raise ValueError(
                        f"Unexpected HTTP status: {response.status}"
                    )

                length = response.headers.get("Content-Length")
                expected_bytes = int(length) if length is not None else None
                received_bytes = 0
                count = 0

                with NamedTemporaryFile(
                    mode="wb",
                    dir=output_dir,
                    prefix=prefix,
                    suffix=".jsonl.part",
                    delete=False,
                ) as output:
                    temporary_path = Path(output.name)
                    max_line_bytes = 4 * 1024 * 1024

                    while line := response.readline(max_line_bytes + 1):
                        if len(line) > max_line_bytes:
                            raise ValueError(
                                "A JSONL record exceeds the 4 MiB limit."
                            )

                        if line.strip():
                            record = json.loads(line)
                            validate_observation(record, start, end)
                            count += 1

                        output.write(line)
                        received_bytes += len(line)

                        if line.strip() and count % 10000 == 0:
                            print(
                                f"Received {count} observations...",
                                flush=True,
                            )

                if (
                    expected_bytes is not None
                    and received_bytes != expected_bytes
                ):
                    raise ValueError(
                        "Export byte count does not match Content-Length."
                    )

            if count != expected:
                raise ValueError(
                    f"Incomplete export or changed source: "
                    f"expected {expected} observations, received {count}."
                )

            latest_count = fetch_sample(
                1, start=start, end=end
            )["total_count"]

            if latest_count != expected:
                raise ValueError(
                    "Source count changed during export; retry required."
                )

            destination = temporary_path.with_suffix("")
            temporary_path.replace(destination)

            print(
                f"Saved {count} observations: {destination}",
                flush=True,
            )
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
                f"Extraction failed: {error}; retrying...",
                file=sys.stderr,
            )
            time.sleep(2 ** (attempt - 1))

        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument(
        "--sample",
        action="store_true",
        help="Preview observations.",
    )
    actions.add_argument(
        "--download",
        action="store_true",
        help="Save a full raw export.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Preview size only.",
    )
    parser.add_argument(
        "--start",
        type=parse_timestamp,
        help="Inclusive timestamp.",
    )
    parser.add_argument(
        "--end",
        type=parse_timestamp,
        help="Exclusive timestamp.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parents[3]
            / "data"
            / "raw"
            / "recent"
        ),
    )
    args = parser.parse_args()

    if args.download and (
        args.start is None or args.end is None
    ):
        parser.error("--download requires both --start and --end.")

    try:
        if args.sample:
            payload = fetch_sample(
                args.limit,
                start=args.start,
                end=args.end,
            )
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            download_observations(
                args.start,
                args.end,
                args.output_dir,
            )

    except (OSError, ValueError, HTTPException) as error:
        parser.exit(1, f"Traffic ingestion failed: {error}\n")
    except KeyboardInterrupt:
        parser.exit(
            130,
            "Traffic ingestion interrupted; rerun to retry.\n",
        )
        

if __name__ == "__main__":
    main()