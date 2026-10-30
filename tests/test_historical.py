"""Offline tests for historical traffic ingestion."""

import argparse
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import call, patch
from urllib.error import HTTPError

from traffic_prediction.data import historical as h


def make_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer, "w", compression=zipfile.ZIP_STORED
    ) as archive:
        archive.writestr("traffic.txt", b"iu_ac;q;k\n799;151;0.9\n")
    return buffer.getvalue()


def make_deflate64_zip() -> bytes:
    """Real Deflate64 fixture: two traffic files, including a 70 KB field.

    Generated once with inflate64.Deflater; no compressor is needed in tests.
    Rows are Convention at 2025-01-01 00:00 (q=200) and 01:00 (q=300).
    """
    import base64

    return base64.b64decode(
        "UEsDBBUAAAAJAAAAAAATmu6UhgAAAAgSAQAMAAAAYm91bmRhcnkudHh07cHBDYMwDAXQO1N0gUgOiFbi38q1"
        "O1gGXDVqcNRgEON3Ed5LO8uMnCbNWdkWlrWYI6dJc1a2heWQDOf4wQ9fqIuzV3mnGeriPEmtikW3LVnT3x8R"
        "L53LXicNY7FDzVMxjMUONU/FwjOXVR0ttX2gGCjeiAaigQgtETpEdDiv+3/nFReh+QNQSwMEFQAAAAkAAAAA"
        "AA8Fb818AAAAmAAAAAgAAABuZXh0LnR4dAXBQQ6CQAwF0L2n8AKTdEA04e9k6x2aDtQ4sXQiFM7ve/VgmWG1"
        "qJmyLyxr84DVombKvrCcYgjOH/zwhYYExybvOkNDgotsm2LRfa9+Ge6PjJfO7diKpqn5qR61Oabmp3rU5ulp"
        "bdVAR92QKCfKV8oj0UiEngg3ZPS4/AFQSwECFQAVAAAACQAAAAAAE5rulIYAAAAIEgEADAAAAAAAAAAAAAAA"
        "AAAAAAAAYm91bmRhcnkudHh0UEsBAhUAFQAAAAkAAAAAAA8Fb818AAAAmAAAAAgAAAAAAAAAAAAAAAAAsAAA"
        "AG5leHQudHh0UEsFBgAAAAACAAIAcAAAAFIBAAAAAA=="
    )


class FakeResponse(io.BytesIO):
    def __init__(self, content: bytes, *, interrupted: bool = False):
        super().__init__(content)
        self.status = 200
        self.headers = {"Content-Length": str(len(content))}
        self.interrupted = interrupted
        self.read_sizes = []

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if self.interrupted:
            if self.tell() > 0:
                raise TimeoutError("Connection interrupted.")
            return super().read(8)
        return super().read(size)


class HistoricalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="traffic-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.archive = h.Archive(
            2024,
            "opendata_txt_2024_zip",
            "https://example.test/2024",
        )
        self.destination = (
            self.root / "2024" / "opendata_txt_2024_zip.zip"
        )

        network = patch.object(
            h,
            "urlopen",
            side_effect=AssertionError("Unexpected network request."),
        )
        self.http = network.start()
        self.addCleanup(network.stop)

        sleeping = patch.object(h.time, "sleep")
        self.sleep = sleeping.start()
        self.addCleanup(sleeping.stop)

    def assert_no_partial_files(self):
        self.assertEqual(list(self.root.rglob("*.part")), [])

    def test_discovery_preserves_duplicate_years(self):
        metadata = {
            "attachments": [
                {"id": "notice", "title": "notice.pdf"},
                {
                    "id": "opendata_txt_2025_zip",
                    "title": "opendata_txt_2025.zip",
                },
                {
                    "id": "opendata_txt_2024_zip",
                    "title": "opendata_txt_2024.zip",
                },
                {
                    "id": "opendata_txt_2025_zip2",
                    "title": "opendata_txt_2025.zip",
                },
            ]
        }
        self.http.side_effect = [
            FakeResponse(json.dumps(metadata).encode())
        ]

        archives = h.discover_archives()

        self.assertEqual(
            [item.year for item in archives], [2024, 2025, 2025]
        )
        self.assertEqual(
            archives[0].url,
            h.DATASET_URL + "attachments/opendata_txt_2024_zip/",
        )
        self.assertNotEqual(archives[1].url, archives[2].url)

    def test_metadata_retries_then_stops(self):
        self.http.side_effect = TimeoutError("Offline.")

        with self.assertRaises(TimeoutError):
            h.discover_archives()

        self.assertEqual(self.http.call_count, 3)
        self.assertEqual(
            self.sleep.call_args_list, [call(1), call(2)]
        )

    def test_metadata_recovers_from_temporary_failures(self):
        metadata = {
            "attachments": [
                {
                    "id": self.archive.attachment_id,
                    "title": "opendata_txt_2024.zip",
                }
            ]
        }
        failures = [
            TimeoutError("Timeout."),
            FakeResponse(b'{"incomplete":'),
            HTTPError(
                h.DATASET_URL, 429, "Rate limited", {}, None
            ),
            HTTPError(
                h.DATASET_URL, 503, "Unavailable", {}, None
            ),
        ]

        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                self.http.reset_mock()
                self.http.side_effect = [
                    failure,
                    FakeResponse(json.dumps(metadata).encode()),
                ]

                archives = h.discover_archives()

                self.assertEqual(len(archives), 1)
                self.assertEqual(archives[0].year, 2024)
                self.assertEqual(self.http.call_count, 2)

    def test_metadata_404_is_not_retried(self):
        self.http.side_effect = HTTPError(
            h.DATASET_URL, 404, "Not found", {}, None
        )

        with self.assertRaises(HTTPError):
            h.discover_archives()

        self.assertEqual(self.http.call_count, 1)
        self.sleep.assert_not_called()

    def test_invalid_metadata_is_rejected(self):
        for metadata in [
            {},
            {"attachments": []},
            {"attachments": None},
        ]:
            with self.subTest(metadata=metadata):
                with self.assertRaises(ValueError):
                    h.parse_archives(metadata)

    def test_year_selection_is_inclusive_and_preserves_attachments(self):
        archives = [
            h.Archive(2023, "a", "https://example.test/a"),
            self.archive,
            h.Archive(2025, "b", "https://example.test/b"),
            h.Archive(2025, "c", "https://example.test/c"),
        ]

        self.assertEqual(list(h.parse_years("2024")), [2024])
        self.assertEqual(
            h.select_archives(
                archives, h.parse_years("2024-2025")
            ),
            archives[1:],
        )

        with self.assertRaises(ValueError):
            h.select_archives(
                archives, h.parse_years("2022-2024")
            )

        with self.assertRaises(argparse.ArgumentTypeError):
            h.parse_years("2025-2023")

    def test_download_uses_bounded_reads_and_saves_valid_zip(self):
        payload = make_zip()
        response = FakeResponse(payload)
        self.http.side_effect = [response]

        result = h.download_archive(self.archive, self.root)

        self.assertEqual(result, self.destination)
        self.assertEqual(result.read_bytes(), payload)
        self.assertTrue(response.read_sizes)
        self.assertTrue(
            all(
                size == 1024 * 1024
                for size in response.read_sizes
            )
        )
        h.validate_zip(result)
        self.assert_no_partial_files()

    def test_valid_existing_archive_is_not_downloaded(self):
        self.destination.parent.mkdir(parents=True)
        self.destination.write_bytes(make_zip())

        self.assertEqual(
            h.download_archive(self.archive, self.root),
            self.destination,
        )
        self.http.assert_not_called()

    def test_interrupted_download_restarts_successfully(self):
        self.http.side_effect = [
            FakeResponse(make_zip(), interrupted=True),
            FakeResponse(make_zip()),
        ]

        result = h.download_archive(self.archive, self.root)

        self.assertEqual(self.http.call_count, 2)
        h.validate_zip(result)
        self.assert_no_partial_files()

    def test_exhausted_download_leaves_no_final_or_partial_file(self):
        self.http.side_effect = [
            FakeResponse(make_zip(), interrupted=True)
            for _ in range(3)
        ]

        with self.assertRaises(TimeoutError):
            h.download_archive(self.archive, self.root)

        self.assertEqual(self.http.call_count, 3)
        self.assertFalse(self.destination.exists())
        self.assert_no_partial_files()

    def test_incomplete_download_is_rejected(self):
        response = FakeResponse(make_zip())
        response.headers["Content-Length"] = str(
            len(make_zip()) + 100
        )
        self.http.side_effect = [response]

        with self.assertRaisesRegex(
            ValueError, "Incomplete download"
        ):
            h.download_archive(
                self.archive, self.root, attempts=1
            )

        self.assertFalse(self.destination.exists())
        self.assert_no_partial_files()

    def test_crc_corruption_is_rejected(self):
        payload = bytearray(make_zip())
        payload[payload.index(b"iu_ac")] ^= 1
        self.http.side_effect = [
            FakeResponse(bytes(payload))
        ]

        with self.assertRaises(ValueError):
            h.download_archive(
                self.archive, self.root, attempts=1
            )

        self.assertFalse(self.destination.exists())
        self.assert_no_partial_files()

    def test_duplicate_year_downloads_use_distinct_files(self):
        first = h.Archive(
            2025, "opendata_txt_2025_zip", self.archive.url
        )
        second = h.Archive(
            2025, "opendata_txt_2025_zip2", self.archive.url
        )
        self.http.side_effect = [
            FakeResponse(make_zip()),
            FakeResponse(make_zip()),
        ]

        first_path = h.download_archive(first, self.root)
        second_path = h.download_archive(second, self.root)

        self.assertNotEqual(first_path, second_path)
        self.assertTrue(
            first_path.exists() and second_path.exists()
        )

    def test_download_cli_requires_years_before_network(self):
        with patch(
            "sys.argv", ["historical.py", "--download"]
        ):
            with self.assertRaises(SystemExit) as error:
                h.main()

        self.assertEqual(error.exception.code, 2)
        self.http.assert_not_called()


    def test_deflate64_download_validates_and_is_reused(self):
        self.http.side_effect = [FakeResponse(make_deflate64_zip())]
        result = h.download_archive(self.archive, self.root, attempts=1)
        h.validate_zip(result)
        self.assertEqual(result.read_bytes(), make_deflate64_zip())
        self.assertEqual(h.download_archive(self.archive, self.root), result)
        self.assertEqual(self.http.call_count, 1)
        self.assert_no_partial_files()

    def test_deflate64_local_crc_failure_is_rejected(self):
        payload = bytearray(make_deflate64_zip())
        payload[14] ^= 1  # CRC in the first local header.
        path = self.root / "bad.zip"
        path.write_bytes(payload)
        with self.assertRaisesRegex(ValueError, "CRC32"):
            h.validate_zip(path)

    def test_deflate64_central_crc_failure_is_rejected(self):
        payload = bytearray(make_deflate64_zip())
        offset = payload.index(b"PK\x01\x02") + 16
        payload[offset] ^= 1
        path = self.root / "bad.zip"
        path.write_bytes(payload)
        with self.assertRaisesRegex(ValueError, "CRC mismatch"):
            h.validate_zip(path)

    def test_deflate64_missing_decoder_has_install_instructions(self):
        path = self.root / "deflate64.zip"
        path.write_bytes(make_deflate64_zip())
        with patch.dict("sys.modules", {"stream_unzip": None}):
            with self.assertRaisesRegex(ValueError, "pip install stream-unzip"):
                h.validate_zip(path)

    def test_standard_zip_does_not_require_optional_decoder(self):
        path = self.root / "ordinary.zip"
        path.write_bytes(make_zip())
        with patch.dict("sys.modules", {"stream_unzip": None}):
            h.validate_zip(path)

    def test_truncated_deflate64_archive_is_rejected(self):
        path = self.root / "truncated.zip"
        path.write_bytes(make_deflate64_zip()[:-15])
        with self.assertRaisesRegex(ValueError, "Invalid"):
            h.validate_zip(path)


if __name__ == "__main__":
    unittest.main()
