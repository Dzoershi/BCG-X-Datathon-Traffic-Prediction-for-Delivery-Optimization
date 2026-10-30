"""Offline tests for recent traffic ingestion."""

import argparse
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

from traffic_prediction.data import traffic as t


def observation(arc="4256", timestamp="2026-10-29T22:00:00Z"):
    return {
        "iu_ac": arc,
        "t_1h": timestamp,
        "q": None,
        "k": 0.0,
        "geo_point_2d": {"lat": 48.8, "lon": 2.3},
    }


def json_lines(*rows):
    return b"".join(
        json.dumps(row).encode() + b"\n" for row in rows
    )


class FakeResponse(io.BytesIO):
    def __init__(self, content, *, interrupted=False):
        super().__init__(content)
        self.status = 200
        self.headers = {"Content-Length": str(len(content))}
        self.interrupted = interrupted
        self.line_sizes = []

    def readline(self, size=-1):
        self.line_sizes.append(size)
        if self.interrupted and self.tell() > 0:
            raise TimeoutError("Connection interrupted.")
        return super().readline(size)


def sample_response(count):
    payload = {
        "total_count": count,
        "results": [observation()] if count else [],
    }
    return FakeResponse(json.dumps(payload).encode())


class TrafficTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(
            prefix="traffic-api-test-"
        )
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.start = t.parse_timestamp("2026-10-29T22:00:00Z")
        self.end = t.parse_timestamp("2026-10-29T23:00:00Z")

        network = patch.object(
            t,
            "urlopen",
            side_effect=AssertionError(
                "Unexpected network request."
            ),
        )
        self.http = network.start()
        self.addCleanup(network.stop)

        sleeping = patch.object(t.time, "sleep")
        self.sleep = sleeping.start()
        self.addCleanup(sleeping.stop)

    def download(self, *, attempts=1):
        return t.download_observations(
            self.start,
            self.end,
            self.root,
            attempts=attempts,
        )

    def assert_no_files(self):
        self.assertEqual(list(self.root.iterdir()), [])

    def test_timestamp_parsing_and_range_validation(self):
        self.assertEqual(
            t.parse_timestamp("2026-10-29T23:00:00+01:00"),
            self.start,
        )
        for value in [
            "invalid",
            "2026-10-29",
            "2026-10-29T22:00:00",
        ]:
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    t.parse_timestamp(value)

        for start, end in [
            (self.start, None),
            (None, self.end),
            (self.end, self.start),
            (self.start, self.start),
        ]:
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    t.build_date_filter(start, end)

    def test_date_filter_has_inclusive_start_and_exclusive_end(self):
        self.assertEqual(
            t.build_date_filter(self.start, self.end),
            "t_1h >= date'2026-10-29T22:00:00Z' "
            "AND t_1h < date'2026-10-29T23:00:00Z'",
        )

    def test_preview_preserves_raw_values_and_sends_filter(self):
        self.http.side_effect = [sample_response(2)]

        payload = t.fetch_sample(
            1, start=self.start, end=self.end
        )

        self.assertEqual(payload["total_count"], 2)
        self.assertEqual(payload["results"], [observation()])
        query = parse_qs(
            urlparse(self.http.call_args.args[0].full_url).query
        )
        self.assertEqual(
            query["where"],
            [t.build_date_filter(self.start, self.end)],
        )

    def test_incomplete_preview_is_rejected(self):
        self.http.side_effect = [sample_response(2)]
        with self.assertRaises(ValueError):
            t.fetch_sample(2, start=self.start, end=self.end)

    def test_complete_export_preserves_bytes_and_uses_bulk_endpoint(self):
        raw = json_lines(observation(), observation("6672"))
        response = FakeResponse(raw)
        self.http.side_effect = [
            sample_response(2),
            response,
            sample_response(2),
        ]

        path = self.download()

        self.assertEqual(path.read_bytes(), raw)
        self.assertEqual(path.suffix, ".jsonl")
        self.assertEqual(list(self.root.glob("*.part")), [])
        self.assertTrue(response.line_sizes)
        self.assertTrue(
            all(
                size == 4 * 1024 * 1024 + 1
                for size in response.line_sizes
            )
        )

        export_url = (
            self.http.call_args_list[1].args[0].full_url
        )
        self.assertTrue(
            urlparse(export_url).path.endswith("/exports/jsonl")
        )
        self.assertEqual(
            parse_qs(urlparse(export_url).query)["limit"],
            ["-1"],
        )
        self.assertEqual(self.http.call_count, 3)

    def test_empty_period_creates_empty_snapshot(self):
        self.http.side_effect = [
            sample_response(0),
            FakeResponse(b""),
            sample_response(0),
        ]
        self.assertEqual(self.download().read_bytes(), b"")

    def test_bad_exports_are_rejected_and_cleaned_up(self):
        cases = [
            ("too few rows", json_lines(observation())),
            ("broken JSON", b'{"iu_ac":'),
            (
                "missing fields",
                json_lines({"error": "bad response"}),
            ),
            (
                "exclusive end",
                json_lines(observation(
                    timestamp="2026-10-29T23:00:00Z"
                )),
            ),
            (
                "before start",
                json_lines(observation(
                    timestamp="2026-10-29T21:00:00Z"
                )),
            ),
            (
                "invalid timestamp",
                json_lines(observation(timestamp="invalid")),
            ),
        ]

        for label, content in cases:
            with self.subTest(case=label):
                self.http.side_effect = [
                    sample_response(2),
                    FakeResponse(content),
                ]
                with self.assertRaises(ValueError):
                    self.download()
                self.assert_no_files()

    def test_content_length_mismatch_is_rejected(self):
        response = FakeResponse(json_lines(observation()))
        response.headers["Content-Length"] = "999999"
        self.http.side_effect = [
            sample_response(1), response
        ]

        with self.assertRaises(ValueError):
            self.download()
        self.assert_no_files()

    def test_changed_source_count_is_rejected(self):
        self.http.side_effect = [
            sample_response(1),
            FakeResponse(json_lines(observation())),
            sample_response(2),
        ]

        with self.assertRaisesRegex(
            ValueError, "Source count changed"
        ):
            self.download()
        self.assert_no_files()

    def test_temporary_request_failures_are_retried(self):
        failures = [
            TimeoutError("Timeout."),
            HTTPError(
                t.RECORDS_URL, 429, "Rate limited", {}, None
            ),
            HTTPError(
                t.RECORDS_URL, 503, "Unavailable", {}, None
            ),
        ]

        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                self.http.reset_mock()
                self.http.side_effect = [
                    failure,
                    sample_response(1),
                    FakeResponse(json_lines(observation())),
                    sample_response(1),
                ]

                self.assertTrue(
                    self.download(attempts=2).exists()
                )
                self.assertEqual(self.http.call_count, 4)

    def test_interrupted_export_restarts_without_duplicate_rows(self):
        raw = json_lines(observation(), observation("6672"))
        self.http.side_effect = [
            sample_response(2),
            FakeResponse(raw, interrupted=True),
            sample_response(2),
            FakeResponse(raw),
            sample_response(2),
        ]

        path = self.download(attempts=2)

        self.assertEqual(path.read_bytes(), raw)
        self.assertEqual(list(self.root.glob("*.part")), [])

    def test_exhausted_retries_leave_no_files(self):
        responses = []
        for _ in range(3):
            responses.extend([
                sample_response(2),
                FakeResponse(
                    json_lines(
                        observation(), observation("6672")
                    ),
                    interrupted=True,
                ),
            ])
        self.http.side_effect = responses

        with self.assertRaises(TimeoutError):
            self.download(attempts=3)

        self.assertEqual(self.http.call_count, 6)
        self.assert_no_files()

    def test_permanent_http_failure_is_not_retried(self):
        self.http.side_effect = HTTPError(
            t.RECORDS_URL, 404, "Not found", {}, None
        )

        with self.assertRaises(HTTPError):
            self.download(attempts=3)

        self.assertEqual(self.http.call_count, 1)
        self.sleep.assert_not_called()
        self.assert_no_files()

    def test_export_can_exceed_records_pagination_limit(self):
        raw = json_lines(observation()) * 10001
        self.http.side_effect = [
            sample_response(10001),
            FakeResponse(raw),
            sample_response(10001),
        ]

        self.assertEqual(self.download().read_bytes(), raw)

    def test_repeated_runs_keep_separate_snapshots(self):
        paths = []
        for _ in range(2):
            self.http.side_effect = [
                sample_response(1),
                FakeResponse(json_lines(observation())),
                sample_response(1),
            ]
            paths.append(self.download())

        self.assertNotEqual(paths[0], paths[1])
        self.assertTrue(
            all(path.exists() for path in paths)
        )

    def test_cli_requires_dates_before_downloading(self):
        with patch("sys.argv", ["traffic.py", "--download"]):
            with self.assertRaises(SystemExit) as error:
                t.main()

        self.assertEqual(error.exception.code, 2)
        self.http.assert_not_called()


    def test_cli_help_without_processing_dependencies(self):
        import builtins
        import runpy

        original_import = builtins.__import__

        def block_processing_imports(name, *args, **kwargs):
            if name.split(".")[0] in {"numpy", "pandas", "yaml", "pyarrow"}:
                raise ModuleNotFoundError(f"Blocked processing dependency: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=block_processing_imports):
            with patch("sys.argv", ["traffic.py", "--help"]):
                with self.assertRaises(SystemExit) as error:
                    runpy.run_path(t.__file__, run_name="__main__")

        self.assertEqual(error.exception.code, 0)
        self.http.assert_not_called()


    def test_single_road_filter_is_reused_for_export_and_count_checks(self):
        raw = json_lines(observation(arc="5671"))
        self.http.side_effect = [sample_response(1), FakeResponse(raw), sample_response(1)]
        result = t.download_observations(
            self.start, self.end, self.root, road_id=5671, attempts=1
        )
        self.assertEqual(result.read_bytes(), raw)
        expected = t.build_observation_filter(self.start, self.end, 5671)
        for request in self.http.call_args_list:
            parameters = parse_qs(urlparse(request.args[0].full_url).query)
            self.assertEqual(parameters["where"], [expected])

    def test_filtered_export_rejects_another_road(self):
        self.http.side_effect = [sample_response(1), FakeResponse(json_lines(observation()))]
        with self.assertRaisesRegex(ValueError, "requested road"):
            t.download_observations(
                self.start, self.end, self.root, road_id=5671, attempts=1
            )
        self.assert_no_files()

    def test_invalid_road_filter_fails_before_network(self):
        for road_id in [True, 0, -1, "5671"]:
            with self.subTest(road_id=road_id):
                with self.assertRaises(ValueError):
                    t.download_observations(
                        self.start, self.end, self.root, road_id=road_id, attempts=1
                    )
        self.http.assert_not_called()


if __name__ == "__main__":
    unittest.main()
