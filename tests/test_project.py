import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProjectContractTests(unittest.TestCase):
    def test_sample_result_is_explicitly_labeled(self):
        for path in (ROOT / "results" / "benchmark.json", ROOT / "docs" / "benchmark.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["metadata"]["data_type"], "sample")
            self.assertIn("SYNTHETIC", data["metadata"].get("sample_disclaimer", "").upper())

    def test_sample_result_has_comparison_and_summary_data(self):
        data = json.loads((ROOT / "results" / "benchmark.json").read_text(encoding="utf-8"))
        summary = data["summary"]
        self.assertGreater(summary["throughput_tokens_per_sec"], 0)
        self.assertGreater(summary["latency_p50_ms"], 0)
        self.assertGreater(summary["latency_p90_ms"], summary["latency_p50_ms"])
        self.assertGreater(summary["latency_p99_ms"], summary["latency_p90_ms"])
        labels = [item["label"] for item in data["benchmarks"]]
        self.assertIn("7B / fp16", labels)
        self.assertIn("13B / int8", labels)

    def test_required_deliverable_files_exist(self):
        for relative in (
            "inference/benchmark.py",
            "dashboard/index.html",
            "dashboard/dashboard.js",
            "Dockerfile",
            "run.sh",
            "README.md",
            "docs/index.html",
            "docs/dashboard.html",
            ".github/workflows/pages.yml",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)


if __name__ == "__main__":
    unittest.main()
