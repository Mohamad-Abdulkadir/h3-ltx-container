import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def report_module():
    spec = importlib.util.spec_from_file_location("benchmark_report", ROOT / "scripts/report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_empty_benchmarks_are_not_fabricated():
    output = report_module().render([], {})
    assert "No measured warm generation runs" in output
    assert "No target GPU/driver/software records" in output


def test_report_reads_real_record_shape_and_separates_loads(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    documents = {
        "request": {"type": "h3.t2v"},
        "resolved": {
            "params": {"width": 1152, "height": 640, "duration_seconds": 5, "steps": 20},
            "profile": "blackwell-96",
            "warm": True,
            "gpu": {"name": "fake GPU", "driver": "test"},
            "software": {"torch": "test"},
        },
        "result": {
            "state": "succeeded",
            "warm": True,
            "wall_seconds": 60,
            "outputs": [{"name": "video.mp4", "duration_seconds": 5}],
        },
        "timeline": {"stages": {"sampling": 50}},
        "metrics": {"vram_bytes_peak": 1024**3, "ram_bytes_peak": 2 * 1024**3},
    }
    for key, value in documents.items():
        (root / (key + ".json")).write_text(json.dumps(value))
    report = report_module()
    records = report.load_records(tmp_path)
    text = report.render(records, {})
    assert "60.000 / 60.000" in text and "50.000" in text
    assert "fake GPU" in text and "video.mp4" in text
    documents["result"]["warm"] = False
    documents["resolved"]["warm"] = False
    records[0].update(documents)
    assert "No measured warm generation runs" in report.render(records, {})


def test_matrix_examples_resolve_all_requests():
    import itertools

    import yaml

    from h3ltx.params import JobRequest

    def resolve(value):
        if isinstance(value, str) and value.startswith("@asset:"):
            return "0" * 32
        if isinstance(value, list):
            return [resolve(v) for v in value]
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        return value

    for file in (ROOT / "examples").glob("*.yaml"):
        document = yaml.safe_load(file.read_text())
        for experiment in document["experiments"]:
            axes = experiment.get("vary", {})
            for values in itertools.product(*axes.values()):
                params = {**experiment.get("params", {}), **dict(zip(axes, values, strict=True))}
                JobRequest(type=experiment["type"], params=resolve(params)).resolve()
