import io
import json
import stat
import subprocess
import zipfile

import httpx
import pytest

from h3ltx.client import connection, pull_runs, unpack
from h3ltx.records import output_metadata


def test_pull_cursor_atomic_and_idempotent(tmp_path):
    identifier = "a" * 32
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("result.json", '{"state":"succeeded"}')

    def handler(request):
        if request.url.path == "/v1/runs/export":
            cursor = int(request.url.params["since_cursor"])
            return httpx.Response(
                200,
                json={
                    "runs": [{"id": identifier, "cursor": 1}] if cursor == 0 else [],
                    "cursor": 1,
                },
            )
        return httpx.Response(200, content=archive.getvalue())

    with httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://runtime.test"
    ) as client:
        assert pull_runs(client, tmp_path) == 1
        assert pull_runs(client, tmp_path) == 0
        assert (tmp_path / identifier / "result.json").is_file()
        assert json.loads((tmp_path / ".h3ltx-cursor.json").read_text())["cursor"] == 1
    with (
        httpx.Client(
            transport=httpx.MockTransport(handler), base_url="https://other.test"
        ) as client,
        pytest.raises(ValueError, match="different server"),
    ):
        pull_runs(client, tmp_path)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "safe/../../../escape"])
def test_zip_traversal(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(name, "bad")
    with pytest.raises(ValueError):
        unpack(archive, tmp_path / "output")


def test_zip_symlink(tmp_path):
    archive = tmp_path / "bad.zip"
    info = zipfile.ZipInfo("link")
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(info, "../escape")
    with pytest.raises(ValueError):
        unpack(archive, tmp_path / "output")


def test_local_dotenv_without_overriding_environment(tmp_path, monkeypatch):
    # This creates a disposable test file, never reads the owner's repository .env.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("API_TOKEN", raising=False)
    monkeypatch.delenv("H3LTX_TOKEN", raising=False)
    (tmp_path / ".env").write_text("API_TOKEN=test-placeholder\nH3LTX_URL=https://local.test\n")
    monkeypatch.setenv("H3LTX_URL", "https://exported.test")
    with connection() as client:
        assert str(client.base_url) == "https://exported.test"
        assert client.headers["Authorization"] == "Bearer test-placeholder"


def test_real_ffmpeg_metadata(tmp_path):
    path = tmp_path / "video.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=size=128x128:rate=24",
            "-t",
            "1",
            "-c:v",
            "libx264",
            str(path),
        ],
        check=True,
    )
    result = output_metadata(path)
    assert result["duration_seconds"] == 1
    assert result["bytes"] > 0
    assert len(result["sha256"]) == 64
    assert (tmp_path / "thumbnail.jpg").is_file()
    assert (tmp_path / "contact-sheet.jpg").is_file()


async def test_url_upload_pins_ip_and_tls_hostname(tmp_path, monkeypatch):
    import h3ltx.uploads as module

    seen = []
    checked = []
    real_client = httpx.AsyncClient

    def resolve(url):
        checked.append(url)
        return (
            ("first.test", "93.184.216.34")
            if len(checked) == 1
            else ("second.test", "93.184.216.35")
        )

    def handler(request):
        seen.append(request)
        assert isinstance(request.extensions["sni_hostname"], str)
        if len(seen) == 1:
            return httpx.Response(302, headers={"Location": "https://second.test/image"})
        return httpx.Response(200, content=b"image")

    monkeypatch.setattr(module, "public_url", resolve)
    monkeypatch.setattr(
        module.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(
        module,
        "probe",
        lambda path: {
            "streams": [{"codec_type": "video", "codec_name": "png"}],
            "format": {"duration": "0"},
        },
    )
    result = await module.Uploads(tmp_path, 100).from_url("https://first.test/image")
    assert result["bytes"] == 5 and result["media_type"] == "image"
    assert len(checked) == 2
    assert seen[0].url.host == "93.184.216.34" and seen[0].headers["Host"] == "first.test"
    assert (
        seen[1].url.host == "93.184.216.35" and seen[1].extensions["sni_hostname"] == "second.test"
    )
