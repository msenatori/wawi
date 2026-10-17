"""Image lookup and offline board rendering, without network access."""
import base64
import json

import pytest

from wawi import images, tools
from wawi.board import Board, render_image_page

PNG = b"\x89PNG\r\n\x1a\n" + b"test-image"


def test_wikipedia_summary_and_commons_credit(monkeypatch):
    calls = []

    def request(url, limit):
        calls.append(url)
        if "wikipedia.org" in url:
            return json.dumps({"title": "Perro", "extract": "Mamífero doméstico.",
                               "thumbnail": {"source": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Dog.jpg/330px-Dog.jpg"},
                               "content_urls": {"desktop": {"page": "https://es.wikipedia.org/wiki/Perro"}}}).encode(), "application/json"
        if "commons.wikimedia.org/w/api.php" in url:
            return json.dumps({"query": {"pages": {"1": {"title": "File:Dog.jpg", "imageinfo": [{
                "thumburl": "https://upload.wikimedia.org/dog.png",
                "descriptionurl": "https://commons.wikimedia.org/wiki/File:Dog.jpg",
                "extmetadata": {"Artist": {"value": "<b>Ana</b>"},
                                "LicenseShortName": {"value": "CC BY-SA 4.0"}}}]}}}}).encode(), "application/json"
        assert url == "https://upload.wikimedia.org/dog.png"
        return PNG, "image/png"

    monkeypatch.setattr(images, "_request", request)
    card = images.find_image("perro")
    assert card.title == "Perro"
    assert card.description == "Mamífero doméstico."
    assert card.credit == "Ana" and card.license == "CC BY-SA 4.0"
    assert card.data_uri == "data:image/png;base64," + base64.b64encode(PNG).decode()
    assert len(calls) == 3


def test_page_og_image_relative_url_and_escaped_board(monkeypatch, tmp_path):
    def request(url, limit):
        if url == "https://example.org/dog":
            return (b'<meta property="og:image" content="/images/dog.png">'
                    b'<meta property="og:title" content="Dog &amp; me">'
                    b'<meta name="description" content="A friendly dog">'), "text/html"
        assert url == "https://example.org/images/dog.png"
        return PNG, "image/png"

    monkeypatch.setattr(images, "_request", request)
    card = images.find_image("", "https://example.org/dog", '<script>alert(1)</script>')
    board = Board(directory=tmp_path)
    page = board.show_image(card)
    assert "data:image/png;base64," in page.html
    assert "<script>alert(1)</script>" not in page.html
    assert "&lt;script&gt;" in page.html
    assert "image-card" in page.html and "max-width:620px" in page.html
    assert "Crédito: example.org" in page.html
    assert "https://example.org/dog" in page.html
    assert board.previous() is None


def test_fallback_to_english_then_commons(monkeypatch):
    def request(url, limit):
        if "es.wikipedia.org" in url:
            raise images.ImageError("not found")
        if "en.wikipedia.org" in url:
            return json.dumps({"title": "Eiffel Tower", "extract": "A tower in Paris.",
                               "thumbnail": {"source": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Eiffel.jpg/330px-Eiffel.jpg"}}).encode(), "application/json"
        if "commons.wikimedia.org/w/api.php" in url:
            return b'{}', "application/json"
        return b"\xff\xd8\xffphoto", "image/jpeg"

    monkeypatch.setattr(images, "_request", request)
    assert images.find_image("Eiffel Tower").description == "A tower in Paris."


@pytest.mark.parametrize("mime,body", [
    ("image/svg+xml", b"<svg><script>alert(1)</script></svg>"),
    ("text/html", b"<img src=x>"),
    ("image/png", b"not really a png"),
])
def test_rejects_unsafe_image_content(monkeypatch, mime, body):
    monkeypatch.setattr(images, "_request", lambda url, limit: (body, mime))
    with pytest.raises(images.ImageError):
        images._data_uri("https://example.org/image")


def test_request_rejects_oversize_and_private_url(monkeypatch):
    class Response:
        headers = {"Content-Length": "6000001", "Content-Type": "image/png"}
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def read(self, *_): raise AssertionError("must reject before download")

    monkeypatch.setattr(images.urllib.request.OpenerDirector, "open", lambda *_a, **_k: Response())
    with monkeypatch.context() as patch:
        patch.setattr(images, "check_url", lambda url: url)
        with pytest.raises(images.ImageError, match="demasiado grande"):
            images._request("https://example.org/photo.png", images.MAX_IMAGE_BYTES)
    with pytest.raises(images.ImageError):
        images._request("http://127.0.0.1/photo.png", images.MAX_IMAGE_BYTES)


def test_tool_is_conditional_and_metadata_is_not_sent_to_model(monkeypatch, tmp_path):
    old = tools._BOARD
    try:
        board = Board(directory=tmp_path)
        tools.set_board(board)
        spec = next(s for s in tools.BOARD_TOOLS if s["name"] == "show_image")
        assert spec["animation"] == "ship" and "perro" in spec["example"]
        model_spec = next(s for s in tools.active_tools() if s["name"] == "show_image")
        assert "animation" not in model_spec and "example" not in model_spec
        card = images.ImageCard("Dog", "Nice dog", "https://example.org", "Ana", "CC0",
                                "data:image/png;base64," + base64.b64encode(PNG).decode())
        monkeypatch.setattr(images, "find_image", lambda *args: card)
        result = tools.dispatch("show_image", {"query": "perro"})
        assert result.ok and result.speak == "Te la dejo en la pizarra."
        assert board.current.format == "image"
    finally:
        tools.set_board(old)


def test_not_found_is_short_spanish_error(monkeypatch, tmp_path):
    old = tools._BOARD
    try:
        tools.set_board(Board(directory=tmp_path))
        monkeypatch.setattr(images, "_wiki_info", lambda query: {})
        monkeypatch.setattr(images, "_commons_info", lambda query: {})
        result = tools.dispatch("show_image", {"query": "coso inexistente"})
        assert not result.ok and result.speak == "No encontré una foto de eso."
    finally:
        tools.set_board(old)
