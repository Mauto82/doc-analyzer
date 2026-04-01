"""
Tests for WebProcessor — URL-based content ingestion.
External libraries (trafilatura, youtube-transcript-api) are mocked.
"""
import pytest
from unittest.mock import MagicMock, patch

from src.processors.web_processor import WebProcessor, _extract_video_id
from src.models.document import Document


# ── _extract_video_id ─────────────────────────────────────────────────────────

class TestExtractVideoId:
    def test_watch_url(self):
        assert _extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_short_url(self):
        assert _extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_embed_url(self):
        assert _extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_v_url(self):
        assert _extract_video_id("https://www.youtube.com/v/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_watch_with_extra_params(self):
        assert _extract_video_id("https://youtube.com/watch?t=30&v=dQw4w9WgXcQ&list=PL") == "dQw4w9WgXcQ"

    def test_non_youtube_returns_none(self):
        assert _extract_video_id("https://example.com/page") is None

    def test_empty_string(self):
        assert _extract_video_id("") is None


# ── WebProcessor.process delegates to process_url ────────────────────────────

class TestProcessDelegation:
    def test_process_calls_process_url(self):
        wp = WebProcessor()
        wp.process_url = MagicMock(return_value=[])
        wp.process("https://example.com")
        wp.process_url.assert_called_once_with("https://example.com")


# ── _process_webpage ──────────────────────────────────────────────────────────

class TestProcessWebpage:
    def _make_wp(self):
        return WebProcessor()

    def test_returns_document_chunks(self):
        wp = self._make_wp()
        with patch("trafilatura.fetch_url", return_value="<html>raw</html>"), \
             patch("trafilatura.extract", return_value="Hello world from webpage"):
            chunks = wp.process_url("https://example.com/page")

        assert len(chunks) > 0
        assert all(isinstance(c, Document) for c in chunks)
        assert "Hello world" in chunks[0].page_content

    def test_metadata_source_and_type(self):
        wp = self._make_wp()
        with patch("trafilatura.fetch_url", return_value="<html>raw</html>"), \
             patch("trafilatura.extract", return_value="Content here"):
            chunks = wp.process_url("https://docs.example.com/api")

        assert chunks[0].metadata["source"] == "https://docs.example.com/api"
        assert chunks[0].metadata["type"] == "webpage"

    def test_fetch_failure_raises_value_error(self):
        wp = self._make_wp()
        with patch("trafilatura.fetch_url", return_value=None):
            with pytest.raises(ValueError, match="Could not fetch content"):
                wp.process_url("https://unreachable.example.com")

    def test_empty_extraction_raises_value_error(self):
        wp = self._make_wp()
        with patch("trafilatura.fetch_url", return_value="<html></html>"), \
             patch("trafilatura.extract", return_value=None):
            with pytest.raises(ValueError, match="No readable content"):
                wp.process_url("https://js-only.example.com")

    def test_whitespace_only_extraction_raises_value_error(self):
        wp = self._make_wp()
        with patch("trafilatura.fetch_url", return_value="<html> </html>"), \
             patch("trafilatura.extract", return_value="   \n  "):
            with pytest.raises(ValueError, match="No readable content"):
                wp.process_url("https://blank.example.com")

    def test_trafilatura_not_installed_raises_runtime_error(self):
        wp = self._make_wp()
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "trafilatura":
                raise ImportError("No module named 'trafilatura'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(RuntimeError, match="trafilatura is not installed"):
                wp._process_webpage("https://example.com")

    def test_large_text_produces_multiple_chunks(self):
        wp = WebProcessor()
        long_text = "word " * 600  # ~3000 chars, > default chunk_size 1000
        with patch("trafilatura.fetch_url", return_value="<html>raw</html>"), \
             patch("trafilatura.extract", return_value=long_text):
            chunks = wp.process_url("https://example.com/long")

        assert len(chunks) > 1


# ── _process_youtube ──────────────────────────────────────────────────────────

class _FakeSnippet:
    """Mimics a youtube-transcript-api v1.x FetchedTranscriptSnippet."""
    def __init__(self, text):
        self.text = text
        self.start = 0.0
        self.duration = 1.0


_FAKE_SEGMENTS = [_FakeSnippet("Hello"), _FakeSnippet("world"), _FakeSnippet("from YouTube")]


def _make_yt_modules(segments=None, error=None):
    """Return fake sys.modules entries for youtube_transcript_api v1.x."""

    class CouldNotRetrieveTranscript(Exception):
        pass

    mock_instance = MagicMock()
    if error:
        mock_instance.fetch.side_effect = error
    else:
        mock_instance.fetch.return_value = segments or _FAKE_SEGMENTS

    mock_cls = MagicMock(return_value=mock_instance)

    return {
        "youtube_transcript_api": MagicMock(YouTubeTranscriptApi=mock_cls),
        "youtube_transcript_api._errors": MagicMock(
            CouldNotRetrieveTranscript=CouldNotRetrieveTranscript
        ),
    }, CouldNotRetrieveTranscript


class TestProcessYoutube:
    _YT_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def _make_wp(self):
        return WebProcessor()

    def test_returns_document_chunks(self):
        wp = self._make_wp()
        mods, _ = _make_yt_modules()
        with patch.dict("sys.modules", mods):
            chunks = wp._process_youtube(self._YT_URL, "dQw4w9WgXcQ")

        assert len(chunks) > 0
        assert all(isinstance(c, Document) for c in chunks)

    def test_metadata_source_type_video_id(self):
        wp = self._make_wp()
        mods, _ = _make_yt_modules()
        with patch.dict("sys.modules", mods):
            chunks = wp._process_youtube(self._YT_URL, "dQw4w9WgXcQ")

        assert chunks[0].metadata["source"] == self._YT_URL
        assert chunks[0].metadata["type"] == "youtube"
        assert chunks[0].metadata["video_id"] == "dQw4w9WgXcQ"

    def test_transcript_text_joined(self):
        wp = self._make_wp()
        mods, _ = _make_yt_modules()
        with patch.dict("sys.modules", mods):
            chunks = wp._process_youtube(self._YT_URL, "dQw4w9WgXcQ")

        full = " ".join(c.page_content for c in chunks)
        assert "Hello" in full
        assert "world" in full

    def test_could_not_retrieve_raises_value_error(self):
        wp = self._make_wp()
        mods, CouldNotRetrieveTranscript = _make_yt_modules(
            error=None  # will be set below after we have the class
        )
        # Rebuild with the actual error class as the side_effect
        mods, CouldNotRetrieveTranscript = _make_yt_modules(
            error=mods["youtube_transcript_api._errors"].CouldNotRetrieveTranscript("disabled")
        )
        with patch.dict("sys.modules", mods):
            with pytest.raises(ValueError, match="No transcript available"):
                wp._process_youtube(self._YT_URL, "dQw4w9WgXcQ")

    def test_empty_transcript_raises_value_error(self):
        wp = self._make_wp()
        blank_segments = [_FakeSnippet("  ")]
        mods, _ = _make_yt_modules(segments=blank_segments)
        with patch.dict("sys.modules", mods):
            with pytest.raises(ValueError, match="Transcript is empty"):
                wp._process_youtube(self._YT_URL, "dQw4w9WgXcQ")

    def test_library_not_installed_raises_runtime_error(self):
        wp = self._make_wp()
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "youtube_transcript_api":
                raise ImportError("No module named 'youtube_transcript_api'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(RuntimeError, match="youtube-transcript-api is not installed"):
                wp._process_youtube(self._YT_URL, "dQw4w9WgXcQ")

    def test_process_url_routes_youtube(self):
        """process_url should call _process_youtube for YouTube links."""
        wp = self._make_wp()
        wp._process_youtube = MagicMock(return_value=[])
        wp._process_webpage = MagicMock(return_value=[])
        wp.process_url(self._YT_URL)
        wp._process_youtube.assert_called_once()
        wp._process_webpage.assert_not_called()

    def test_process_url_routes_webpage(self):
        """process_url should call _process_webpage for non-YouTube URLs."""
        wp = self._make_wp()
        wp._process_youtube = MagicMock(return_value=[])
        wp._process_webpage = MagicMock(return_value=[])
        wp.process_url("https://docs.python.org/3/")
        wp._process_webpage.assert_called_once()
        wp._process_youtube.assert_not_called()
