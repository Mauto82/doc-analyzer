"""
WebProcessor: import content from URLs into the RAG pipeline.

Supported sources:
- YouTube videos  — transcript via youtube-transcript-api (no API key required)
- Web pages       — clean text extraction via trafilatura
- API docs        — same as web pages; trafilatura handles structured HTML well
"""
import os
import re
from typing import List

from src.models.document import Document
from src.utils.text_splitter import RecursiveCharacterTextSplitter
from .base.document_processor import DocumentProcessor

_YOUTUBE_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?.*v=|embed/|v/)|youtu\.be/)([A-Za-z0-9_-]{11})"
)


def _extract_video_id(url: str) -> str | None:
    m = _YOUTUBE_RE.search(url)
    return m.group(1) if m else None


class WebProcessor(DocumentProcessor):
    """
    Fetch and chunk content from a URL.

    The process() method is kept for interface compatibility but is not used
    directly — call process_url(url) instead, which accepts a URL string
    rather than a file path.
    """

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        )

    # ── Public ────────────────────────────────────────────────────────────────

    def process(self, file_path: str) -> List[Document]:
        """Treat file_path as a URL (interface compatibility)."""
        return self.process_url(file_path)

    def process_url(self, url: str) -> List[Document]:
        """Fetch content from url and return Document chunks."""
        video_id = _extract_video_id(url)
        if video_id:
            return self._process_youtube(url, video_id)
        return self._process_webpage(url)

    # ── Private ───────────────────────────────────────────────────────────────

    def _process_youtube(self, url: str, video_id: str) -> List[Document]:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            from youtube_transcript_api._errors import CouldNotRetrieveTranscript
        except ImportError:
            raise RuntimeError(
                "youtube-transcript-api is not installed. "
                "Add it to requirements.txt and rebuild."
            )

        try:
            # v1.x API: instantiate the class, then call .fetch()
            api = YouTubeTranscriptApi()
            transcript = api.fetch(video_id, languages=["it", "en", "en-US", "en-GB"])
            segments = list(transcript)
        except CouldNotRetrieveTranscript as exc:
            raise ValueError(
                f"No transcript available for this video. "
                f"It may be disabled or the video may not exist. ({exc})"
            )

        # Snippets expose .text as an attribute (and also support ['text'])
        text = " ".join(s.text for s in segments).strip()
        if not text:
            raise ValueError("Transcript is empty.")

        doc = Document(
            page_content=text,
            metadata={"source": url, "type": "youtube", "video_id": video_id},
        )
        return self.text_splitter.split_documents([doc])

    def _process_webpage(self, url: str) -> List[Document]:
        try:
            import trafilatura
        except ImportError:
            raise RuntimeError(
                "trafilatura is not installed. "
                "Add it to requirements.txt and rebuild."
            )

        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            raise ValueError(f"Could not fetch content from '{url}'. "
                             "Check that the URL is accessible.")

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )
        if not text or not text.strip():
            raise ValueError(
                f"No readable content extracted from '{url}'. "
                "The page may require JavaScript or authentication."
            )

        doc = Document(
            page_content=text.strip(),
            metadata={"source": url, "type": "webpage"},
        )
        return self.text_splitter.split_documents([doc])
