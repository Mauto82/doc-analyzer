from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Note:
    question: str
    answer: str
    sources: list = field(default_factory=list)   # [{"file": ..., "page": ..., "score": ...}]
    collection: str = "default"
    role: str = "default"
    id: int = 0
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()

    @property
    def title(self) -> str:
        """First 60 chars of the question, used as display title."""
        return self.question[:60] + ("…" if len(self.question) > 60 else "")
