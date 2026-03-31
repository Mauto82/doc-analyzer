"""
Prompt configuration.

Role prompts are loaded automatically from src/prompts/*.md.

File format:
    # Display Name       ← first line, used as label in the UI (optional)

    Prompt body text…   ← everything after the heading

The filename stem becomes the role key (e.g. legal.md → "legal").
Files are sorted alphabetically; "default" is always placed first.

To add a new role: drop a .md file in src/prompts/ and call
registry.reload() (or restart the server).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

BASE_PROMPT = """\
Based on the following document excerpts, answer the question.
Use ONLY the information provided in these excerpts to formulate your answer.
If the answer requires information from multiple sections, please specify which parts you're referencing.

{role_prompt}

Document excerpts: {context}

Question: {question}

Please provide your answer in the same language as the question, \
using only information from the provided excerpts:"""


@dataclass(frozen=True)
class RoleConfig:
    key: str
    name: str
    prompt: str


class PromptRegistry:
    """
    Auto-discovers and holds all role prompt configurations.

    Usage:
        from src.config.prompts import registry

        registry["legal"]          # RoleConfig
        "legal" in registry        # True/False
        registry.get_prompt("legal")  # prompt string, raises ValueError if missing
        registry.reload()          # re-read files from disk (no restart needed)
    """

    def __init__(self, prompts_dir: Path) -> None:
        self._dir = prompts_dir
        self._configs: Dict[str, RoleConfig] = {}
        self.reload()

    # ── Discovery ─────────────────────────────────────────────────────────────

    def reload(self) -> None:
        """Re-scan the prompts directory and rebuild the registry."""
        self._configs = self._load_from_dir(self._dir)

    @staticmethod
    def _parse_file(path: Path) -> tuple[str, str]:
        """
        Parse a prompt .md file.
        Returns (display_name, prompt_body).

        If the first non-empty line starts with "# ", it is used as the
        display name and stripped from the prompt body.
        """
        text = path.read_text(encoding="utf-8").strip()
        lines = text.splitlines()

        if lines and lines[0].startswith("# "):
            name = lines[0][2:].strip()
            body = "\n".join(lines[1:]).strip()
        else:
            name = path.stem.replace("_", " ").title()
            body = text

        return name, body

    @staticmethod
    def _load_from_dir(directory: Path) -> Dict[str, RoleConfig]:
        if not directory.exists():
            return {}

        paths = sorted(directory.glob("*.md"))
        # "default" always first regardless of alphabetical order
        default_path = directory / "default.md"
        if default_path in paths:
            paths.remove(default_path)
            paths.insert(0, default_path)

        configs: Dict[str, RoleConfig] = {}
        for path in paths:
            key = path.stem
            name, prompt = PromptRegistry._parse_file(path)
            configs[key] = RoleConfig(key=key, name=name, prompt=prompt)

        return configs

    # ── Access ────────────────────────────────────────────────────────────────

    def __contains__(self, key: object) -> bool:
        return key in self._configs

    def __getitem__(self, key: str) -> RoleConfig:
        return self._configs[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._configs)

    def keys(self):
        return self._configs.keys()

    def get_prompt(self, role: str) -> str:
        if role not in self._configs:
            raise ValueError(
                f"Invalid role '{role}'. Available: {', '.join(self._configs)}"
            )
        return self._configs[role].prompt

    def as_api_list(self) -> List[Dict[str, str]]:
        """Serialise for the /api/status response: [{key, name}, ...]."""
        return [{"key": rc.key, "name": rc.name} for rc in self._configs.values()]

    # ── Backward compat ───────────────────────────────────────────────────────

    @property
    def prompts(self) -> Dict[str, str]:
        """Dict[key → prompt_body] — kept for code that still uses ROLE_PROMPTS."""
        return {k: v.prompt for k, v in self._configs.items()}


# ── Module-level singletons ───────────────────────────────────────────────────

registry = PromptRegistry(PROMPTS_DIR)

# Backward-compatible alias — avoids touching every existing reference.
ROLE_PROMPTS: Dict[str, str] = registry.prompts
