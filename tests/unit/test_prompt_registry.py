"""Unit tests for PromptRegistry — auto-discovery and hot-reload."""
import pytest
from pathlib import Path

from src.config.prompts import PromptRegistry


@pytest.fixture
def prompts_dir(tmp_path: Path) -> Path:
    return tmp_path / "prompts"


@pytest.fixture
def registry(prompts_dir: Path) -> PromptRegistry:
    prompts_dir.mkdir()
    return PromptRegistry(prompts_dir)


def write_prompt(directory: Path, filename: str, content: str) -> None:
    (directory / filename).write_text(content, encoding="utf-8")


# ── Loading ───────────────────────────────────────────────────────────────────

def test_empty_directory_returns_empty_registry(registry):
    assert list(registry.keys()) == []


def test_loads_prompt_with_heading(registry, prompts_dir):
    write_prompt(prompts_dir, "legal.md", "# Legal Analyst\n\nYou are a lawyer.")
    registry.reload()
    assert "legal" in registry
    assert registry["legal"].name == "Legal Analyst"
    assert registry["legal"].prompt == "You are a lawyer."


def test_loads_prompt_without_heading(registry, prompts_dir):
    write_prompt(prompts_dir, "simple.md", "Just a plain prompt.")
    registry.reload()
    assert registry["simple"].name == "Simple"   # titlecased from stem
    assert registry["simple"].prompt == "Just a plain prompt."


def test_default_is_always_first(registry, prompts_dir):
    write_prompt(prompts_dir, "zzz_last.md", "# Z\nLast alphabetically.")
    write_prompt(prompts_dir, "default.md", "# General\nDefault prompt.")
    write_prompt(prompts_dir, "aaa_first.md", "# A\nFirst alphabetically.")
    registry.reload()
    keys = list(registry.keys())
    assert keys[0] == "default"


def test_ignores_non_md_files(registry, prompts_dir):
    write_prompt(prompts_dir, "notes.txt", "This should be ignored.")
    write_prompt(prompts_dir, "legal.md", "# Legal\nLegal prompt.")
    registry.reload()
    assert "notes" not in registry
    assert "legal" in registry


# ── Access ────────────────────────────────────────────────────────────────────

def test_contains_operator(registry, prompts_dir):
    write_prompt(prompts_dir, "legal.md", "# Legal\nLegal prompt.")
    registry.reload()
    assert "legal" in registry
    assert "nonexistent" not in registry


def test_get_prompt_returns_body(registry, prompts_dir):
    write_prompt(prompts_dir, "legal.md", "# Legal\nLegal prompt body.")
    registry.reload()
    assert registry.get_prompt("legal") == "Legal prompt body."


def test_get_prompt_raises_on_unknown_role(registry):
    with pytest.raises(ValueError, match="Invalid role"):
        registry.get_prompt("nonexistent")


def test_as_api_list_format(registry, prompts_dir):
    write_prompt(prompts_dir, "legal.md", "# Legal Analyst\nLegal prompt.")
    registry.reload()
    api_list = registry.as_api_list()
    assert isinstance(api_list, list)
    assert api_list[0] == {"key": "legal", "name": "Legal Analyst"}


# ── Hot-reload ────────────────────────────────────────────────────────────────

def test_reload_picks_up_new_file(registry, prompts_dir):
    write_prompt(prompts_dir, "legal.md", "# Legal\nLegal prompt.")
    registry.reload()
    assert "legal" in registry
    assert "financial" not in registry

    write_prompt(prompts_dir, "financial.md", "# Financial\nFinancial prompt.")
    registry.reload()
    assert "financial" in registry


def test_reload_removes_deleted_file(registry, prompts_dir):
    path = prompts_dir / "temp.md"
    write_prompt(prompts_dir, "temp.md", "# Temp\nTemp prompt.")
    registry.reload()
    assert "temp" in registry

    path.unlink()
    registry.reload()
    assert "temp" not in registry


# ── Missing directory ─────────────────────────────────────────────────────────

def test_missing_directory_returns_empty(tmp_path):
    reg = PromptRegistry(tmp_path / "nonexistent")
    assert list(reg.keys()) == []
