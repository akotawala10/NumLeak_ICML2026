"""Tests for the minimal .env loader."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from factor_leak.env import load_dotenv


def test_loads_simple_key_value(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("FOO", raising=False)
    env = tmp_path / ".env"
    env.write_text("FOO=bar\n")
    loaded = load_dotenv(env)
    assert loaded == {"FOO": "bar"}
    assert os.environ["FOO"] == "bar"


def test_strips_quotes(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("FOO", raising=False)
    monkeypatch.delenv("BAZ", raising=False)
    env = tmp_path / ".env"
    env.write_text('FOO="bar"\nBAZ=\'qux\'\n')
    load_dotenv(env)
    assert os.environ["FOO"] == "bar"
    assert os.environ["BAZ"] == "qux"


def test_skips_comments_and_blank(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("A", raising=False)
    env = tmp_path / ".env"
    env.write_text("# comment\n\nA=1\n  # indented comment\n")
    loaded = load_dotenv(env)
    assert loaded == {"A": "1"}


def test_shell_wins_over_file(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PRESET", "from-shell")
    env = tmp_path / ".env"
    env.write_text("PRESET=from-file\n")
    loaded = load_dotenv(env)
    assert loaded == {}
    assert os.environ["PRESET"] == "from-shell"


def test_override_true_takes_file_value(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OVR", "from-shell")
    env = tmp_path / ".env"
    env.write_text("OVR=from-file\n")
    loaded = load_dotenv(env, override=True)
    assert loaded == {"OVR": "from-file"}
    assert os.environ["OVR"] == "from-file"


def test_missing_file_returns_empty(tmp_path: Path):
    loaded = load_dotenv(tmp_path / "nonexistent.env")
    assert loaded == {}


def test_skips_malformed_lines(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("GOOD", raising=False)
    env = tmp_path / ".env"
    env.write_text("no equals sign\nGOOD=yes\n=no_key\n")
    loaded = load_dotenv(env)
    assert loaded == {"GOOD": "yes"}
