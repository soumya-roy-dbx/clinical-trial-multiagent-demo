"""Shared types and the BaseAgent contract."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentResponse:
    answer: str
    source: str               # "genie" | "vector" | "hybrid" | "mock"
    confidence: float = 0.0
    data: Any | None = None   # tabular result, when applicable
    sql: str | None = None    # generated SQL, when applicable
    chunks: list[dict] = field(default_factory=list)  # retrieved passages


class BaseAgent(ABC):
    name: str = "base"

    @abstractmethod
    def handle(self, question: str) -> AgentResponse: ...
