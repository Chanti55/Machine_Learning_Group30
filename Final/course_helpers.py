


from __future__ import annotations

import json
import hashlib
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path



PLOT_BLUE = "#1779A8"
PLOT_ORANGE = "#C96A00"
CLASS_COLOURS = (PLOT_BLUE, PLOT_ORANGE)




@dataclass(frozen=True)
class CleaningStep:
    column: str
    action: str
    reason: str
    rows_affected: int
    carries: dict | None = None  # the decision in a form later weeks can apply


class CleaningLog:
    # records what was cleaned and why, the cleaning itself happens in the notebook

    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        self.steps: list[CleaningStep] = []

    def record(self, column: str, action: str, reason: str, rows_affected: int,
               carries: dict | None = None) -> None:
        if not reason.strip():
            raise ValueError("every cleaning step needs a stated reason")
        self.steps.append(
            CleaningStep(column, action, reason, int(rows_affected), deepcopy(carries))
        )

    def records(self) -> list[dict]:
        # plain dicts for pd.DataFrame(...)
        return [
            {"column": step.column, "action": step.action,
             "reason": step.reason, "rows_affected": step.rows_affected}
            for step in self.steps
        ]

    def to_json(self, path) -> None:
        # json keeps carries nested, a csv would flatten it
        payload = {
            "dataset": self.dataset,
            "steps": [
                {"column": step.column, "action": step.action, "reason": step.reason,
                 "rows_affected": step.rows_affected, "carries": step.carries}
                for step in self.steps
            ],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path) -> "CleaningLog":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        log = cls(payload["dataset"])
        for step in payload["steps"]:
            log.record(step["column"], step["action"], step["reason"],
                       step["rows_affected"], step.get("carries"))
        return log

    def plan(self, column: str):
        # what the step for this column carries forward, None if nothing
        for step in self.steps:
            if step.column == column:
                return deepcopy(step.carries)
        return None

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        return f"CleaningLog(dataset={self.dataset!r}, steps={len(self.steps)})"