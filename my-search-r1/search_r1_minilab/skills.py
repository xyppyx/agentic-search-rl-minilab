"""Frozen, answer-free search skills used only by the training teacher."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SearchSkill:
    skill_id: str
    teacher_context: str


@dataclass(frozen=True)
class SkillBank:
    version: str
    digest: str
    default_skill_id: str
    by_source: dict[str, str]
    skills: dict[str, SearchSkill]

    def select(self, data_source: str) -> SearchSkill:
        skill_id = self.by_source.get(data_source.lower(), self.default_skill_id)
        return self.skills[skill_id]


def load_skill_bank(path: Path) -> SkillBank:
    """Load a fixed bank; source-only selection cannot use answers or future tools."""
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("skill bank must be a JSON object")
    version = payload.get("version")
    default_id = payload.get("default_skill_id")
    definitions = payload.get("skills")
    by_source = payload.get("by_source")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("skill bank version is required")
    if not isinstance(definitions, list) or not definitions:
        raise ValueError("skill bank must contain skills")
    if not isinstance(by_source, dict):
        raise ValueError("skill bank by_source must be an object")
    skills: dict[str, SearchSkill] = {}
    for definition in definitions:
        if not isinstance(definition, dict):
            raise ValueError("each skill must be an object")
        skill_id = definition.get("id")
        content = definition.get("teacher_context")
        if not isinstance(skill_id, str) or not skill_id.strip():
            raise ValueError("skill id is required")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"skill {skill_id} has empty teacher_context")
        if skill_id in skills:
            raise ValueError(f"duplicate skill id: {skill_id}")
        skills[skill_id] = SearchSkill(skill_id, content.strip())
    if default_id not in skills:
        raise ValueError("default_skill_id is not in skills")
    for source, skill_id in by_source.items():
        if not isinstance(source, str) or not source or source != source.lower():
            raise ValueError("by_source keys must be nonempty lowercase strings")
        if skill_id not in skills:
            raise ValueError(f"unknown skill id for source {source}")
    return SkillBank(
        version=version,
        digest=hashlib.sha256(raw).hexdigest(),
        default_skill_id=default_id,
        by_source=dict(by_source),
        skills=skills,
    )
