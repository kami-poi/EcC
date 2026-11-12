"""Source pairs and audit-only provenance; only source text enters the model."""

from dataclasses import dataclass

from model.labels import Targets


@dataclass(frozen=True, slots=True)
class PairRecord:
    record_id: str
    pre_code: str | None
    post_code: str | None
    targets: Targets | None
    repo_id: str | None
    commit_sha: str | None
    source_id: str | None
    language: str | None
