"""Four initial artifact kinds; files live outside the model conversation."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re


KINDS = frozenset(("evidence", "mesh", "gaussian", "scene"))


@dataclass(frozen=True)
class ArtifactManifest:
    artifact_id: str
    artifact_type: str
    path: str
    format: str
    tool_name: str
    tool_version: str
    coordinate_frame: str
    units: str
    input_artifact_ids: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    observed_coverage: dict = field(default_factory=dict)
    license: dict | None = None
    source: str = "generated"
    editable: bool = False
    renderable: bool = False
    normalization: dict = field(default_factory=dict)

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.artifact_id):
            raise ValueError("artifact_id must be an opaque file-safe ID")
        if self.artifact_type not in KINDS:
            raise ValueError("unsupported artifact kind")
        if self.source not in ("observed", "generated", "retrieval"):
            raise ValueError("unsupported artifact source")
        if self.source == "retrieval" and not (self.license and self.license.get("name")):
            raise ValueError("retrieval assets require license metadata")
        if not isinstance(self.normalization, dict):
            raise ValueError("normalization metadata must be an object")


class ArtifactStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def register(self, manifest):
        path = (self.root / manifest.path).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("artifact payload must be a file inside the artifact store")
        record = asdict(manifest)
        record["path"] = path.relative_to(self.root).as_posix()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        record["sha256"] = digest.hexdigest()
        target = self.root / f"{manifest.artifact_id}.manifest.json"
        with target.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, ensure_ascii=False)
        return record


@dataclass(frozen=True)
class EvidenceBundle:
    root: Path
    observed_views: dict[str, str]

    def resolve_observed_view(self, view_id):
        """Resolve only explicitly supplied observations, including symlink checks."""
        if view_id not in self.observed_views:
            raise ValueError("view is not observed evidence")
        root = Path(self.root).resolve()
        path = (root / self.observed_views[view_id]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("observed view must be a file in the evidence bundle")
        return path
