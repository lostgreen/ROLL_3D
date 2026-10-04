"""Load API credentials from an ignored file without exposing their values."""

import os
from pathlib import Path
import re
import stat


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "secrets/api.env"


def credential_path(path=None):
    value = path or os.environ.get("SCENEACT_SECRETS_FILE")
    return Path(value).expanduser().resolve() if value else DEFAULT_PATH.resolve()


def load_credentials(path=None, required=False):
    target = credential_path(path)
    if not target.exists():
        if required:
            raise FileNotFoundError(f"credential file not found: {target}")
        return target
    if not target.is_file():
        raise ValueError(f"credential path is not a file: {target}")
    if os.name == "posix":
        exposed = target.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO)
        if exposed:
            raise PermissionError(
                f"credential file must not be group/world accessible: {target}; run chmod 600")
    for number, raw in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", line)
        if not match:
            raise ValueError(f"invalid credential assignment at {target}:{number}")
        key, value = match.groups()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)
    return target
