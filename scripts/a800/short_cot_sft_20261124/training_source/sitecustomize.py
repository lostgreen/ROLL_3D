import sys
from pathlib import Path
p="/m2v_intern/xuboshen/zgw/ROLL/artifacts/k_sft_native_thinking_v2_20261116/vendor"
if Path(p).is_dir() and p not in sys.path:sys.path.append(p)
