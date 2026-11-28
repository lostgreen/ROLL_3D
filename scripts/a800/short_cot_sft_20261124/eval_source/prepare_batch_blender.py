"""Prepare only explicitly declared public inputs; safe to resume per scene."""
import json
from pathlib import Path
import sys
import bpy
from mathutils import Vector
sys.path.insert(0, str(Path(__file__).parent))
from prepare_inputs import prepare_one, validate_rows, dump

root = Path(sys.argv[sys.argv.index('--')+1])
config = json.loads((root/'preparation_config.json').read_text())
reports = []
for row in config['rows']:
    receipt = root/'public'/row['id']/'preparation_receipt.json'
    if receipt.exists():
        old = json.loads(receipt.read_text())
        assert old['passed'] and old['after_reload']['meshes'] == 0
        continue
    validate_rows(root, {'rows':[row]})
    reports.append(prepare_one(bpy, Vector, root, config, row))
dump(root/'preparation_batch_receipt.json', dict(passed=True, newly_prepared=len(reports)))
