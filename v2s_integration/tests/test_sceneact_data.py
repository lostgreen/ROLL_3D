"""Source-layout compatibility without loading meshes or exposing private GT."""
import json
from pathlib import Path

import pytest

from v2s_integration.data.sceneact_kenney import canonical_path, select_rooms, furniture_aliases


def room(root, name, count, complete=True):
    folder = root / 'scenes' / name
    (folder / 'canonical').mkdir(parents=True)
    objects = [{'anon_id': f'object_{i:02d}', 'src_glb': f'Original_{i}.glb'} for i in range(count)]
    (folder / 'gt.json').write_text(json.dumps({'objects': objects}))
    for item in objects[:count if complete else count - 1]:
        (folder / 'canonical' / (item['anon_id'] + '.glb')).write_bytes(b'fixture')
    return folder, objects


def test_selects_complete_room_type_and_anonymized_assets(tmp_path):
    room(tmp_path, 'a_LivingDiningRoom-1', 1)
    room(tmp_path, 'b_DiningRoom-2', 2, complete=False)
    chosen, objects = room(tmp_path, 'c_DiningRoom-3', 3)
    room(tmp_path, 'd_DiningRoom-4', 4)
    assert select_rooms(tmp_path, ['DiningRoom']) == [str(chosen)]
    assert canonical_path(chosen, objects[0]).name == 'object_00.glb'
    with pytest.raises(ValueError, match='No complete'):
        select_rooms(tmp_path, ['Bedroom'])


def test_older_original_filename_fallback(tmp_path):
    folder, objects = room(tmp_path, 'a_Bedroom-1', 1, complete=False)
    source = folder / 'canonical' / objects[0]['src_glb']
    source.write_bytes(b'fixture')
    assert canonical_path(folder, objects[0]) == source
    assert select_rooms(tmp_path, ['Bedroom']) == [str(folder)]


def test_furniture_task_synonyms():
    assert 'bookshelf' in furniture_aliases('bookcaseOpen')
    assert 'couch' in furniture_aliases('loungeSofa')
    assert 'pendant lamp' in furniture_aliases('lampSquareCeiling')
    assert 'bed' in furniture_aliases('bedDouble')
