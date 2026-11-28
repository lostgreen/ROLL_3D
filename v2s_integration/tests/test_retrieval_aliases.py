"""Natural task categories resolve opaque vendor names without changing payloads."""
import json

from v2s_integration.runtime.session import create_session  # install vendor paths
from tools.retrieval import RetrievalProvider


def test_alias_and_category_are_searchable_and_public(tmp_path):
    (tmp_path / 'furniture.glb').write_bytes(b'index-contract-only')
    index = tmp_path / 'index.json'
    index.write_text(json.dumps({'assets': [
        {'asset_id': 'item_a', 'name': 'bookcaseOpen', 'glb_path': 'furniture.glb',
         'category': 'storage', 'aliases': ['bookshelf', 'shelf']},
        {'asset_id': 'item_b', 'name': 'loungeSofa', 'glb_path': 'furniture.glb',
         'category': 'seating', 'aliases': ['couch', 'loveseat']}]}))
    provider = RetrievalProvider(index, None)
    for query, expected in [('bookshelf', 'item_a'), ('storage', 'item_a'), ('couch', 'item_b')]:
        hits = json.loads(provider.search({'query': query}).text)['assets']
        assert hits[0]['asset_id'] == expected
        assert hits[0]['payload_available']
        assert hits[0]['category']
        assert hits[0]['aliases']
