"""Versioned V2S envelope for offline rollout archives.

This is a portable extension payload, NOT an assertion that an arbitrary ROLL
DataProto/save_content release accepts it. Token IDs/masks must come from ROLL;
we preserve them when supplied and never synthesize training masks from text.
"""
from copy import deepcopy

FORMAT = 'video2scene.roll.offline.v1'


def export_rollout(trajectory, training=None):
    if trajectory.get('schema') != 'video2scene.trajectory.v2':
        raise ValueError('canonical trajectory v2 required')
    return {'schema':FORMAT, 'trajectory_id':trajectory['trajectory_id'],
            'video2scene':deepcopy(trajectory), 'training':deepcopy(training)}


def import_rollout(dump):
    if dump.get('schema') != FORMAT:
        raise ValueError('unsupported ROLL dump; requires a version-specific mapping')
    trajectory = deepcopy(dump['video2scene'])
    if trajectory.get('schema') != 'video2scene.trajectory.v2' or trajectory.get('trajectory_id') != dump.get('trajectory_id'):
        raise ValueError('trajectory identity mismatch')
    return trajectory
