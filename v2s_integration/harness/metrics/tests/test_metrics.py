import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from v2s_metrics import MetricConfig, compare_images, compare_points, sample_surface, Instance, compare_instances, compare_views
from v2s_metrics.evaluate import evaluate_manifest, read_json
from v2s_metrics.geometry import load_points
from v2s_metrics.views import aggregate_views


def box(offset=(0, 0, 0)):
    return np.array([[x, y, z] for x in (0., 1.) for y in (0., 1.) for z in (0., 1.)]) + offset


def test_image_identity_and_analytic_error():
    a = np.zeros((16, 16, 3), dtype=np.uint8)
    identity = compare_images(a, a)
    assert identity['mse']['value'] == 0
    assert identity['ssim']['value'] == pytest.approx(1)
    assert identity['psnr_db']['value'] == 100 and identity['psnr_db']['capped']
    changed = compare_images(a, np.full(a.shape, 255, dtype=np.uint8))
    assert changed['mse']['value'] == 1 and changed['psnr_db']['value'] == 0
    json.dumps(identity, allow_nan=False)


@pytest.mark.parametrize('bad', [np.zeros((16, 16, 4)), np.full((16, 16, 3), np.nan),
                                 np.full((16, 16, 3), 255.), np.zeros((16, 16, 3), dtype=np.uint16)])
def test_image_rejects_ambiguous_input(bad):
    with pytest.raises(ValueError):
        compare_images(bad, np.zeros((16, 16, 3)))


def test_no_implicit_resize():
    with pytest.raises(ValueError, match='shapes differ'):
        compare_images(np.zeros((16, 16, 3)), np.zeros((17, 16, 3)))


def test_small_image_retains_mse_but_marks_ssim_unavailable():
    result = compare_images(np.zeros((2, 2, 3)), np.zeros((2, 2, 3)))
    assert result['mse']['value'] == 0
    assert result['ssim']['status'] == 'invalid_input'


def test_lpips_missing_dependency_is_explicit(monkeypatch):
    import v2s_metrics.image as image
    def absent(*args):
        raise ImportError('weights not cached')
    monkeypatch.setattr(image, 'lpips_distance', absent)
    a = np.zeros((64, 64, 3))
    result = compare_images(a, a, MetricConfig(image_metrics=('mse', 'lpips')))
    assert result['lpips']['value'] is None
    assert result['lpips']['status'] == 'dependency_unavailable'
    assert result['mse']['value'] == 0


def test_view_failure_is_in_denominator_and_splits_are_separate(tmp_path):
    a = np.zeros((16, 16, 3))
    views = [{'id': 'a', 'split': 'observed', 'prediction': a, 'reference': a},
             {'id': 'b', 'split': 'hidden', 'prediction': a, 'reference': a + .5},
             {'id': 'c', 'split': 'hidden', 'prediction': tmp_path/'missing.png', 'reference': a}]
    result = compare_views(views)
    hidden = result['splits']['hidden']['mse']
    assert hidden['n_expected'] == 2 and hidden['n_valid'] == 1
    assert hidden['coverage'] == .5 and hidden['status'] == 'partial'
    assert hidden['mean'] == .25
    assert result['splits']['observed']['mse']['mean'] == 0


def test_worst_tail_direction_and_rounding():
    records = [{'metrics': {'lpips': {'value': float(i), 'status': 'ok'},
                            'ssim': {'value': float(i), 'status': 'ok'}}} for i in range(6)]
    result = aggregate_views(records, ['lpips', 'ssim'])
    assert result['lpips']['worst_fraction_mean'] == 4.5
    assert result['ssim']['worst_fraction_mean'] == .5
    assert result['ssim']['n_worst'] == 2


def test_duplicate_view_id_rejected():
    with pytest.raises(ValueError, match='unique'):
        compare_views([{'id': 'same'}, {'id': 'same'}])


def test_geometry_analytic_distance_units_and_strict_threshold():
    result = compare_points([[.05, 0, 0]], [[0, 0, 0]])
    assert result['cd_mean_m']['value'] == .05
    assert result['accuracy_m']['value'] == .05
    assert result['thresholds']['0.05m']['fscore']['value'] == 0
    assert result['thresholds']['0.1m']['fscore']['value'] == 1


def test_geometry_empty_prediction_penalized_reference_not_silently_skipped():
    result = compare_points(np.empty((0, 3)), box())
    assert result['status'] == 'prediction_empty' and result['cd_mean_m']['value'] is None
    assert result['thresholds']['0.05m']['fscore']['value'] == 0
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError):
        compare_points(box(), np.empty((0, 3)))


def test_geometry_identity_and_no_hidden_alignment():
    assert compare_points(box(), box())['cd_mean_m']['value'] == 0
    assert compare_points(box((3, 0, 0)), box())['cd_mean_m']['value'] > 1


def test_surface_sampling_area_weighted_reproducible_and_degenerate_safe():
    vertices = np.array([[0,0,0],[1,0,0],[0,1,0],[10,0,0],[12,0,0],[10,2,0]])
    faces = np.array([[0,1,2],[3,4,5],[0,0,0]])
    sampled = sample_surface(vertices, faces, 10000, 1)
    assert np.array_equal(sampled, sample_surface(vertices, faces, 10000, 1))
    assert np.mean(sampled[:,0] > 5) == pytest.approx(.8, abs=.02)
    assert np.all(sampled[:,2] == 0)
    with pytest.raises(ValueError, match='nondegenerate'):
        sample_surface(vertices, np.array([[0,0,0]]))


def test_surface_metric_stable_under_triangle_subdivision():
    a = np.array([[0,0,0],[1,0,0],[0,1,0]])
    b = np.vstack([a, [.5,0,0], [.5,.5,0], [0,.5,0]])
    p = sample_surface(a, np.array([[0,1,2]]), 8000, 3)
    q = sample_surface(b, np.array([[0,3,5],[3,1,4],[5,4,2],[3,4,5]]), 8000, 7)
    assert compare_points(p, q)['cd_mean_m']['value'] < .01


def test_missing_and_extra_objects_both_penalized():
    gt = [Instance('chair', box(), 'chair'), Instance('bed', box((3,0,0)), 'bed')]
    pred = [Instance('chair1', box(), 'chair'), Instance('chair2', box(), 'chair')]
    result = compare_instances(pred, gt)
    assert result['detection']['precision']['value'] == .5
    assert result['detection']['recall']['value'] == .5
    assert result['detection']['f1']['value'] == .5
    assert result['missing_reference_ids'] == ['bed']
    assert len(result['extra_prediction_ids']) == 1
    assert result['macro_fscore_missing_as_zero']['0.05m']['world']['value'] == .5


def test_object_shape_and_position_separated_without_scale_fit():
    result = compare_instances([Instance('p', box((.5,0,0)))], [Instance('g', box())])
    match = result['matches'][0]
    assert match['pose']['center_error_m']['value'] == .5
    assert match['world']['cd_mean_m']['value'] > 0
    assert match['centered_shape']['cd_mean_m']['value'] == 0
    scaled = compare_instances([Instance('p', box() * 1.5)], [Instance('g', box())])['matches'][0]
    assert scaled['centered_shape']['cd_mean_m']['value'] > 0
    assert scaled['pose']['aabb_extent_relative_error']['value'] == .5


@pytest.mark.parametrize('pred', [[Instance('p', box((10,0,0)), 'chair')], [Instance('p', box(), 'sofa')]])
def test_matching_does_not_force_invalid_pairs(pred):
    result = compare_instances(pred, [Instance('g', box(), 'chair')])
    assert result['n_matched'] == 0 and result['detection']['f1']['value'] == 0


def test_assignment_prefers_maximum_number_of_gated_matches():
    # p0 can match g0/g1, p1 only g0; nearest greedy would leave a GT unmatched.
    result = compare_instances([Instance('p0', box((.1,0,0))), Instance('p1', box((-.8,0,0)))],
                               [Instance('g0', box()), Instance('g1', box((.9,0,0)))])
    assert result['n_matched'] == 2
    assert {m['prediction_id']:m['reference_id'] for m in result['matches']} == {'p0':'g1','p1':'g0'}


def test_instance_empty_cases():
    result = compare_instances([], [Instance('g', box())])
    assert result['detection']['f1']['value'] == 0
    assert result['macro_fscore_missing_as_zero']['0.05m']['world']['value'] == 0
    empty = compare_instances([], [])
    assert empty['detection']['f1']['value'] is None
    json.dumps(empty, allow_nan=False)


def test_rotation_symmetry_and_invalid_matrix():
    rz90 = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    pred = Instance('p', box(), rotation=rz90)
    gt = Instance('g', box(), rotation=np.eye(3))
    assert compare_instances([pred], [gt])['matches'][0]['pose']['rotation_error_deg']['value'] == 90
    gt.symmetries = (rz90,)
    assert compare_instances([pred], [gt])['matches'][0]['pose']['rotation_error_deg']['value'] == 0
    with pytest.raises(ValueError, match='orthogonal'):
        Instance('invalid', box(), rotation=np.eye(3)*2)


def manifest_fixture(tmp_path):
    np.save(tmp_path/'p.npy', box())
    np.save(tmp_path/'g.npy', box())
    manifest = {'schema_version':1, 'protocol':{'units':'m','alignment':'none'},
                'geometry':{'prediction':'p.npy','reference':'g.npy'}}
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_manifest_paths_hashes_overrides_and_json(tmp_path):
    path, _ = manifest_fixture(tmp_path)
    report = evaluate_manifest(path, {'distance_thresholds_m':[.03]})
    assert report['evaluation_status'] == 'complete'
    assert report['geometry']['thresholds']['0.03m']['fscore']['value'] == 1
    assert len(report['inputs'][str((tmp_path/'p.npy').resolve())]['sha256']) == 64
    json.dumps(report, allow_nan=False)


def test_manifest_missing_file_reported_incomplete(tmp_path):
    path, manifest = manifest_fixture(tmp_path)
    manifest['geometry']['prediction'] = 'absent.npy'
    path.write_text(json.dumps(manifest))
    report = evaluate_manifest(path)
    assert report['evaluation_status'] == 'incomplete'
    assert report['geometry']['status'] == 'invalid_input'


@pytest.mark.parametrize('config', [{'distance_thresholds_m':[]}, {'distance_thresholds_m':[float('nan')]},
                                   {'worst_fraction':0}, {'ssim_window':4}, {'surface_samples':True},
                                   {'image_metrics':['unknown']}, {'lpips_allow_download':'false'}])
def test_config_validation(config):
    with pytest.raises(ValueError):
        MetricConfig(**config)


def test_duplicate_and_nonfinite_json_rejected(tmp_path):
    p = tmp_path/'invalid.json'
    for text in ('{"x":1,"x":2}', '{"x":NaN}'):
        p.write_text(text)
        with pytest.raises(ValueError):
            read_json(p)


def test_cli_and_input_overwrite_protection(tmp_path):
    manifest, _ = manifest_fixture(tmp_path)
    entry = Path(__file__).resolve().parents[1]/'evaluate.py'
    output = tmp_path/'results.json'
    cmd = [sys.executable, str(entry), '--manifest', str(manifest), '--output', str(output)]
    process = subprocess.run(cmd, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert read_json(output)['evaluation_status'] == 'complete'
    original = manifest.read_bytes()
    process = subprocess.run(cmd[:-1]+[str(manifest)], capture_output=True, text=True)
    assert process.returncode == 1 and manifest.read_bytes() == original


def test_mesh_node_transform_applied_before_sampling(tmp_path):
    trimesh = pytest.importorskip('trimesh')
    scene = trimesh.Scene()
    transform = np.eye(4)
    transform[:3,3] = [4,2,1]
    scene.add_geometry(trimesh.creation.box(), transform=transform)
    path = tmp_path/'transformed.glb'
    scene.export(path)
    sampled = load_points(path, MetricConfig(surface_samples=3000))
    assert np.all(sampled.min(axis=0) >= np.array([3.5,1.5,.5]) - 1e-12)
    assert np.all(sampled.max(axis=0) <= np.array([4.5,2.5,1.5]) + 1e-12)
    assert np.linalg.norm(sampled.mean(axis=0)-[4,2,1]) < .04


def test_distinct_precision_thresholds_do_not_overwrite():
    result = compare_points([[0,0,0]], [[0,0,0]], MetricConfig(distance_thresholds_m=(.02000001, .02000002)))
    assert len(result['thresholds']) == 2


def test_optional_backend_failure_does_not_erase_other_metrics(monkeypatch):
    import v2s_metrics.image as image
    def broken(*args):
        raise RuntimeError('torchvision binary mismatch')
    monkeypatch.setattr(image, 'lpips_distance', broken)
    a = np.zeros((64,64,3))
    result = compare_images(a, a, MetricConfig(image_metrics=('mse','lpips')))
    assert result['mse']['value'] == 0
    assert result['lpips']['status'] == 'backend_error'
