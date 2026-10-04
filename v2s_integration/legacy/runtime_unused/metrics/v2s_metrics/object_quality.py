"""v0.2 object diagnostics. Unit-box shape is separate from world placement.

Upright yaw search is an explicitly bounded alignment diagnostic, not arbitrary
SO(3) invariance. No alignment is applied to the world-coordinate score.
"""
from dataclasses import replace
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from .config import MetricConfig
from .geometry import points, compare_points
from .instances import Instance, pose_metrics, rotation


def unit_box(p, frame=None):
    p = points(p, allow_empty=False)
    if frame is not None:
        p = p @ rotation(frame)
    lo, hi = p.min(0), p.max(0)
    extent = hi-lo
    scale = float(extent.max())
    if scale <= 1e-12:
        raise ValueError('degenerate object has zero extent')
    return (p-(lo+hi)/2)/scale, scale, extent


def yaw_matrix(angle):
    c,s = math.cos(angle), math.sin(angle)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])


def shape_pair(p, g, yaw_steps=24, sample_limit=1500):
    """One isotropic scale per object; no independent xyz stretching."""
    p = p if isinstance(p, Instance) else Instance('pred',p)
    g = g if isinstance(g, Instance) else Instance('gt',g)
    if type(yaw_steps) is not int or yaw_steps < 1:
        raise ValueError('yaw_steps must be positive integer')
    if type(sample_limit) is not int or sample_limit < 1:
        raise ValueError('sample_limit must be positive integer')
    pp = p.points @ p.rotation if p.rotation is not None else p.points
    gg = g.points @ g.rotation if g.rotation is not None else g.points
    gn,gs,ge = unit_box(gg)
    # Deterministic uniform index subset used only for alignment selection.
    gsub = gn[np.linspace(0,len(gn)-1,min(sample_limit,len(gn))).astype(int)]
    gt = cKDTree(gsub)
    best = None
    for k in range(yaw_steps):
        angle = 2*math.pi*k/yaw_steps
        pn,ps,pe = unit_box(pp @ yaw_matrix(angle))
        sub = pn[np.linspace(0,len(pn)-1,min(sample_limit,len(pn))).astype(int)]
        cd = .5*(gt.query(sub)[0].mean()+cKDTree(sub).query(gsub)[0].mean())
        if best is None or cd < best[0]:
            best = (float(cd),angle,pn,ps,pe)
    _,angle,pn,ps,pe = best
    config=MetricConfig(distance_thresholds_m=(.02,.05,.1))
    scores=compare_points(pn,gn,config)
    # Re-label lengths: compare_points operates on arrays but these have unit-box units.
    for key in ('cd_mean_m','accuracy_m','completeness_m'):
        scores[(key[:-2]+'_unit')]=scores.pop(key)
        scores[(key[:-2]+'_unit')]['unit']='unit_box'
    scores['threshold_units']='fraction of independently normalized maximum extent'
    scores['thresholds']={key.replace('m','unit'):value for key,value in scores['thresholds'].items()}
    return {'status':'ok','metrics':scores,'alignment':{'translation':'bbox_center',
        'scale':'one uniform maximum-extent scalar per object; aspect ratio preserved',
        'orientation':('supplied_frames' if p.rotation is not None and g.rotation is not None else 'world_upright')+f'+yaw_grid_{yaw_steps}',
        'yaw_rad':angle,'prediction_scale_m':ps,'reference_scale_m':gs,
        'prediction_extent_aligned_m':pe.tolist(),'reference_extent_m':ge.tolist(),
        'limitation':'yaw-optimized shape diagnostic, not unrestricted rotation invariance or semantic pose'}}


def compare_object_sets(prediction, reference, config=None, yaw_steps=24, max_shape_cd=.4):
    """No hard location gate. Assignment is geometry/category evidence, not verified semantics."""
    config=config or MetricConfig()
    if not math.isfinite(max_shape_cd) or max_shape_cd<=0:raise ValueError('invalid shape gate')
    for group in (prediction,reference):
        if len({x.id for x in group})!=len(group):raise ValueError('duplicate instance IDs')
    n,m=len(prediction),len(reference); pairs={}; matches=[]
    if n and m:
        cost=np.full((n+m,n+m),1e6);cost[n:,m:]=0
        cost[:n,m:]=.5;cost[n:,:m]=.5
        for i,p in enumerate(prediction):
            for j,g in enumerate(reference):
                if p.category is not None and g.category is not None and p.category!=g.category:continue
                try:shape=shape_pair(p,g,yaw_steps)
                except ValueError:continue
                cd=shape['metrics']['cd_mean_unit']['value']
                if cd>max_shape_cd:continue
                distance=np.linalg.norm(p.center-g.center)
                cost[i,j]=.8*cd/max_shape_cd+.19*distance/(distance+1)
                pairs[i,j]=shape
        rows,cols=linear_sum_assignment(cost)
        for i,j in zip(rows,cols):
            if (i,j) not in pairs:continue
            p,g=prediction[i],reference[j];shape=pairs[i,j]
            pose=pose_metrics(p,g)
            extent_p=np.array(shape['alignment']['prediction_extent_aligned_m']);extent_g=np.array(shape['alignment']['reference_extent_m'])
            valid=(extent_p>1e-9)&(extent_g>1e-9)
            size_error=float(np.linalg.norm(np.log(extent_p[valid]/extent_g[valid]))) if valid.all() else None
            matches.append({'prediction_id':p.id,'reference_id':g.id,'category_evidence':'declared_not_independently_verified',
                'shape':shape,'world':compare_points(p.points,g.points,config),'pose':pose,
                'continuous':{'translation_exp_sigma_0_5m':float(np.exp(-pose['center_error_m']['value']/.5)),
                'size_log_l2_aligned':size_error,'size_exp_sigma_1':float(np.exp(-size_error)) if size_error is not None else None},
                'continuous_status':'diagnostic_uncalibrated_not_rl_reward'})
    gp={x['reference_id'] for x in matches};pp={x['prediction_id'] for x in matches}
    macro={}
    for space in ('world','shape'):
        for match in matches:
            metrics=match[space] if space=='world' else match['shape']['metrics']
            for threshold,v in metrics['thresholds'].items():macro.setdefault(space,{}).setdefault(threshold,[]).append(v['fscore']['value'])
    # Emit thresholds even if nothing matches.
    for space,thresholds in [('world',compare_points(np.empty((0,3)),np.array([[0.,0,0]]),config)['thresholds']),
                             ('shape',{'0.02unit':{},'0.05unit':{},'0.1unit':{}})]:
        for key in thresholds:macro.setdefault(space,{}).setdefault(key,[])
    object_set_world = {key: {'precision': sum(values)/n if n else (0. if m else None),
        'recall': sum(values)/m if m else None,
        'f1': 2*sum(values)/(n+m) if n+m else None,
        'fractional_tp': sum(values)} for key, values in macro['world'].items()}
    macro={space:{k:sum(v)/m if m else None for k,v in vals.items()} for space,vals in macro.items()}
    return {'schema':'object_quality/0.2','matching':{'mode':'category+unit_box_shape+soft_position',
        'shape_cd_gate':max_shape_cd,'hard_center_gate':False,'presence_semantics':'geometric assignment count, not verified semantic existence'},
        'n_prediction':n,'n_reference':m,'n_matched':len(matches),
        'assignment_precision':len(matches)/n if n else (0. if m else None),
        'assignment_recall':len(matches)/m if m else None,
        'object_set_world':object_set_world,
        'object_set_formula':'2*sum(matched surface F)/(n_prediction+n_reference); empty-empty undefined',
        'macro_fscore_missing_as_zero':macro,'missing':[g.id for g in reference if g.id not in gp],
        'extra':[p.id for p in prediction if p.id not in pp],'matches':matches}
