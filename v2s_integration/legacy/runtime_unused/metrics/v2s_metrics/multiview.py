"""Calibrated pinhole reprojection diagnostics, optical-axis depth, +Z cameras.
Camera matrices map local CV (right, down, forward) to world. Invalid depth is 0.
"""
import numpy as np
from .image import rgb, compare_images


def reprojection(source_depth, target_depth, source_rgb, target_rgb, K_source, K_target,
                 world_source, world_target, occlusion_tolerance_m=.03):
    a,b=np.asarray(source_depth,float),np.asarray(target_depth,float)
    if a.ndim!=2 or b.ndim!=2 or not np.isfinite(a).all() or not np.isfinite(b).all() or (a<0).any() or (b<0).any():
        raise ValueError('depth must be finite nonnegative HxW optical Z')
    ia,ib=rgb(source_rgb),rgb(target_rgb)
    if ia.shape[:2]!=a.shape or ib.shape[:2]!=b.shape:raise ValueError('RGB/depth sizes must match')
    ka,kb=np.asarray(K_source,float),np.asarray(K_target,float)
    wa,wb=np.asarray(world_source,float),np.asarray(world_target,float)
    if ka.shape!=(3,3) or kb.shape!=(3,3) or wa.shape!=(4,4) or wb.shape!=(4,4):raise ValueError('invalid camera shape')
    if not all(np.isfinite(x).all() for x in (ka,kb,wa,wb)):raise ValueError('nonfinite camera')
    if not np.isfinite(occlusion_tolerance_m) or occlusion_tolerance_m<0:raise ValueError('invalid tolerance')
    yy,xx=np.nonzero(a>0);n=len(xx)
    if not n:return {'status':'no_source_surface','n_source':0,'n_valid':0,'coverage':0.,'depth_mae_m':None,'rgb_mae':None}
    xyz=np.linalg.inv(ka)@np.vstack((xx,yy,np.ones(n)))*a[yy,xx]
    target=np.linalg.inv(wb)@wa@np.vstack((xyz,np.ones(n)))
    z=target[2]; uv=kb@target[:3]
    positive=z>1e-8
    uv[:2,positive]/=z[positive]
    x=np.rint(np.clip(uv[0],-1e7,1e7)).astype(int);y=np.rint(np.clip(uv[1],-1e7,1e7)).astype(int)
    inside=positive&(x>=0)&(x<b.shape[1])&(y>=0)&(y<b.shape[0])
    ids=np.where(inside)[0];td=b[y[ids],x[ids]]
    # Exclude hidden source points, not points floating in front of the target.
    valid=(td>0)&(z[ids]<=td+occlusion_tolerance_m);ids=ids[valid]
    return {'status':'ok' if len(ids) else 'no_overlap','n_source':n,'n_projected_inside':int(inside.sum()),
        'n_valid':len(ids),'coverage':len(ids)/n,
        'depth_mae_m':float(np.mean(np.abs(z[ids]-b[y[ids],x[ids]]))) if len(ids) else None,
        'rgb_mae':float(np.mean(np.abs(ia[yy[ids],xx[ids]]-ib[y[ids],x[ids]]))) if len(ids) else None,
        'protocol':'nearest pixel, optical depth, occlusion mask; report both ordered directions',
        'limitation':'self-consistency is not GT fidelity; specular appearance may vary across views'}


def object_crops(prediction, reference, gt_mask, config=None, padding=8, min_side=64):
    """Shared GT crop; never recenter/resize prediction to hide placement errors."""
    a,b=rgb(prediction),rgb(reference);mask=np.asarray(gt_mask,bool)
    if a.shape!=b.shape or mask.shape!=a.shape[:2]:raise ValueError('crop shapes disagree')
    if padding<0 or min_side<1:raise ValueError('invalid crop parameters')
    y,x=np.nonzero(mask)
    if not len(x):return {'status':'not_visible','metrics':{},'visible_pixels':0}
    x0,x1=max(0,int(x.min())-padding),min(a.shape[1],int(x.max())+1+padding)
    y0,y1=max(0,int(y.min())-padding),min(a.shape[0],int(y.max())+1+padding)
    # Expand the same crop in both images, without upsampling the small object.
    dx=max(0,min_side-(x1-x0));dy=max(0,min_side-(y1-y0))
    x0=max(0,x0-dx//2);x1=min(a.shape[1],max(x1,x0+min_side));x0=max(0,min(x0,x1-min_side))
    y0=max(0,y0-dy//2);y1=min(a.shape[0],max(y1,y0+min_side));y0=max(0,min(y0,y1-min_side))
    return {'status':'ok','bbox_xyxy':[x0,y0,x1,y1],'visible_pixels':len(x),
            'metrics':compare_images(a[y0:y1,x0:x1],b[y0:y1,x0:x1],config),
            'protocol':'same GT pixel crop with context; includes placement; no independent resize'}
