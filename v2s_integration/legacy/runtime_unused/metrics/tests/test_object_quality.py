import numpy as np
import pytest
from v2s_metrics.instances import Instance
from v2s_metrics.object_quality import shape_pair, compare_object_sets, yaw_matrix
from v2s_metrics.multiview import reprojection, object_crops


def cloud():
    r=np.random.default_rng(3)
    return r.uniform(-.5,.5,(400,3))*[1,2,.4]


def test_shape_translation_scale_yaw_and_world_separation():
    g=cloud();p=(g@yaw_matrix(np.pi/2))*2+[4,5,6]
    s=shape_pair(p,g,yaw_steps=4)
    assert s['metrics']['cd_mean_unit']['value']<1e-12
    out=compare_object_sets([Instance('p',p,'chair')],[Instance('g',g,'chair')],yaw_steps=4)
    assert out['n_matched']==1  # No hard center gate.
    assert out['matches'][0]['world']['cd_mean_m']['value']>1
    assert out['matches'][0]['continuous']['translation_exp_sigma_0_5m']<.01


def test_aspect_not_normalized_away():
    g=cloud();s=shape_pair(g*[3,1,1],g,4)
    assert s['metrics']['cd_mean_unit']['value']>.02


def test_missing_extra_and_categories():
    g=cloud();out=compare_object_sets([Instance('a',g,'chair'),Instance('b',g+1,'lamp')],
             [Instance('x',g,'chair'),Instance('y',g+3,'bed')],yaw_steps=4)
    assert out['n_matched']==1 and out['missing']==['y'] and out['extra']==['b']
    assert out['macro_fscore_missing_as_zero']['world']['0.05m']==.5
    empty=compare_object_sets([], [Instance('x',g)])
    assert empty['macro_fscore_missing_as_zero']['shape']['0.05unit']==0


def test_reprojection_plane_and_missing():
    d=np.ones((10,10));im=np.zeros((10,10,3));K=np.array([[10,0,5],[0,10,5],[0,0,1]])
    v=reprojection(d,d,im,im,K,K,np.eye(4),np.eye(4))
    assert v['coverage']==1 and v['depth_mae_m']==0 and v['rgb_mae']==0
    v=reprojection(d,d,im,im+.5,K,K,np.eye(4),np.eye(4))
    assert v['rgb_mae']==.5
    v=reprojection(d,d*0,im,im,K,K,np.eye(4),np.eye(4))
    assert v['status']=='no_overlap' and v['depth_mae_m'] is None
    v=reprojection(d*.5,d,im,im,K,K,np.eye(4),np.eye(4))
    assert v['depth_mae_m']==.5 # Foreground inconsistency is not hidden by mask.


def test_crop_does_not_recenter_prediction():
    gt=np.zeros((100,100,3));gt[40:50,40:50]=1
    mask=gt[:,:,0]>0;pred=np.roll(gt,20,axis=1)
    out=object_crops(pred,gt,mask)
    assert out['metrics']['mse']['value']>0
    assert object_crops(pred,gt,mask*False)['status']=='not_visible'
    with pytest.raises(ValueError):shape_pair(np.zeros((3,3)),cloud())


def test_reprojection_translated_camera_known_plane():
    depth=np.ones((10,10));K=np.array([[10,0,4.5],[0,10,4.5],[0,0,1.]])
    src=np.repeat(np.linspace(0,.9,10)[None,:,None],10,axis=0);src=np.repeat(src,3,axis=2)
    target=np.roll(src,-1,axis=1)
    pose=np.eye(4);pose[0,3]=.1
    r=reprojection(depth,depth,src,target,K,K,np.eye(4),pose)
    assert r['coverage']==pytest.approx(.9)
    assert r['rgb_mae']==pytest.approx(0)
    assert r['depth_mae_m']==pytest.approx(0)


def test_object_set_world_penalizes_duplicate_and_empty():
    g=cloud();ref=[Instance('gt',g)]
    result=compare_object_sets([Instance('p',g),Instance('duplicate',g)],ref,yaw_steps=4)
    assert result['macro_fscore_missing_as_zero']['world']['0.05m']==1
    assert result['object_set_world']['0.05m']['f1']==pytest.approx(2/3)
    assert compare_object_sets([],ref)['object_set_world']['0.05m']['f1']==0
    assert compare_object_sets([],[])['object_set_world']['0.05m']['f1'] is None
