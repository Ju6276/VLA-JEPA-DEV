"""CPU contract tests; no checkpoint, CUDA, or robot access required."""
import ast
import importlib.util
import sys
import types
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'starVLA/model/modules/world_model'
# Load only numerical modules; do not initialize the production VLM registry.
package = types.ModuleType('foresight_test_modules')
package.__path__ = [str(BASE)]
sys.modules[package.__name__] = package

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

delta = load('foresight_test_modules.delta_jepa', BASE / 'delta_jepa.py')
aligned = load('foresight_test_modules.aligned_foresight', BASE / 'aligned_foresight.py')
evalmod = load('candidate_eval', ROOT / 'starVLA/tools/candidate_evaluation.py')


def test_horizon_contract():
    aligned.validate_horizons(8, 4, 40)
    aligned.validate_horizons(4, 4, 40)
    for h, k, H in [(4, 8, 40), (41, 4, 40), (1, 1, 40), (None, 4, 40), (True, 2, 40)]:
        with pytest.raises(ValueError): aligned.validate_horizons(h, k, H)


def test_current_encoding_input_independent_of_future():
    video = np.arange(2*1*6*2*2*3).reshape(2,1,6,2,2,3)
    current, target = aligned.endpoint_video_pair(video, 5, 8)
    changed = video.copy(); changed[:,:,1:] += 10000
    other, newtarget = aligned.endpoint_video_pair(changed, 5, 8)
    np.testing.assert_array_equal(current, other)
    assert not np.array_equal(target, newtarget)
    np.testing.assert_array_equal(current[:,:,0], video[:,:,0])
    np.testing.assert_array_equal(target[:,:,0], video[:,:,5])
    with pytest.raises(ValueError): aligned.endpoint_video_pair(video, 4, 8)


def test_endpoint_has_one_ordered_token_for_every_command():
    torch.manual_seed(7)
    model = aligned.EndpointActionEncoder(6, 12, 16, 8)
    a = torch.randn(1,8,6,requires_grad=True)
    out = model(a)
    assert out.shape == (1,8,16)
    out.square().sum().backward()
    assert (a.grad.abs().sum(-1) > 0).all()
    # Order changes the position/content associations, even with a single visual block.
    assert not torch.allclose(model(a.flip(1)).flip(1), out)
    with pytest.raises(ValueError): model(a[:,:4])


def test_endpoint_attention_allows_every_action_token():
    source = ast.parse((BASE/'vj2_modules.py').read_text())
    fn = next(x for x in source.body if isinstance(x,ast.FunctionDef) and x.name=='build_action_block_causal_attention_mask')
    ns={'torch':torch}; exec(compile(ast.Module(body=[fn],type_ignores=[]),'mask','exec'),ns)
    mask=ns['build_action_block_causal_attention_mask'](1,2,2,add_tokens=8)
    assert mask[8:, :8].all()  # every endpoint patch reads every evaluated command
    legacy=ns['build_action_block_causal_attention_mask'](3,2,2,add_tokens=8)
    assert not legacy[:12,12:].any()  # legacy predictor really is block causal


def test_factorized_prior_blocks_cross_group_history():
    torch.manual_seed(9)
    model = aligned.FactorizedActionPrior([4,2],3,hidden_dim=16,num_layers=2).eval()
    a=torch.randn(2,8,6); state=torch.randn(2,1,3)
    b=a.clone(); b[:,:,4:]+=3
    torch.testing.assert_close(model.predict_next(a,state)[:,:,:4], model.predict_next(b,state)[:,:,:4])
    # Swapping pairings preserves total separable energy.
    cross1=torch.cat([a[0:1,:,:4],a[1:2,:,4:]],-1)
    cross2=torch.cat([a[1:2,:,:4],a[0:1,:,4:]],-1)
    state=state[:1].expand(2,-1,-1)
    energy,_=aligned.weighted_prior_energy(model,a,state,[4,2],[4,2])
    crossed,_=aligned.weighted_prior_energy(model,torch.cat([cross1,cross2]),state,[4,2],[4,2])
    torch.testing.assert_close(energy.sum(),crossed.sum())


def test_joint_prior_can_use_other_command_group():
    torch.manual_seed(11)
    model=delta.ActionDynamicsPrior(6,3,16,2).eval()
    a=torch.randn(1,8,6); state=torch.randn(1,1,3)
    b=a.clone(); b[:,:,4:]+=3
    assert not torch.allclose(model.predict_next(a,state)[:,:,:4],model.predict_next(b,state)[:,:,:4])


def test_group_energy_reproduces_coordinate_average_and_trains():
    model=delta.ActionDynamicsPrior(6,3,16,2)
    a=torch.randn(3,8,6); state=torch.randn(3,1,3)
    e,groups=aligned.weighted_prior_energy(model,a,state,[4,2],[4,2])
    torch.testing.assert_close(e,model.energy(a,state))
    torch.testing.assert_close(e,groups[:,0]*2/3+groups[:,1]/3)
    e.mean().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_factorized_budget_matches_joint():
    model=aligned.FactorizedActionPrior([64,14],46,512,2)
    joint=delta.ActionDynamicsPrior(78,46,512,2)
    count=lambda m:sum(p.numel() for p in m.parameters())
    assert abs(count(model)-count(joint))/count(joint)<.01


def test_normalization_roundtrip_preserves_controller_commands():
    a=np.array([[1.,2.],[-2.,4.]])
    canonical,converted=evalmod.convert_normalization(a,[10,20],[2,3],[5,7],[4,8])
    np.testing.assert_allclose(converted*[4,8]+[5,7],canonical)
    np.testing.assert_allclose(canonical,a*[2,3]+[10,20])
    with pytest.raises(ValueError): evalmod.convert_normalization(a,[0,0],[0,1],[0,0],[1,1])


def test_regret_requires_independent_complete_outcomes():
    r=evalmod.selection_metrics([.1,.9,.8],[0,2,5])
    assert r=={'selected_index':1,'selected_return':2.,'candidate_set_regret':3.,'oracle_return':5.}
    with pytest.raises(ValueError): evalmod.selection_metrics([1,2],[None,3])


def resize_images(images, target_size):
    # Load the actual recursive preprocessing function without trainer imports.
    source=ast.parse((ROOT/'starVLA/training/trainer_utils/trainer_tools.py').read_text())
    fn=next(x for x in source.body if isinstance(x,ast.FunctionDef) and x.name=='resize_images')
    ns={'Image':__import__('PIL.Image',fromlist=['Image'])}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'resize','exec'),ns)
    return ns['resize_images'](images,target_size)


def framework_method(name):
    source=ast.parse((ROOT/'starVLA/model/framework/VLA_JEPA.py').read_text())
    cls=next(x for x in source.body if isinstance(x,ast.ClassDef) and x.name=='VLA_JEPA')
    fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name==name)
    ns={'torch':torch,'np':np,'Image':__import__('PIL.Image',fromlist=['Image']), 'Optional':__import__('typing').Optional, 'resize_images':resize_images}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'framework_method','exec'),ns)
    return ns[name]


def test_actual_endpoint_training_path_encodes_separate_observations():
    from PIL import Image
    class Harness:
        prediction_horizon=3
        aligned_endpoint=True
        num_temporal_frames=4
        config=types.SimpleNamespace(datasets=types.SimpleNamespace(vla_data={"resolution_size":2}))
        _encode_aligned_training_pair=framework_method('_encode_aligned_training_pair')
        _prepare_observation_images=framework_method('_prepare_observation_images')
        def _images_to_video_batch(self,images):
            return np.stack([np.stack([np.asarray(x[0])]*8)[None] for x in images])
        def _encode_video_batch(self,videos):
            self.seen.append(videos.copy())
            return torch.full((1,8,4),float(videos.mean()))
    h=Harness(); h.seen=[]
    video=np.stack([np.full((2,2,3),x,np.uint8) for x in [10,20,30,40]])[None]
    current,target,nt=h._encode_aligned_training_pair(
        [{'video':video,'prediction_horizon':3}],[[Image.fromarray(video[0,0])]])
    assert len(h.seen)==2 and (h.seen[0]==10).all() and (h.seen[1]==40).all()
    assert nt==2 and current.shape==(1,8,4) and target.shape==(1,2,4)
    with pytest.raises(ValueError):
        h._encode_aligned_training_pair([{'video':video,'prediction_horizon':2}],[])


def test_actual_endpoint_predictor_path_has_one_visual_block():
    class Predictor(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight=torch.nn.Parameter(torch.ones(1))
        def forward(self,x,a):
            self.shapes=(x.shape,a.shape)
            return x*self.weight + a.mean(dim=1,keepdim=True)
    h=types.SimpleNamespace(aligned_endpoint=True,num_temporal_frames=4,vj_predictor=Predictor())
    video=torch.randn(1,8,4); actions=torch.randn(1,6,4,requires_grad=True)
    current,target,pred=framework_method('_predict_future_latent')(h,video,actions)
    assert target is None and h.vj_predictor.shapes==(torch.Size([1,2,4]),torch.Size([1,6,4]))
    torch.testing.assert_close(current,video[:,-2:])
    pred.sum().backward()
    assert (actions.grad.abs().sum(-1)>0).all()


def test_scoring_scope_ignores_unevaluated_tail():
    torch.manual_seed(21)
    prior=delta.ActionDynamicsPrior(6,3,16,2).eval()
    h=types.SimpleNamespace(use_delta_jepa=True, aligned_endpoint=True,
        prediction_horizon=4,delta_jepa_blend_vlm_tokens=True,
        candidate_action_encoder=aligned.EndpointActionEncoder(6,12,16,4))
    a=torch.randn(1,8,6); b=a.clone(); b[:,4:]+=10
    tokens=torch.randn(1,24,16); state=torch.randn(1,1,3)
    cond=framework_method('_build_predictor_action_cond')
    torch.testing.assert_close(cond(h,tokens,a),cond(h,tokens,b))
    ea,_=aligned.weighted_prior_energy(prior,a[:,:4],state,[4,2],[4,2])
    eb,_=aligned.weighted_prior_energy(prior,b[:,:4],state,[4,2],[4,2])
    torch.testing.assert_close(ea,eb)
    c=a.clone(); c[:,3]+=1
    assert not torch.allclose(cond(h,tokens,a),cond(h,tokens,c))


def test_training_and_online_spatial_preprocessing_match():
    from PIL import Image
    raw=np.random.default_rng(3).integers(0,256,(13,19,3),dtype=np.uint8)
    h=types.SimpleNamespace(aligned_endpoint=True,
        config=types.SimpleNamespace(datasets=types.SimpleNamespace(
            vla_data={'resolution_size':8,'image_size':[6,6]})))
    preprocess=framework_method('_prepare_observation_images')
    online=preprocess(h,[[Image.fromarray(raw)]])[0][0]
    dataset_image=Image.fromarray(raw).resize((8,8))
    training=preprocess(h,[[dataset_image]])[0][0]
    np.testing.assert_array_equal(np.asarray(online),np.asarray(training))


def test_recording_roundtrip_and_integrity(tmp_path):
    import copy
    import json
    recorder=load('decision_recorder',ROOT/'tools/recording/save_decision.py')
    candidates=np.arange(24,dtype=float).reshape(1,2,4,3)/10
    response={'all_candidates':candidates,'normalized_actions':candidates[:,1],
        'execution_contract':{'H':4,'h':3,'K':3},'candidate_trace':{
        'goal_progress':[[1.,3.]],'prior_energy':[[1.,2.]],'joint_score':[[.9,2.8]],
        'prior_group_energy':[[[1.,1.],[2.,2.]]],'command_groups':[2,1],
        'command_group_weights':[2,1],'selected_index':[1],'beta':.1,
        'current_latent':[[0,1]],'goal_latent':[[1,0]],
        'observation_rgb':[np.zeros((2,2,3),dtype=np.uint8)],
        'goal_rgb':[np.ones((2,2,3),dtype=np.uint8)]}}
    normalization={'source_normalization_id':'train-v1','controller_id':'sonic-hash',
        'decoder_id':'decoder-hash','coordinate_order':['m0','m1','u0'],
        'center':[10,20,30],'scale':[2,3,4]}
    canonical=candidates*[2,3,4]+[10,20,30]
    kwargs={'controller_commands':canonical,'command_normalization':normalization}
    path=recorder.save_decision(tmp_path,'episode',1,{'state':[[0,0]]},response,{},**kwargs)
    record=json.loads((path/'record.json').read_text())
    assert record['schema_version']==2 and record['outcome'] is None
    assert record['execution_contract']=={'H':4,'h':3,'K':3}
    with np.load(path/'decision.npz') as bank:
        np.testing.assert_array_equal(bank['controller_commands'],canonical)
        np.testing.assert_array_equal(bank['prior_group_energy'],response['candidate_trace']['prior_group_energy'])
    with pytest.raises(FileExistsError):
        recorder.save_decision(tmp_path,'episode',1,{},response,{},**kwargs)
    with pytest.raises(ValueError,match='reproduce the controller'):
        recorder.save_decision(tmp_path,'episode',2,{},response,{},
            controller_commands=canonical+1,command_normalization=normalization)
    bad=copy.deepcopy(response);bad['candidate_trace']['prior_group_energy'][0][0][0]=99
    with pytest.raises(ValueError,match='Group residuals'):
        recorder.save_decision(tmp_path,'episode',3,{},bad,{})


def test_dataset_factory_excludes_padded_targets_and_actions():
    # Exercise the actual factory's index filter without decoding videos.
    source=ast.parse((ROOT/'starVLA/dataloader/lerobot_datasets.py').read_text())
    fn=next(x for x in source.body if isinstance(x,ast.FunctionDef)
            and x.name=='make_LeRobotSingleDataset')
    class FakeDataset:
        trajectory_ids=[0,1]
        trajectory_lengths=[5,3]
        def __init__(self,**kwargs):
            self._all_steps=[(e,t) for e,n in zip(self.trajectory_ids,self.trajectory_lengths) for t in range(n)]
        @property
        def all_steps(self): return self._all_steps
    class DataConfig:
        def __init__(self,**kwargs): pass
        def modality_config(self): return {}
        def transform(self): return None
    ns={'Path':Path,'LeRobotSingleDataset':FakeDataset,
        'ROBOT_TYPE_CONFIG_MAP':{'test':DataConfig},'ROBOT_TYPE_TO_EMBODIMENT_TAG':{'test':'tag'}}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'dataset_factory','exec',
                 flags=__import__('__future__').annotations.compiler_flag),ns)
    factory=ns['make_LeRobotSingleDataset']
    result=factory(Path('/unused'),'test','test',action_horizon=4,video_horizon=3,prediction_horizon=2)
    assert result.all_steps==[(0,0),(0,1)]
    result=factory(Path('/unused'),'test','test',action_horizon=2,video_horizon=5,prediction_horizon=4)
    assert result.all_steps==[(0,0)]
    with pytest.raises(ValueError,match='No unpadded'):
        factory(Path('/unused'),'test','test',action_horizon=8,video_horizon=3,prediction_horizon=2)
    with pytest.raises(ValueError,match='video_horizon'):
        factory(Path('/unused'),'test','test',action_horizon=4,video_horizon=4,prediction_horizon=2)
