"""Genuine scenario-specific CPU regressors for measured FEMTO observations."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from real_rul_model import RULNetwork, digest_state, errors, write_json


def predict(model,x,scale):
    model.eval()
    with torch.no_grad():return np.maximum(0,torch.cat([model(b) for b in x.split(256)]).numpy()*scale)


def fit(a,y,architecture,epochs,config,seed,validation=None):
    torch.manual_seed(seed);torch.set_num_threads(config['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    mean=a.mean(0);std=np.maximum(a.std(0),1e-6);target_scale=max(float(np.max(y)),1)
    x=torch.from_numpy(((a-mean)/std).astype(np.float32)[:,None,:]);target=torch.from_numpy((y/target_scale).astype(np.float32))
    model=RULNetwork(architecture,1,a.shape[1]);initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    opt=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    best=float('inf');chosen=None;chosen_epoch=None;history=[];steps=0;gradients=0
    if validation is not None:
        vx=torch.from_numpy(((validation[0]-mean)/std).astype(np.float32)[:,None,:])
        initial_validation=errors(predict(model,vx,target_scale),validation[1])
    else:initial_validation=None
    for epoch in range(1,epochs+1):
        model.train();total=0.
        for indices in torch.randperm(len(x)).split(config['batch_size']):
            opt.zero_grad(set_to_none=True)
            loss=((model(x[indices])-target[indices])**2).mean();loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            gradients+=int(bool(torch.isfinite(norm)) and float(norm)>0)
            opt.step();steps+=1;total+=float(loss.detach())*len(indices)
        row={'epoch':epoch,'train_mse_normalized':total/len(x)}
        if validation is not None:
            row['development']=errors(predict(model,vx,target_scale),validation[1])
            better=row['development']['mae']<best
            if better:best=row['development']['mae']
        else:better=True
        history.append(row)
        if better:chosen={k:v.detach().clone() for k,v in model.state_dict().items()};chosen_epoch=epoch
    checkpoint={'state_dict':chosen,'mean':torch.from_numpy(mean),'std':torch.from_numpy(std),
        'target_scale':target_scale,'architecture':architecture,'features':a.shape[1]}
    untrained=dict(checkpoint,state_dict=initial)
    audit={'optimizer_steps':steps,'nonzero_gradient_steps':gradients,
        'changed_parameter_tensors':sum(not torch.equal(initial[k],chosen[k]) for k in chosen),
        'initial_state_sha256':digest_state(initial),'selected_state_sha256':digest_state(chosen),
        'selected_epoch':chosen_epoch,'epochs_executed':epochs,'train_rows':len(a),
        'initial_validation':initial_validation,'history':history,'target_unit':'seconds'}
    return checkpoint,untrained,audit


def train(data,output,architecture='mlp',epochs_override=None):
    data=Path(data);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    config=json.loads((data/'training_config.json').read_text())
    assert architecture=='mlp','Only the qualified MLP backend is supplied for measured bearing features'
    epochs=config['epochs'] if epochs_override is None else epochs_override
    assert isinstance(epochs,int) and not isinstance(epochs,bool) and epochs>=1
    raw=np.load(data/'train.npz',allow_pickle=False);val=np.load(data/'val.npz',allow_pickle=False)
    assert set(config['fit_unit_ids']).isdisjoint(config['validation_unit_ids'])
    assert set(config['fit_unit_ids']+config['validation_unit_ids'])==set(raw['unit_ids'].tolist())
    assert np.isfinite(raw['features']).all() and np.isfinite(raw['labels']).all()
    models=[];untrained=[];audits=[]
    for scenario in [1,2,3]:
        mask=(raw['scenario_ids']==scenario)&np.isin(raw['unit_ids'],config['fit_unit_ids'])
        vm=val['scenario_ids']==scenario
        assert mask.any() and vm.any()
        model,_,development=fit(raw['features'][mask],raw['labels'][mask],architecture,epochs,config,
            config['seed']+scenario,(val['features'][vm],val['labels'][vm]))
        all_training=raw['scenario_ids']==scenario
        final,initial,refit=fit(raw['features'][all_training],raw['labels'][all_training],architecture,
            development['selected_epoch'],config,config['seed']+100+scenario)
        final['scenario_id']=scenario;initial['scenario_id']=scenario
        models.append(final);untrained.append(initial)
        audits.append({'scenario_id':scenario,'development':development,'final_refit':refit,
            'fit_unit_ids':sorted(set(raw['unit_ids'][mask].tolist())),
            'validation_unit_ids':sorted(set(val['unit_ids'][vm].tolist())),
            'final_training_unit_ids':sorted(set(raw['unit_ids'][all_training].tolist()))})
    common={'dataset':'FEMTO','format':'femto_npz','prediction_unit':'seconds'}
    torch.save(dict(common,models=models),output/'model.pt')
    torch.save(dict(common,models=untrained),output/'untrained_model.pt')
    audit={'dataset':'FEMTO','scenarios':audits,'test_labels_accessed':False,
        'train_validation_group_overlap':[],'real_training':True,
        'optimizer_steps':sum(x[p]['optimizer_steps'] for x in audits for p in ['development','final_refit']),
        'nonzero_gradient_steps':sum(x[p]['nonzero_gradient_steps'] for x in audits for p in ['development','final_refit']),
        'changed_parameter_tensors':sum(x['final_refit']['changed_parameter_tensors'] for x in audits),
        'normalization':'fitting data only; validation excluded during selection',
        'development_score_scope':'held-out learning-bearing prefixes; final model refit includes those learning bearings',
        'training_source':'Original measured learning runs; annotated terminal-time remaining seconds'}
    write_json(output/'training_audit.json',audit);return audit


def infer(checkpoint_path,input_path,output_path):
    torch.set_num_threads(2)
    checkpoint=torch.load(checkpoint_path,weights_only=True,map_location='cpu')
    assert checkpoint['format']=='femto_npz'
    raw=np.load(input_path,allow_pickle=False)
    assert 'labels' not in raw,'Inference inputs may not contain targets'
    assert raw['dataset'].item()=='FEMTO' and np.isfinite(raw['features']).all()
    ids=raw['sample_ids'];assert len(ids)==len(set(ids.tolist()))==len(raw['features'])
    values=np.full(len(ids),np.nan)
    for item in checkpoint['models']:
        mask=raw['scenario_ids']==item['scenario_id']
        if not mask.any():continue
        model=RULNetwork(item['architecture'],1,item['features']);model.load_state_dict(item['state_dict'])
        x=(raw['features'][mask]-item['mean'].numpy())/item['std'].numpy()
        values[mask]=predict(model,torch.from_numpy(x.astype(np.float32)[:,None,:]),item['target_scale'])
    assert np.isfinite(values).all() and (values>=0).all()
    result={'predictions':[{'sample_id':str(i),'unit_id':int(u),'scenario_id':int(s),'predicted_rul':float(v)}
        for i,u,s,v in zip(ids,raw['unit_ids'],raw['scenario_ids'],values)],'unit':'seconds'}
    write_json(output_path,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('train');a.add_argument('--data',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--epochs',type=int,default=30)
    a=sub.add_parser('infer');a.add_argument('--model',type=Path,required=True);a.add_argument('--input',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='train':train(a.data,a.output,epochs_override=a.epochs)
    else:infer(a.model,a.input,a.output)
