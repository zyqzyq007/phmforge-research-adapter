"""Run an authored program while recording actual supported learning calls."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit',required=True,type=Path)
    parser.add_argument('--mode',choices=['fit','predict'],required=True);parser.add_argument('program');parser.add_argument('arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args();events=[];updates=[]
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
    from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,GradientBoostingClassifier,HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression,RidgeClassifier,SGDClassifier
    from sklearn.neighbors import KNeighborsClassifier,NearestCentroid
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC,LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge,LinearRegression,Lasso,ElasticNet,SGDRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR,LinearSVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.cross_decomposition import PLSRegression
    def patch(cls):
        original=cls.fit
        def fit(self,X,y=None,*a,**kw):
            event=dict(class_name=cls.__name__,rows=len(X),status='started');events.append(event);started=time.monotonic()
            result=original(self,X,y,*a,**kw)
            event.update(status='completed',seconds=time.monotonic()-started,
                fitted_classes=self.classes_.tolist() if hasattr(self,'classes_') else [],n_features=int(getattr(self,'n_features_in_',0)))
            return result
        cls.fit=fit
    for cls in [LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis,ExtraTreesClassifier,RandomForestClassifier,
        GradientBoostingClassifier,HistGradientBoostingClassifier,LogisticRegression,RidgeClassifier,SGDClassifier,
        KNeighborsClassifier,NearestCentroid,GaussianNB,MLPClassifier,SVC,LinearSVC,DecisionTreeClassifier,
        ExtraTreesRegressor,RandomForestRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor,
        Ridge,LinearRegression,Lasso,ElasticNet,SGDRegressor,KNeighborsRegressor,MLPRegressor,SVR,LinearSVR,
        DecisionTreeRegressor,PLSRegression]:patch(cls)
    source='\n'.join(p.read_text() for p in Path(args.program).parent.rglob('*.py'))
    if 'torch' in source:
        import torch
        for name in ['Adam','AdamW','SGD','RMSprop','Adagrad','Adadelta']:
            cls=getattr(torch.optim,name);original=cls.step
            def step(self,*a,_original=original,_name=name,**kw):
                params=[p for g in self.param_groups for p in g['params']]
                before=[p.detach().reshape(-1)[:16].clone() for p in params]
                result=_original(self,*a,**kw)
                changed=any(not torch.equal(b,p.detach().reshape(-1)[:16]) for b,p in zip(before,params))
                updates.append(dict(optimizer=_name,sampled_parameters_changed=changed));return result
            cls.step=step
    status='started'
    try:
        sys.argv=[args.program,*args.arguments]
        sys.path.insert(0,str(Path(args.program).parent))
        runpy.run_path(args.program,run_name='__main__');status='completed'
    except SystemExit as error:
        status='completed' if error.code in [0,None] else 'error';raise
    finally:
        args.audit.write_text(json.dumps(dict(status=status,mode=args.mode,estimator_fit_events=events,
            completed_estimator_fits=sum(e['status']=='completed' for e in events),optimizer_steps=len(updates),
            nonzero_sampled_optimizer_steps=sum(e['sampled_parameters_changed'] for e in updates),
            optimizer_parameter_change_is_sampled=True),indent=2)+'\n')


if __name__=='__main__':main()
