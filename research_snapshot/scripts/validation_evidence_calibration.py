"""Exploratory update-specific calibration of real validation decisions.

Nested/grouped validation and inverse-error fusion are existing ideas. This
module is not a novelty claim and never reads deployment observations/labels.
"""
import math

def context(conditions):
    return 'c_'+'_'.join(str(int(x)) for x in sorted(conditions))

def mean(values):
    assert values and all(math.isfinite(x) for x in values)
    return sum(values)/len(values)

def choose(incumbent,candidate,delta):
    return candidate if delta < -1e-12 else incumbent

def cv_weight(cv_regret,validation_regret):
    assert 0 <= cv_regret <= 1 and 0 <= validation_regret <= 1
    total=cv_regret+validation_regret
    return validation_regret/total if total else .5

def decide(table,pairs,conditions):
    """table[program][training_context]['scores'][target]['record_error']."""
    assert len(conditions)==len(set(conditions))==3
    def loss(program,train,target):
        value=table[program][context(train)]['scores'][target]['record_error']
        assert math.isfinite(value) and 0<=value<=1
        return value
    rows=[]
    for pair in pairs:
        old,new=pair['incumbent'],pair['candidate'];outer=[]
        for held in conditions:
            train=[c for c in conditions if c!=held]
            inner={p:mean([loss(p,[j for j in train if j!=k],'train_'+str(int(k))) for k in train]) for p in [old,new]}
            validation={p:mean([loss(p,train,t) for t in ['development','independent_check']]) for p in [old,new]}
            target={p:loss(p,train,'train_'+str(int(held))) for p in [old,new]}
            selected_cv=choose(old,new,inner[new]-inner[old])
            selected_val=choose(old,new,validation[new]-validation[old])
            oracle=min(target.values())
            outer.append(dict(held_condition=held,training_conditions=train,inner_CV_errors=inner,
                full_validation_errors=validation,outer_errors=target,CV_selected=selected_cv,
                validation_selected=selected_val,CV_selection_regret=target[selected_cv]-oracle,
                validation_selection_regret=target[selected_val]-oracle))
        cv_regret=mean([x['CV_selection_regret'] for x in outer])
        val_regret=mean([x['validation_selection_regret'] for x in outer])
        cv_errors={p:mean([loss(p,[j for j in conditions if j!=k],'train_'+str(int(k))) for k in conditions]) for p in [old,new]}
        val_errors={p:mean([loss(p,conditions,t) for t in ['development','independent_check']]) for p in [old,new]}
        check_errors={p:loss(p,conditions,'independent_check') for p in [old,new]}
        worst={p:max([loss(p,[j for j in conditions if j!=k],'train_'+str(int(k))) for k in conditions]+
                     [loss(p,conditions,t) for t in ['development','independent_check']]) for p in [old,new]}
        delta_cv=cv_errors[new]-cv_errors[old];delta_val=val_errors[new]-val_errors[old]
        weight=cv_weight(cv_regret,val_regret)
        margins=dict(keep=0,CV=delta_cv,full_validation_equal_conditions=delta_val,
            last_check_only=check_errors[new]-check_errors[old],equal_CV_and_full_validation=.5*(delta_cv+delta_val),
            worst_individual_legal_condition=worst[new]-worst[old],
            pair_specific=weight*delta_cv+(1-weight)*delta_val)
        rows.append(dict(**pair,outer_calibration=outer,calibration_CV_regret=cv_regret,
            calibration_validation_regret=val_regret,weight_CV=weight,full_CV_errors=cv_errors,
            full_validation_errors=val_errors,full_check_errors=check_errors,
            methods={name:dict(program=choose(old,new,delta),decision_margin=delta) for name,delta in margins.items()}))
    shared=cv_weight(mean([r['calibration_CV_regret'] for r in rows]),mean([r['calibration_validation_regret'] for r in rows]))
    for row in rows:
        old,new=row['incumbent'],row['candidate']
        delta=shared*(row['full_CV_errors'][new]-row['full_CV_errors'][old])+(1-shared)*(row['full_validation_errors'][new]-row['full_validation_errors'][old])
        row['methods']['shared_calibration']=dict(program=choose(old,new,delta),decision_margin=delta)
    return dict(pairs=rows,shared_weight_CV=shared,final_data_read=False,
                informative_view_disagreements=sum(r['methods']['CV']['program']!=r['methods']['full_validation_equal_conditions']['program'] for r in rows),
                new_meta_model_fits=0)
