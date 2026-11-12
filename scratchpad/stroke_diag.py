import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
P=sys.argv[1]
lab=pd.read_csv(f'{P}/modeling/labels.csv')
C=['MI','Stroke','HF','AF','PAD']
models={'cox':('baselines_survival','cox_risk_{c}'),'xgb':('baselines_survival','xgb_surv_risk_{c}'),
        'tkg':('tgn_survival','cif_{c}_at_{h}d'),'pg':('patient_gnn_survival','cif_{c}_at_{h}d')}
h=1095
rows=[]
for m,(d,t) in models.items():
    pr=pd.read_csv(f'{P}/{d}/predictions_test.csv'); df=lab.merge(pr,on='subject_id')
    for c in C:
        s=df[t.format(c=c,h=h)]
        others=df[[t.format(c=o,h=h) for o in C if o!=c]].rank(pct=True).mean(axis=1)
        e,dur=df.endpoint_type,df.time_to_event_days
        pos=(e==c)&(dur<=h)
        comp=(e!=c)&(e!='censored')&(dur<h)
        free=dur>=h
        a_std=roc_auc_score(pos[pos|comp|free],s[pos|comp|free])
        a_cens=roc_auc_score(pos[pos|free],s[pos|free])            # negatives = event-free only
        a_comp=roc_auc_score(pos[pos|comp],s[pos|comp]) if comp.sum() else np.nan  # pos vs competing-event pts
        anyev=(e!='censored')&(dur<=h)
        rows.append(dict(model=m,cause=c,auc_standard=a_std,auc_vs_eventfree=a_cens,auc_vs_competing=a_comp,
             auc_for_any_event=roc_auc_score(anyev[anyev|free],s[anyev|free]),
             spearman_with_other_causes=spearmanr(s,others)[0]))
r=pd.DataFrame(rows).round(3); print(r.to_string())
