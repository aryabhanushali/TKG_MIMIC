import numpy as np, pandas as pd
from scipy.stats import spearmanr
C=['MI','Stroke','HF','AF','PAD']
for k in range(5):
    f=f'tkg_output_ehrshot/cv/fold{k}'
    pr=pd.read_csv(f'{f}/patient_gnn_survival/predictions_test.csv')
    st=pd.read_csv(f'{f}/modeling/static_features.csv'); d=pr.merge(st,on='subject_id')
    lab=pd.read_csv(f'{f}/modeling/labels.csv'); ntr=((lab.split=='train')&(lab.endpoint_type=='Stroke')).sum()
    cv={c: d[f'cif_{c}_at_1095d'].std()/d[f'cif_{c}_at_1095d'].mean() for c in C}
    print(f'fold{k} train strokes={ntr}', 'CV(sd/mean):',{c:round(v,3) for c,v in cv.items()},
          'rho(stroke,age)=%.2f'%spearmanr(d.cif_Stroke_at_1095d,d.age_at_index)[0],
          'rho(HF,age)=%.2f'%spearmanr(d.cif_HF_at_1095d,d.age_at_index)[0],
          'mean stroke cif=%.4f'%d.cif_Stroke_at_1095d.mean())
h=pd.read_csv('tkg_output_ehrshot/cv/fold0/patient_gnn_survival/history.csv'); print(h.columns.tolist()); print(h.tail(3).to_string())
