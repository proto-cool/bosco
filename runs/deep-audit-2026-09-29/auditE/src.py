import sys, json, numpy as np, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'/Users/nickd/projects/bosco/scripts')
import brain_train as BT, brain_check as BC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
ant=BC.antenna(); Xtr,ytr=BT.harm('train'); Xva,yva=BT.harm('val')
its=json.load(open('/Users/nickd/projects/bosco/data/cache/v1-harm-dev/items.json'))['items']
src=np.array([it['source'] for it in its if it['task']=='harmful' and it['split']=='val'])
lr=LogisticRegression(C=1,class_weight='balanced',max_iter=5000).fit(ant(Xtr),ytr); s=lr.decision_function(ant(Xva))
lr7=LogisticRegression(C=1,class_weight='balanced',max_iter=5000).fit(Xtr,ytr); s7=lr7.decision_function(Xva)
for k in np.unique(src):
    q=src==k; print(k, q.sum(), 'logistic46 AUC %.3f  logistic768 AUC %.3f'%(roc_auc_score(yva[q],s[q]),roc_auc_score(yva[q],s7[q])))
