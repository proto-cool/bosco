import sys, numpy as np, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'/Users/nickd/projects/bosco/scripts')
import brain_train as BT, brain_check as BC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score as BA, roc_auc_score
from sklearn.neural_network import MLPClassifier
ant=BC.antenna(); Xtr,ytr=BT.harm('train'); Xva,yva=BT.harm('val')
A,B=ant(Xtr),ant(Xva)
print('antenna feature range', A.min(), A.max(), 'mean', A.mean(), 'sd per ch mean', A.std(0).mean())
for C in (0.01,0.1,1,10,100,1000):
    lr=LogisticRegression(C=C,class_weight='balanced',max_iter=5000).fit(A,ytr)
    cv=cross_val_score(LogisticRegression(C=C,class_weight='balanced',max_iter=5000),A,ytr,cv=5,scoring='balanced_accuracy').mean()
    print(f'C={C}: train-CV BA {cv:.4f}  val BA {BA(yva,lr.predict(B)):.4f} val AUC {roc_auc_score(yva,lr.decision_function(B)):.4f} train BA {BA(ytr,lr.predict(A)):.4f}')
lr=LogisticRegression(C=1,class_weight='balanced',max_iter=5000).fit(Xtr,ytr); print('768-d C=1 val BA',BA(yva,lr.predict(Xva)), 'AUC',roc_auc_score(yva,lr.decision_function(Xva)))
mlp=MLPClassifier((256,),max_iter=300,random_state=1,early_stopping=True).fit(A,ytr); print('MLP-256 on 46 ch val BA', BA(yva,mlp.predict(B)),'AUC',roc_auc_score(yva,mlp.predict_proba(B)[:,1]))
# random sparse expansion like KCs: 2000 random ReLU units thresholded to ~5% active, then logistic
rng=np.random.default_rng(1); W=rng.normal(size=(46,2000))*(rng.random((46,2000))<7/46)
H=A@W; th=np.quantile(H,0.95,axis=0); HH=np.maximum(H-th,0); HB=np.maximum(B@W-th,0)
lr=LogisticRegression(C=1,class_weight='balanced',max_iter=5000).fit(HH,ytr); print('random 2000-unit 5% sparse expansion + logistic val BA',BA(yva,lr.predict(HB)),'AUC',roc_auc_score(yva,lr.decision_function(HB)))
