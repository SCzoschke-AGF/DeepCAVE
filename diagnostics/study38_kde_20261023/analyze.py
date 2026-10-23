"""Read-only study-38 KDE diagnostic. Writes only beneath this script's directory."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/study38-mpl')
from pathlib import Path
from types import SimpleNamespace
import hashlib, json, math, sqlite3, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from statsmodels.nonparametric.kernel_density import KDEMultivariate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from deepcave.evaluators.kde import kde_encoding, kde_logpdf, fit_kde, split_good_bad
from deepcave.evaluators.coverage_candidates import numeric_space, generate_full_space_candidates
from deepcave.runs.converters.optuna import OptunaRun

OUT=Path(__file__).resolve().parent
IN=OUT/'inputs'
DB=Path('/home/sczoschke/projects/streamingmodel_NonRouterHHs/optuna.db')
NAME='cat_20261006-2021_random_opt-ap-v1_data:20260924T005634Z-2ad3'
RNG=np.random.default_rng(20261023)
def savejson(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist(),allow_nan=True))
def csv(name,rows):
    df=pd.DataFrame(rows);df.to_csv(OUT/name,index=False);return df

# A read transaction captures this study consistently; no Optuna RDBStorage is opened.
con=sqlite3.connect(DB.as_uri()+'?mode=ro',uri=True)
con.execute('BEGIN')
assert con.execute('select study_name from studies where study_id=38').fetchone()[0]==NAME
rows=con.execute('select trial_id,number,state from trials where study_id=38 order by number').fetchall()
records=[];pending=[]
for tid,num,status in rows:
    pars={n:(int(v) if json.loads(d)['name']=='IntDistribution' else v) for n,v,d in con.execute('select param_name,param_value,distribution_json from trial_params where trial_id=?',(tid,))}
    val=con.execute('select value,value_type from trial_values where trial_id=? and objective=0',(tid,)).fetchone()
    if status=='COMPLETE' and val and val[1]=='FINITE' and val[0] is not None and np.isfinite(val[0]):records.append(dict(trial_number=num,ap=val[0],**pars))
    if status in ('WAITING','RUNNING'):
        fixed=con.execute("select value_json from trial_system_attributes where trial_id=? and key='fixed_params'",(tid,)).fetchone()
        if fixed:pending.append(dict(trial_number=num,**json.loads(fixed[0])))
state_counts=dict(con.execute('select state,count(*) from trials where study_id=38 group by state'))
con.rollback();con.close()
df=csv('completed_trials.csv',records);pend=csv('proposed_configurations.csv',pending)
run=OptunaRun.from_path(IN);space=numeric_space(run);names=list(space);di=names.index('depth')
configs=df[names].to_dict('records');pc=pend[names].to_dict('records')
assert len(pc)==27
Y=df.ap.to_numpy();X,types,transforms=kde_encoding([SimpleNamespace(params=c) for c in configs],space)
cache=json.loads((IN/'candidate_scoring.json').read_text())['result'];metadata=cache['model']
assert df.trial_number.tolist()==metadata['training_trials']
with np.load(IN/'model.npz',allow_pickle=False) as z:
    models=[KDEMultivariate(z[f'{g}_data'],types,bw=z[f'{g}_bw']) for g in ('good','bad')]
G,B=split_good_bad(Y,np.arange(len(Y)),True)
assert df.iloc[G].trial_number.tolist()==metadata['good_trials']
assert np.allclose(models[0].data,X[G]) and np.allclose(models[1].data,X[B])
def enc(configurations):return kde_encoding([SimpleNamespace(params=c) for c in configurations],space)[0]
def logs(ms,queries):
    l,g=(kde_logpdf(m,queries) for m in ms);return l,g,l-g

def projected(ms,queries,keep):
    return logs([KDEMultivariate(m.data[:,keep],''.join(types[k] for k in keep),bw=m.bw[keep]) for m in ms],queries[:,keep])
def depth_logs(ms):
    return [kde_logpdf(KDEMultivariate(m.data[:,[di]],types[di],bw=m.bw[[di]]),np.arange(6)[:,None]) for m in ms]

def wilson(k,n):
    z=1.95996398454;p=k/n;center=(p+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n);return center-half,center+half
lmar,gmar=depth_logs(models)
summary=[]
for d,group in df.groupby('depth'):
    pos=group.index.to_numpy();ng=int(np.isin(pos,G).sum());nb=len(pos)-ng
    r=dict(depth=d,n=len(pos),median_ap=group.ap.median(),p90_ap=group.ap.quantile(.9),max_ap=group.ap.max(),good_count=ng,bad_count=nb,good_rate=ng/len(pos),empirical_ratio=(ng/len(G))/(nb/len(B)),kde_good_density=np.exp(lmar[int(d)-10]),kde_bad_density=np.exp(gmar[int(d)-10]),kde_ratio=np.exp(lmar[int(d)-10]-gmar[int(d)-10]))
    for fraction in (.1,.2):
        threshold=np.quantile(Y,1-fraction);k=int((group.ap>=threshold).sum());lo,hi=wilson(k,len(group));r.update({f'top{int(fraction*100)}_rate':k/len(group),f'top{int(fraction*100)}_count':k,f'top{int(fraction*100)}_low':lo,f'top{int(fraction*100)}_high':hi})
    samples=RNG.choice(group.ap.to_numpy(),size=(1500,len(group)),replace=True)
    for label,q in (('median',.5),('p90',.9)):
        lo,hi=np.quantile(np.quantile(samples,q,axis=1),[.025,.975]);r[f'{label}_low']=lo;r[f'{label}_high']=hi
    summary.append(r)
emp=csv('empirical_depth_summary.csv',summary)
pool=cache['full_space_candidates'];pool_configs=[c['configuration'] for c in pool];PX=enc(pool_configs)
l,g,sc=logs(models,PX);cached=np.array([c['kde_score'] for c in pool]);maxerr=float(np.max(abs(sc-cached)));assert maxerr<1e-9
recreated,sampling=generate_full_space_candidates(run,cache['options'])
assert [c['configuration'] for c in recreated]==pool_configs
pl,pg,ps=logs(models,enc(pc));proposal_rows=[];counter=[]
for i,c in enumerate(pc):
    match=[j for j,p in enumerate(pool_configs) if p==c]
    proposal_rows.append(dict(trial_number=int(pend.iloc[i].trial_number),**c,log_l=pl[i],log_g=pg[i],score=ps[i],pool_match_count=len(match),pool_rank=int(1+(sc>ps[i]+1e-12).sum())))
    cc=[dict(c,depth=d) for d in range(10,16)];ll,gg,ss=logs(models,enc(cc))
    for j,d in enumerate(range(10,16)):counter.append(dict(proposal_trial=int(pend.iloc[i].trial_number),depth=d,log_l=ll[j],log_g=gg[j],score=ss[j],winner=d==10+int(np.argmax(ss))))
proposals=csv('proposal_scores.csv',proposal_rows);cf=csv('proposal_depth_counterfactuals.csv',counter)
pool_table=csv('pool_scores.csv',[dict(id=c['id'],**c['configuration'],log_l=ll,log_g=gg,score=s) for c,ll,gg,s in zip(pool,l,g,sc)])
concentration=[]
for k in (27,100,500,10000):
    top=np.argsort(-sc,kind='stable')[:k]
    concentration.extend(dict(top_k=k,depth=d,count=int((pool_table.iloc[top].depth==d).sum()),fraction=float((pool_table.iloc[top].depth==d).mean())) for d in range(10,16))
conc=csv('pool_concentration.csv',concentration)
# Algebraic conditional density ratio: joint minus projected other-parameter ratio.
# This is not an independent feature attribution or a product of marginal KDEs.
rest=[j for j in range(len(names)) if j!=di]
_,_,restscore=projected(models,enc(pc),rest)
proposals['other_parameter_log_ratio']=restscore
proposals['conditional_depth_log_ratio']=ps-restscore
proposals.to_csv(OUT/'proposal_scores.csv',index=False)
parity=float(np.max(abs(np.log(models[0].pdf(enc(pc)))-pl)))

# Chronological prospective-style evaluation; labels use training targets alone.
def top_weights(scores,fraction=.2):
    k=math.ceil(len(scores)*fraction);cut=np.sort(scores)[-k];above=scores>cut;tie=scores==cut;weights=above.astype(float);weights[tie]=(k-above.sum())/tie.sum();return weights/k

def metrics(y,ap,s):
    w=top_weights(s)
    return dict(roc_auc=roc_auc_score(y,s) if len(np.unique(y))==2 else float('nan'),average_precision=average_precision_score(y,s) if y.sum() else float('nan'),precision_top20=float(w@y),mean_ap_top20=float(w@ap),delta_ap_top20=float(w@ap-ap.mean()),spearman_ap=float(spearmanr(s,ap).statistic) if np.ptp(s)>0 else 0.)

foldrows=[];holdrows=[];depthrows=[];sensitivity=[];fitdiags={};rocdata=[]
for frac in (.6,.8):
    n=int(len(df)*frac);end=int(len(df)*.8) if frac==.6 else len(df);train=np.arange(n);test=np.arange(n,end)
    good,bad=split_good_bad(Y,train,True);ms=[];diagnostics=[]
    for members in (good,bad):
        m,info=fit_kde(X[members],types,space);assert info['success'];ms.append(m);diagnostics.append(info)
    fitdiags[str(frac)]=diagnostics
    threshold=float(Y[good[-1]]);y=(Y[test]>=threshold).astype(int);actual=Y[test]
    ltest,gtest,kdescore=logs(ms,X[test]);_,_,nodepth=projected(ms,X[test],rest)
    depthrate={d:(np.sum((df.depth.to_numpy()[train]==d)&np.isin(train,good))+1)/(np.sum(df.depth.to_numpy()[train]==d)+2) for d in range(10,16)}
    baselines={'KDE':kdescore,'Depth-only':np.array([depthrate[d] for d in df.depth.iloc[test]]),'KDE excluding depth (projected)':nodepth,'Constant':np.full(len(test),len(good)/len(train))}
    for model,scores in baselines.items():
        result=metrics(y,actual,scores)
        boot=[]
        for b in range(1000):
            ix=RNG.integers(0,len(test),len(test));boot.append(metrics(y[ix],actual[ix],scores[ix]))
        row=dict(split=f'{int(frac*100)}%→next20%',model=model,n_train=n,n_test=len(test),n_good=int(y.sum()),threshold=threshold,**result)
        for key in ('roc_auc','average_precision','precision_top20','delta_ap_top20','spearman_ap'):
            lo,hi=np.nanquantile([v[key] for v in boot],[.025,.975]);row[key+'_low']=lo;row[key+'_high']=hi
        foldrows.append(row)
        if model in ('KDE','Depth-only'):rocdata.append((frac,model,y,scores))
    for j,t in enumerate(test):holdrows.append(dict(split=frac,trial_number=int(df.iloc[t].trial_number),depth=int(df.iloc[t].depth),ap=Y[t],good=int(y[j]),log_l=ltest[j],log_g=gtest[j],score=kdescore[j],depth_baseline=baselines['Depth-only'][j]))
    for d in range(10,16):
        mask=df.depth.to_numpy()[test]==d
        if mask.sum():depthrows.append(dict(split=frac,depth=d,n=int(mask.sum()),n_good=int(y[mask].sum()),mean_ap=float(actual[mask].mean()),**metrics(y[mask],actual[mask],kdescore[mask])))
    flat=[dict(c,depth=d) for c in pc for d in range(10,16)];ss=logs(ms,enc(flat))[2].reshape(27,6)
    sp=logs(ms,PX)[2];top=np.argsort(-sp,kind='stable')[:27];lm,gm=depth_logs(ms)
    sensitivity.append(dict(subset=f'chronological_{frac}',n_train=n,depth15_counterfactual_wins=int((ss.argmax(1)==5).sum()),depth15_top27=int((pool_table.iloc[top].depth==15).sum()),marginal_winner=int(10+np.argmax(lm-gm))))
validation=csv('heldout_metrics.csv',foldrows);held=csv('heldout_predictions.csv',holdrows);csv('heldout_by_depth.csv',depthrows)
# Refit random 80% subsets: descriptive stability, not additional independent tests.
for seed in range(20):
    train=np.sort(np.random.default_rng(seed).choice(len(df),int(.8*len(df)),replace=False));good,bad=split_good_bad(Y,train,True);ms=[]
    for members in (good,bad):
        m,info=fit_kde(X[members],types,space);assert info['success'];ms.append(m)
    qs=logs(ms,enc([dict(c,depth=d) for c in pc for d in range(10,16)]))[2].reshape(27,6)
    scores=logs(ms,PX)[2];top=np.argsort(-scores,kind='stable')[:27];lm,gm=depth_logs(ms)
    sensitivity.append(dict(subset=f'random80_seed{seed}',n_train=len(train),depth15_counterfactual_wins=int((qs.argmax(1)==5).sum()),depth15_top27=int((pool_table.iloc[top].depth==15).sum()),marginal_winner=int(10+np.argmax(lm-gm))))
sens=csv('subset_stability.csv',sensitivity)
# Quantify conditional vs marginal preferences across the full candidate contexts.
allcf=logs(models,enc([dict(c,depth=d) for c in pool_configs for d in range(10,16)]))[2].reshape(len(pool),6)
poolwins=np.bincount(allcf.argmax(1),minlength=6)
provenance=dict(study=NAME,study_id=38,objective='Average Precision (Objective0), maximize',database=str(DB),database_access='sqlite mode=ro; one SELECT transaction; rollback and close',states=state_counts,n_completed=len(df),n_good=len(G),n_bad=len(B),good_threshold=float(Y[G[-1]]),top10_threshold=float(np.quantile(Y,.9)),top20_quantile_threshold=float(np.quantile(Y,.8)),settings=cache['options'],types=types,transforms=transforms,production_diagnostics=metadata['diagnostics'],chronological_fit_diagnostics=fitdiags,max_cached_score_error=maxerr,max_statsmodels_logpdf_error=parity,generated_pool_exact_match=True,proposed_pool_ranks=proposals.pool_rank.tolist(),proposal_depth_counts=pend.depth.value_counts().to_dict(),proposal_depth_wins=cf[cf.winner].depth.value_counts().to_dict(),pool_depth_wins=dict(zip(range(10,16),poolwins.tolist())),source_hashes={n:hashlib.sha256((Path('deepcave/evaluators')/n).read_bytes()).hexdigest() for n in ('kde.py','coverage_candidates.py')},input_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in IN.iterdir()},python=sys.version)
savejson('provenance.json',provenance)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(12,8))
axes[0,0].boxplot([df[df.depth==d].ap for d in range(10,16)],tick_labels=list(range(10,16)),showfliers=False);axes[0,0].plot(range(1,7),emp.max_ap,'r^',label='Maximum');axes[0,0].set(xlabel='Depth',ylabel='Average Precision',title='Observed distributions (unequal counts)');axes[0,0].legend()
axes[0,1].plot(emp.depth,emp.empirical_ratio,'o-',label='Empirical P(depth|Good)/P(depth|Bad)');axes[0,1].plot(emp.depth,emp.kde_ratio,'s-',label='KDE marginal ratio');axes[0,1].axhline(1,color='gray',ls=':');axes[0,1].set(xlabel='Depth',ylabel='Good/Bad density ratio',title='Good = training top 20%');axes[0,1].legend(fontsize=8)
for offset,(key,label) in enumerate(((10000,'Unranked pool'),(27,'Top 27'))):
    sub=conc[conc.top_k==key];axes[1,0].bar(sub.depth+(offset-.5)*.34,sub.fraction,width=.34,label=label)
axes[1,0].set(xlabel='Depth',ylabel='Fraction',title='Concentration appears after ranking');axes[1,0].legend()
for frac,model,y,score in rocdata:
    fpr,tpr,_=roc_curve(y,score);axes[1,1].plot(fpr,tpr,label=f'{int(frac*100)}% train: {model} (AUC {roc_auc_score(y,score):.2f})')
axes[1,1].plot([0,1],[0,1],'k:',alpha=.4);axes[1,1].set(xlabel='False positive rate',ylabel='True positive rate',title='Chronological held-out identification');axes[1,1].legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT/'diagnostic_overview.png',dpi=170);fig.savefig(OUT/'diagnostic_overview.pdf');plt.close(fig)
pivot=cf.pivot(index='proposal_trial',columns='depth',values='score');relative=pivot.sub(pivot.max(axis=1),axis=0)
fig,ax=plt.subplots(figsize=(8,9));im=ax.imshow(relative,vmin=-8,vmax=0,cmap='viridis',aspect='auto');ax.set_xticks(range(6),range(10,16));ax.set_yticks(range(27),pivot.index);ax.set(xlabel='Counterfactual depth (other parameters fixed)',ylabel='Pending trial number',title='Full joint KDE score relative to best depth per configuration');fig.colorbar(im,ax=ax,label='log ratio − row maximum');fig.tight_layout();fig.savefig(OUT/'depth_counterfactuals.png',dpi=170);plt.close(fig)
print('EMPIRICAL\n',emp.to_string(index=False));print('VALIDATION\n',validation[['split','model','roc_auc','average_precision','precision_top20','delta_ap_top20','spearman_ap']].to_string(index=False));print('PROVENANCE\n',json.dumps({k:provenance[k] for k in ('max_cached_score_error','proposal_depth_counts','proposal_depth_wins','pool_depth_wins','proposed_pool_ranks')},indent=2));print('STABILITY\n',sens.to_string(index=False))
