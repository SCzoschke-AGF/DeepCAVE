"""Additional uncertainty and implementation audits; input files only, separate outputs."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/study38-mpl')
from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.stats import fisher_exact, chi2_contingency
from sklearn.metrics import roc_auc_score,average_precision_score
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from deepcave.evaluators.kde import kde_logpdf
O=Path(__file__).resolve().parent
p=json.loads((O/'provenance.json').read_text());df=pd.read_csv(O/'completed_trials.csv');h=pd.read_csv(O/'heldout_predictions.csv');s=pd.read_csv(O/'empirical_depth_summary.csv');q=pd.read_csv(O/'proposal_scores.csv');c=pd.read_csv(O/'proposal_depth_counterfactuals.csv');pool=pd.read_csv(O/'pool_scores.csv')
rng=np.random.default_rng(915)
paired=[]
for split,group in h.groupby('split'):
 y=group.good.to_numpy();k=group.score.to_numpy();d=group.depth_baseline.to_numpy();a=[]
 for _ in range(2500):
  ix=rng.integers(0,len(y),len(y))
  if len(np.unique(y[ix]))<2:continue
  a.append([roc_auc_score(y[ix],k[ix])-roc_auc_score(y[ix],d[ix]),average_precision_score(y[ix],k[ix])-average_precision_score(y[ix],d[ix])])
 a=np.array(a);paired.append(dict(split=split,auc_gain=roc_auc_score(y,k)-roc_auc_score(y,d),auc_gain_low=np.quantile(a[:,0],.025),auc_gain_high=np.quantile(a[:,0],.975),ap_gain=average_precision_score(y,k)-average_precision_score(y,d),ap_gain_low=np.quantile(a[:,1],.025),ap_gain_high=np.quantile(a[:,1],.975)))
pd.DataFrame(paired).to_csv(O/'paired_kde_vs_depth.csv',index=False)
# Within-depth discrimination, stratified within each chronological test cohort.
within=[]
for split,group in h.groupby('split'):
 concordance=[]
 for depth,g in group.groupby('depth'):
  pos=g[g.good==1].score.to_numpy();neg=g[g.good==0].score.to_numpy()
  if len(pos) and len(neg):
   mat=pos[:,None]-neg[None,:];concordance.extend(((mat>0)+.5*(mat==0)).ravel())
 within.append(dict(split=split,same_depth_pairs=len(concordance),same_depth_pairwise_auc=np.mean(concordance)))
pd.DataFrame(within).to_csv(O/'within_depth_discrimination.csv',index=False)
# Depth marginal normalized on allowed finite lattice; does boundary truncation create preference?
l=s.kde_good_density.to_numpy();g=s.kde_bad_density.to_numpy();s['finite_domain_ratio']=(l/l.sum())/(g/g.sum());s[['depth','kde_ratio','finite_domain_ratio']].to_csv(O/'finite_domain_depth_ratio.csv',index=False)
comparisons=[]
for other in (12,13,14):
 a=s[s.depth==15].iloc[0];b=s[s.depth==other].iloc[0];table=[[a.good_count,a.bad_count],[b.good_count,b.bad_count]];odds,pval=fisher_exact(table)
 comparisons.append(dict(comparison=f'15 vs {other}',odds_ratio=odds,p_unadjusted=pval))
pd.DataFrame(comparisons).to_csv(O/'depth_good_rate_comparisons.csv',index=False)
# Direct self-check of logpdf against statsmodels, and analytic LOO gradient finite differences.
with np.load(O/'inputs/model.npz') as z:
 types=p['types'];names=list(p['transforms']);di=names.index('depth');models=[KDEMultivariate(z[f'{g}_data'],types,bw=z[f'{g}_bw']) for g in ('good','bad')]
 from deepcave.evaluators.kde import loo_objective
 from scipy.optimize._numdiff import approx_derivative
 gradient=[]
 for m in models:
  continuous=np.array([x=='c' for x in types]);theta=m.bw.copy();theta[continuous]=np.log(theta[continuous]);f=loo_objective(m.data,types);numeric=approx_derivative(lambda t:f(t)[0],theta,method='3-point').ravel();analytic=f(theta)[1];gradient.append(float(abs(numeric-analytic).max()))
 # A product of per-feature marginal KDEs is only a diagnostic alternative, not production.
 from deepcave.evaluators.coverage_candidates import numeric_space,_encode
 from deepcave.runs.converters.optuna import OptunaRun
 run=OptunaRun.from_path(O/'inputs');space=numeric_space(run);queries=_encode(pool[names].to_dict('records'),space)
 score=np.zeros(len(queries))
 for j,kind in enumerate(types):
  for sign,m in ((1,models[0]),(-1,models[1])):
   mm=KDEMultivariate(m.data[:,[j]],kind,bw=m.bw[[j]]);score+=sign*kde_logpdf(mm,queries[:,[j]])
 naive=pool.iloc[np.argsort(-score)[:27]].depth.value_counts().to_dict()
 # Near-ties quantify how confidently each context ranks its best depth.
 pivot=c.pivot(index='proposal_trial',columns='depth',values='score');margin=np.sort(pivot.to_numpy(),axis=1)[:,-1]-np.sort(pivot.to_numpy(),axis=1)[:,-2]
 result=dict(paired=paired,within_depth=within,gradient_max_absolute_error=gradient,finite_good_mass=float(l.sum()),finite_bad_mass=float(g.sum()),finite_renormalized_winner=int(s.iloc[np.argmax(s.finite_domain_ratio)].depth),naive_independent_marginal_top27_depth_counts=naive,counterfactual_winner_margin_median=float(np.median(margin)),counterfactual_winner_margin_min=float(margin.min()),conditional_depth_log_ratio_range=[q.conditional_depth_log_ratio.min(),q.conditional_depth_log_ratio.max()],conditional_depth_log_ratio_median=q.conditional_depth_log_ratio.median(),marginal_depth15_log_ratio=float(np.log(s[s.depth==15].kde_ratio.iloc[0])),pool_depth_counts=pool.depth.value_counts().to_dict(),fisher= comparisons)
 (O/'supplement.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x))
 print(json.dumps(result,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x))
