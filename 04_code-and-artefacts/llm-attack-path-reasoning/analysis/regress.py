import sys, os, math, importlib.util, numpy as np, pandas as pd
art=sys.argv[1] if len(sys.argv)>1 else "."  # folder that holds engine.py
spec=importlib.util.spec_from_file_location("st", os.path.join(art,"stats.py")); st=importlib.util.module_from_spec(spec); spec.loader.exec_module(st)
r=pd.read_csv('analysis/runs_of_record.csv'); r=r[r.scored & (r.system=='C1')]
def fit(sub,label):
    y=sub.success.values.astype(float); d=sub.depth.values.astype(float); k=sub.k.values.astype(float)/10
    cl=list(sub.scenario); n=len(y)
    dc,kc=d-d.mean(),k-k.mean()
    X2=np.column_stack([np.ones(n),dc,kc]); X3=np.column_stack([np.ones(n),dc,kc,dc*kc])
    out={'label':label,'n':n,'succ':int(y.sum()),'fail':int(n-y.sum())}
    b2,ll2,ok2=st.irls(X2,y); b3,ll3,ok3=st.irls(X3,y)
    se2=np.sqrt(np.diag(st.cluster_robust_cov(X2,y,b2,cl))); se3=np.sqrt(np.diag(st.cluster_robust_cov(X3,y,b3,cl)))
    res=[]
    for j,nm in ((1,'depth'),(2,'distractors')):
        keep=[i for i in range(3) if i!=j]; _,llr,_=st.irls(X2[:,keep],y); chi=2*(ll2-llr)
        res.append((nm,b2[j],se2[j],math.exp(b2[j]),math.exp(b2[j]-1.96*se2[j]),math.exp(b2[j]+1.96*se2[j]),chi,st.chi2_sf(chi,1)))
    chi=2*(ll3-ll2)
    res.append(('interaction',b3[3],se3[3],math.exp(b3[3]),math.exp(b3[3]-1.96*se3[3]),math.exp(b3[3]+1.96*se3[3]),chi,st.chi2_sf(chi,1)))
    print(f"\n== {label}: n={n}, successes={out['succ']}, failures={out['fail']}, converged={ok2},{ok3}")
    for t in res: print(f"  {t[0]:<12} b={t[1]:+.3f} SE={t[2]:.3f} OR={t[3]:.2f} [{t[4]:.2f}, {t[5]:.2f}] chi2(1)={t[6]:.2f} p={t[7]:.4f}")
    return res
inrng=lambda s,dmin,dmax,kmin,kmax: s[(s.depth.between(dmin,dmax))&(s.k.between(kmin,kmax))]
print("##### REPRODUCE PAPER (filter by depth/k range, 3.1 without the unparseable run)")
p31=r[(r.model=='gemini-3.1-flash-lite')&(~r.origin.str.contains('unparseable'))]
fit(inrng(p31,4,7,20,60),'3.1-flash-lite paper set')
fit(inrng(r[r.model=='gemini-3.5-flash-lite'],4,7,20,60),'3.5-flash-lite paper set')
fit(inrng(r[r.model=='qwen2.5:14b'],2,4,5,20),'qwen paper set')
print("\n##### REVISED (grid membership by scenario file)")
fit(r[(r.model=='gemini-3.1-flash-lite')&(r.grid=='main')],'3.1-flash-lite main grid')
fit(r[(r.model=='gemini-3.5-flash-lite')&(r.grid=='main')],'3.5-flash-lite main grid')
fit(r[(r.model=='qwen2.5:14b')&(r.grid=='low')],'qwen low grid')
print("\n##### SENSITIVITY: 3.1 excluding the unparseable run (159)")
fit(p31[p31.grid=='main'],'3.1 main 159')
print("\n##### extra: 3.5-flash-lite low grid")
fit(r[(r.model=='gemini-3.5-flash-lite')&(r.grid=='low')],'3.5 low')
