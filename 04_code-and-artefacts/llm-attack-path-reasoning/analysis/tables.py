import sys, os, json, math, pandas as pd, numpy as np
art=sys.argv[1] if len(sys.argv)>1 else "."  # folder that holds engine.py
def wil(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return f"{k}/{n}, {100*p+1e-9:.1f}% [{100*max(0,c-h):.0f}–{100*min(1,c+h):.0f}]"
r=pd.read_csv('analysis/runs_of_record.csv')
s=r[r.scored]
s1=s[s.system!='C2']
print("=== TABLE 6: scored runs (+ unscored attempts)")
for (sy,mo),g in r.groupby(['system','model']):
    sc=g[g.scored]; un=g[~g.scored]
    grids=sc.groupby('grid').size().to_dict()
    print(f"{sy} {mo:<26} scored={len(sc):5d} grids={grids} unscored={len(un)}  all:{wil(int(sc.success.sum()),len(sc))}",
          " | main:"+wil(int(sc[sc.grid=='main'].success.sum()),len(sc[sc.grid=='main'])) if (sc.grid=='main').any() else '',
          " | low:"+wil(int(sc[sc.grid=='low'].success.sum()),len(sc[sc.grid=='low'])) if (sc.grid=='low').any() else '')
print('TOTAL scored', s.shape[0], 'unscored', (~r.scored).sum())
print("\n=== TABLE 7/8: main grid by depth and by k (pure 160)")
for mo in ['gemini-3.1-flash-lite','gemini-3.5-flash-lite','random']:
    g=s1[(s1.model==mo)&(s1.grid=='main')]
    print(mo,'by depth:',{d:wil(int(x.success.sum()),len(x)) for d,x in g.groupby('depth')})
    print(mo,'by k    :',{k:wil(int(x.success.sum()),len(x)) for k,x in g.groupby('k')})
print("\n=== TABLE 9: low grid (pure 72)")
for mo in ['qwen2.5:14b','gemini-3.5-flash-lite','random']:
    g=s1[(s1.model==mo)&(s1.grid=='low')]
    print(mo,'by k:',{k:wil(int(x.success.sum()),len(x)) for k,x in g.groupby('k')})
    print(mo,'by d:',{d:wil(int(x.success.sum()),len(x)) for d,x in g.groupby('depth')})
q=s[(s.model=='qwen2.5:14b')&(s.grid=='main')]
print('qwen main partial cells:', q.groupby(['depth','k']).success.agg(['sum','size']).to_dict('index'))
print("\n=== C2 by k (pooled both models)")
c2=s[s.system=='C2']; print(c2.groupby('k').success.agg(['sum','size']).to_dict('index'))
print(c2.groupby(['model','k']).success.agg(['sum','size']).to_dict('index'))
print("\n=== 3.5-flash vs 3.5-flash-lite on common instances")
a=s[(s.model=='gemini-3.5-flash')].set_index('scenario'); b=s1[(s1.model=='gemini-3.5-flash-lite')].set_index('scenario')
cm=a.index.intersection(b.index); print(len(cm), a.loc[cm].success.sum(), b.loc[cm].success.sum(), a.loc[cm].groupby('k').size().to_dict(), a.loc[cm].depth.unique())
