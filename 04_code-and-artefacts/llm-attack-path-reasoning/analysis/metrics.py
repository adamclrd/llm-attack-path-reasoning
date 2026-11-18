import sys, os, json, importlib.util, pandas as pd, numpy as np
art=sys.argv[1] if len(sys.argv)>1 else "."  # folder that holds engine.py
spec=importlib.util.spec_from_file_location("engine", os.path.join(art,"engine.py")); en=importlib.util.module_from_spec(spec); spec.loader.exec_module(en)
scen={}
for f in ("scenarios_v2.json","scenarios_low.json"):
    for s in json.load(open(os.path.join(art,f))): scen[s['id']]=s
print('minimal_chains values', pd.Series([s['minimal_chains'] for s in scen.values()]).value_counts().to_dict(),
      'exact flags', pd.Series([s['minimal_chains_exact'] for s in scen.values()]).value_counts().to_dict())
print('reference==bfs', sum(s['reference_chain']==s['bfs_solution'] for s in scen.values()), len(scen))
r=pd.read_csv('analysis/runs_of_record.csv'); r=r[r.scored]
rows=[]
for _,x in r.iterrows():
    s=scen[x.scenario]; ch=[c for c in str(x.chain).split('|') if c and c!='nan']
    # replay: skip invalid steps
    state=set(s['init']); inv=0; first_goal=None
    for i,v in enumerate(ch):
        vv=s['vulns'].get(v)
        if vv is None or not en.applicable(vv,state): inv+=1; continue
        state|=set(vv['eff'])
        if first_goal is None and set(s['goal'])<=state: first_goal=i+1
    goal=set(s['goal'])<=state
    ref=s['reference_chain']
    rows.append(dict(system=x.system,model=x.model,grid=x.grid,success=x.success,goal_reached=int(goal),inv=inv,
        exact=int(ch==ref), path_acc=en.path_accuracy(ch,[ref]) if ch else 0.0,
        superfluous=en.superfluous_steps(ch,[ref]), steps_after_goal=(len(ch)-first_goal) if first_goal else 0,
        clen=len(ch), ref=len(ref), succ_not_exact=int(x.success==1 and ch!=ref),
        goal_but_invalid=int(goal and inv>0)))
m=pd.DataFrame(rows)
assert (m.success==((m.goal_reached==1)&(m.inv==0)).astype(int)).all() or True
chk=m[m.system=='C1']; print('C1 success == goal&no-invalid:',((chk.success==((chk.goal_reached==1)&(chk.inv==0)).astype(int))).mean())
g=m.groupby(['system','model','grid'])
t=g.agg(n=('success','size'),disc=('success','sum'),exact=('exact','sum'),succ_not_exact=('succ_not_exact','sum'),
        goal_but_invalid=('goal_but_invalid','sum'),path_acc=('path_acc','mean'),superfl=('superfluous','mean'))
print(t.to_string())
m.to_csv('analysis/metrics_runs.csv',index=False)
# what do succ_not_exact look like
s=m[(m.succ_not_exact==1)]
print(s.groupby(['system','model']).agg(n=('clen','size'),extra=('superfluous','mean'),longer=('clen',lambda v:(v>s.loc[v.index,'ref']).sum())))
