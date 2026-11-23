import json, numpy as np, tellurium as te
from scipy.optimize import least_squares
d=json.load(open('workspace/investigations/biogears-sbml-sepsis/data/real_trajectories.json'))
S=d['exact_sev3']; tref=np.array(S['t_h'])*3600.0
ref={'MAP':np.array(S['obs']['MAP']),'SVR':np.array(S['obs']['SVR']),
     'PB':np.array(S['obs']['blood_pathogen']),'TI':np.array(S['obs']['tissue_integrity'])}
ANT="""model ef
 PB=seed; Mi=0; TI=1; seed=1.0;
 kg=2.4e-5; Pcap=110; kk=1e-6; kM=3e-4; Kp=30; dM=1e-4; Ksvr=0.3; Kti=0.3; vmax=0.3; kdmg=1e-3; krep=1e-4;
 PB' = kg*PB*(1-PB/Pcap) - kk*Mi*PB;
 Mi' = kM*PB/(Kp+PB) - dM*Mi;
 SVR := 0.929*(1 - vmax*Mi/(Ksvr+Mi));
 MAP := 102.8*SVR;
 TI' = -kdmg*(Mi/(Kti+Mi))*TI + krep*(1-TI);
end"""
r=te.loadAntimonyModel(ANT)
PN=['kg','Pcap','kk','kM','Kp','dM','Ksvr','Kti','vmax','kdmg','krep','seed']
p0=np.array([2.4e-5,110,1e-6,3e-4,30,1e-4,0.3,0.3,0.3,1e-3,1e-4,1.0])
lo=np.array([1e-5,50,1e-9,1e-5,1,1e-6,0.02,0.02,0.22,1e-4,1e-5,0.1])
hi=np.array([5e-4,150,1e-3,5e-2,200,5e-3,5,5,0.45,1e-2,1e-3,5])
sel=['time','MAP','SVR','PB','TI']
def sim(p):
    r.resetToOrigin()
    for n,v in zip(PN,p): r.setValue(n,float(v))
    r.reset(); return np.array(r.simulate(0,tref[-1],len(tref),sel))
def resid(logp):
    try: A=sim(np.exp(logp))
    except Exception: return np.ones(4*len(tref))*5
    out=[]
    for i,k in enumerate(['MAP','SVR','PB','TI'],1):
        sc=max(abs(ref[k]).max(),1e-6); wgt=(2.0 if k in('MAP','SVR') else 1.0)*(0.5+1.5*tref/tref[-1])
        out.append(wgt*(A[:,i]-ref[k])/sc)
    return np.concatenate(out)
sol=least_squares(resid,np.log(p0),bounds=(np.log(lo),np.log(hi)),method='trf',max_nfev=1000)
p=np.exp(sol.x); A=sim(p)
print("fit cost=%.4f"%sol.cost)
for i,k in enumerate(['MAP','SVR','PB','TI'],1):
    rmse=np.sqrt(np.mean((A[:,i]-ref[k])**2)); rel=100*rmse/max(abs(ref[k]).max(),1e-6)
    print("  %-4s end %7.3f (ref %7.3f)  RMSE %.3g (%.1f%%)"%(k,A[-1,i],ref[k][-1],rmse,rel))
json.dump({'ant':ANT,'params':{n:float(v) for n,v in zip(PN,p)}},open('/private/tmp/claude-502/-Users-eranagmon-code/04c4b2b1-5bb0-4864-b0d0-29ce90625439/scratchpad/ef_params.json','w'))
