"""Full thermal average fitted to the data (Sec. IV, The full thermal average).
Exact Eq. for rho (common.rho) on the five curves at D = 0.20-0.38: shared R_el, amplitude c (D^2 + D0^2), kappa, Delta.
Single level versus symmetric +-Delta, with and without the marginal real part. Data decimated x3 for speed.
Then predicts the temperature-dependent part at D = 0.14 and 0.10 and reports the D^4 share and T_max."""
import numpy as np, common
from scipy.optimize import least_squares
common.X=np.linspace(-22,22,441); common.MF=1/(4*np.cosh(common.X/2)**2)
from common import load, Ds, rho
C,C10=load(); idx=range(1,7); CC=[(C[i][0][::3],C[i][1][::3]) for i in idx]; DD=Ds[list(idx)]
def model(q,T,D,kw): return rho(T,q[0],q[1]*(D**2+q[2]**2),q[3],q[4],**kw)
def res(q,kw): return np.concatenate([model(q,T,D,kw)-R for (T,R),D in zip(CC,DD)])
def rms(f): return 1e3*np.sqrt(np.mean(f**2))
for kw in ({'level':'single','real':True},{'level':'sym','real':True},{'level':'single','real':False},{'level':'sym','real':False}):
    best=None
    for a0 in (0.5,1.5,4.0):
        for dd in (1.0,1.5):
            b=least_squares(res,[0.63,16,0.18,a0,dd*a0*58],args=(kw,),bounds=([0,0.01,0,0.05,1],[2,1000,0.6,30,3000]),x_scale='jac',max_nfev=300)
            if best is None or b.cost<best.cost: best=b
    q=best.x; J=best.jac; s2=np.sum(best.fun**2)/(len(best.fun)-5)
    err=np.sqrt(np.abs(np.diag(np.linalg.pinv(J.T@J)))*s2)
    Tm=np.linspace(20,140,121); tmax=Tm[np.argmax(model(q,Tm,0.30,kw))]
    Tg=np.linspace(12,32,21); sl=np.array([np.polyfit(Tg,model(q,Tg,D,kw),1)[0]*1e3 for D in Ds]); x2=Ds**2
    cq=np.linalg.lstsq(np.c_[np.ones(7),x2,x2**2],sl,rcond=None)[0]; quart=cq[2]*0.38**4/np.dot(cq,[1,0.38**2,0.38**4])
    print(f"{kw}: rms {rms(best.fun):.1f} Ohm | R_el={q[0]:.3f} c={q[1]:.2f} D0={q[2]:.3f}+-{err[2]:.3f} kappa={q[3]:.2f}+-{err[3]:.2f} Delta={q[4]:.0f}+-{err[4]:.0f} K "
          f"| T_max={tmax:.0f} K | quartic share of model slope at 0.38: {100*quart:+.1f}%")
    for (T,R),D in ((C[0],0.14),(C10,0.10)):
        p=model(q,T,D,kw); dR=lambda y:np.interp(57,T,y)-np.interp(12.5,T,y)
        print(f"     D={D:.2f}: T-dependent part data/model = {dR(R)/dR(p):.2f}; offset data-model = {1e3*np.mean(R-p):+.0f} Ohm")
