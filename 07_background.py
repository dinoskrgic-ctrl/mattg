"""Field-independent background and the value of D0 (Sec. IV, Background).
All eight curves, 12-78 K: R = R_el(D) + W2 (D^2 + D0^2) F(T/T*) + b (T - 12 K), R_el free per curve, b = 0 versus free;
also restricted to D >= 0.20. Writes fit_bg.npy and offsets_bg.npy."""
import numpy as np
from scipy.optimize import least_squares
from common import load, Ds
C,C10=load(); allC=[C10]+C; DD=np.r_[0.10,Ds]
F=lambda T,s:(T/s)/(1+(T/s)**2)
def res(q,bfree,sel):
    W2,D0,s=q[:3]; b=q[3] if bfree else 0.; out=[]
    for i in sel:
        T,R=allC[i]; m=W2*(DD[i]**2+D0**2)*F(T,s)+b*(T-12); out.append(R-m-np.mean(R-m))
    return np.concatenate(out)
for name,sel in (('all eight',range(8)),('D>=0.20',range(2,8))):
  for bfree in (False,True):
    bb=least_squares(res,[16,0.16,58]+([0.] if bfree else []),args=(bfree,sel))
    J=bb.jac; s2=np.sum(bb.fun**2)/(len(bb.fun)-len(bb.x)-len(sel)); err=np.sqrt(np.diag(np.linalg.inv(J.T@J))*s2)
    per=[]; k=0
    for i in sel:
        n=len(allC[i][0]); per.append(1e3*np.sqrt(np.mean(bb.fun[k:k+n]**2))); k+=n
    print(f"{name:9s} b {'free' if bfree else '=0  '}: rms {1e3*np.sqrt(np.mean(bb.fun**2)):.1f} Ohm; W2={bb.x[0]:.2f} D0={bb.x[1]:.3f}+-{err[1]:.3f} T*={bb.x[2]:.1f}"
          +(f" b={1e3*bb.x[3]:.2f}+-{1e3*err[3]:.2f} Ohm/K" if bfree else '')+'; per-curve rms '+' '.join(f'{x:.0f}' for x in per))
# save the all-eight, b-free model and its offsets for the figures; also test: elastic offsets under this model
bb=least_squares(res,[16,0.16,58,0.],args=(True,range(8))); W2,D0,s,b=bb.x
offs=[]
for i in range(8):
    T,R=allC[i]; m=W2*(DD[i]**2+D0**2)*F(T,s)+b*(T-12); offs.append(np.mean(R-m))
offs=np.array(offs); print('offsets R_el(D) with background, D=0.10..0.38:',np.round(offs,3))
np.save('fit_bg.npy',np.r_[W2,D0,s,b]); np.save('offsets_bg.npy',offs)
# slope test of the background model: 12-32 K slopes implied
Tg=np.linspace(12,32,41); sl=[np.polyfit(Tg,W2*(D**2+D0**2)*F(Tg,s)+b*(Tg-12),1)[0]*1e3 for D in DD]
print('implied 12-32 K slopes:',np.round(sl,2),'(measured 5.48 7.26 12.07 15.35 18.16 21.52 27.53 32.83)')
