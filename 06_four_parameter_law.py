"""Four-parameter law and the elastic/inelastic split (Sec. IV, Figs. 2 and 3b).
R = R_el + W2 (D^2 + D0^2) x/(1+x^2), x = T/T*, fitted to the five curves at D = 0.20-0.38; extrapolated to 0.14 and 0.10
with a free offset and amplitude factor; per-curve elastic offsets with the law fixed. Writes fit4.npy and offsets.npy."""
import numpy as np
from scipy.optimize import least_squares
from common import load, Ds
C,C10=load()
F=lambda T,s:(T/s)/(1+(T/s)**2)
idx=range(1,7); CC=[C[i] for i in idx]; DD=Ds[1:]
def rc(q): return np.concatenate([q[0]+q[1]*(D**2+q[2]**2)*F(T,q[3])-R for (T,R),D in zip(CC,DD)])
b=least_squares(rc,[0.6,17,0.15,58]); q=b.x
J=b.jac; s2=np.sum(b.fun**2)/(len(b.fun)-4); err=np.sqrt(np.diag(np.linalg.inv(J.T@J))*s2)
print(f'4-param fit D=0.20-0.38: R_el={q[0]:.3f}, W2={q[1]:.2f} kOhm/(V/nm)^2, D0={q[2]:.3f}+-{err[2]:.3f}, T*={q[3]:.1f}, rms={1e3*np.sqrt(np.mean(b.fun**2)):.1f} Ohm')
np.save('fit4.npy',q)
for (T,R),D in ((C[0],0.14),(C10,0.10)):
    f=F(T,q[3]); W=q[1]*(D**2+q[2]**2)
    # free offset AND free amplitude factor kappa multiplying the predicted T-dependent part
    X=np.c_[np.ones_like(f),W*f]; c=np.linalg.lstsq(X,R,rcond=None)[0]
    print(f'  D={D:.2f}: best amplitude factor (data/extrapolation) = {c[1]:.2f}; elastic offset = {c[0]:.3f} kOhm (vs {q[0]:.3f}); rms {1e3*np.sqrt(np.mean((R-X@c)**2)):.0f} Ohm')
# per-curve offsets with W fixed by the 4-param law (all 8 curves)
offs=[]
for (T,R),D in list(zip(C,Ds))+[(C10,0.10)]:
    offs.append(np.mean(R-q[1]*(D**2+q[2]**2)*F(T,q[3])))
print('elastic offsets with the law fixed (D=0.14..0.38, 0.10):',np.round(offs,3))
np.save('offsets.npy',np.array(offs))
