"""Robustness of the four-parameter law to the choice of curves (Sec. IV).
Free offsets, a common offset and the admixture law R_el + c (D^2 + D0^2) F(T/T*) on all seven curves at D >= 0.14,
on the five curves at D >= 0.20 (reliable low-T traces) and on 0.20-0.34; each extrapolated to D = 0.14 and 0.10."""
import numpy as np
from scipy.optimize import least_squares
from common import load, Ds
C,C10=load()
F=lambda T,s:(T/s)/(1+(T/s)**2)
def rms(f): return 1e3*np.sqrt(np.mean(f**2))
for name,idx in (('all seven (0.14-0.38)',range(7)),('five (0.20-0.38)',range(1,7)),('0.20-0.34',range(1,6))):
    CC=[C[i] for i in idx]; DD=Ds[list(idx)]
    def ra(q):  # free offset+amp
        out=[]
        for T,R in CC:
            f=F(T,q[0]); X=np.c_[np.ones_like(f),f]; c=np.linalg.lstsq(X,R,rcond=None)[0]; out.append(R-X@c)
        return np.concatenate(out)
    ba=least_squares(ra,[58],bounds=([10],[300]))
    def rb(q):  # common offset, free amp
        out=[]
        for T,R in CC:
            f=F(T,q[0]); W=np.dot(f,R-q[1])/np.dot(f,f); out.append(R-q[1]-W*f)
        return np.concatenate(out)
    bb=least_squares(rb,[58,0.6],bounds=([10,0],[300,2]))
    def rc(q):  # common offset, W = c(D^2+D0^2): 4 params
        return np.concatenate([q[0]+q[1]*(D**2+q[2]**2)*F(T,q[3])-R for (T,R),D in zip(CC,DD)])
    bc=least_squares(rc,[0.6,17,0.15,58],bounds=([0,0,0,10],[2,500,0.6,300]))
    J=bc.jac; s2=np.sum(bc.fun**2)/(len(bc.fun)-4); err=np.sqrt(np.diag(np.linalg.inv(J.T@J))*s2)
    print(f'{name}: free offsets {rms(ba.fun):.1f} Ohm (T*={ba.x[0]:.1f}); common offset {rms(bb.fun):.1f} Ohm (R_el={bb.x[1]:.3f}); '
          f'4-param admixture {rms(bc.fun):.1f} Ohm: R_el={bc.x[0]:.3f}+-{err[0]:.3f}, c={bc.x[1]:.1f}+-{err[1]:.1f}, D0={bc.x[2]:.3f}+-{err[2]:.3f}, T*={bc.x[3]:.1f}+-{err[3]:.1f}')
    q=bc.x
    for (T,R),D in ((C[0],0.14),(C10,0.10)):
        p=q[0]+q[1]*(D**2+q[2]**2)*F(T,q[3]); print(f'     predicted D={D:.2f} curve (not fitted): rms {rms(p-R):.0f} Ohm, mean offset {1e3*np.mean(p-R):+.0f} Ohm, ratio of T-dependent parts (data/model, 12->57 K): '
          f'{(np.interp(57,T,R)-np.interp(12.5,T,R))/(np.interp(57,T,p)-np.interp(12.5,T,p)):.2f}')
