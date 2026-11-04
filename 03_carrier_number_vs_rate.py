"""Carrier number versus scattering rate (Sec. IV, elastic and inelastic parts).
R(T,D) = K(D) [R0 + W(D) F(T)] with a shared peaked profile F: how much of the field dependence can a Matthiessen
prefactor K(D) (a change of Dirac carrier number or mass) carry, and how much needs a new inelastic rate W(D)?"""
import numpy as np
from scipy.optimize import least_squares
from common import load, Ds
C,C10=load()
prof={'x/(1+x^2)':lambda T,s:(T/s)/(1+(T/s)**2),'x e^-x':lambda T,s:(T/s)*np.exp(-T/s)}
def rms(f): return 1e3*np.sqrt(np.mean(f**2))
for pn,F in prof.items():
    print(f'=== profile {pn}')
    # (c)
    def rc(q):
        R0,c,D0,k,s=q; return np.concatenate([(1+k*D**2)*(R0+c*(D**2+D0**2)*F(T,s))-R for (T,R),D in zip(C,Ds)])
    b=min((least_squares(rc,[0.5,17,0.15,k0,58],bounds=([0,0,0,-5,10],[2,500,0.6,50,300])) for k0 in (0,2,6)),key=lambda z:z.cost)
    J=b.jac; s2=np.sum(b.fun**2)/(len(b.fun)-5); err=np.sqrt(np.diag(np.linalg.inv(J.T@J))*s2)
    R0,c,D0,k,s=b.x
    print(f' (c) K=1+kD^2, common R0, W=c(D^2+D0^2): rms {rms(b.fun):.1f} Ohm; R0={R0:.3f}+-{err[0]:.3f} kOhm c={c:.1f}+-{err[1]:.1f} D0={D0:.3f}+-{err[2]:.3f} k={k:.2f}+-{err[3]:.2f} (V/nm)^-2 scale={s:.1f} K')
    print(f'     K(0.38)/K(0.14) = {(1+k*0.38**2)/(1+k*0.14**2):.2f}; inelastic rate ratio (0.38/0.14) = {(0.38**2+D0**2)/(0.14**2+D0**2):.2f}')
    T10,R10=C10; p=(1+k*0.01)*(R0+c*(0.01+D0**2)*F(T10,s)); print(f'     D=0.10 predicted with no new parameter: rms {rms(p-R10):.0f} Ohm, mean offset {1e3*np.mean(p-R10):+.0f} Ohm')
    # (c0) same with k=0
    def rc0(q): return rc([q[0],q[1],q[2],0.,q[3]])
    b0=least_squares(rc0,[0.5,17,0.15,58],bounds=([0,0,0,10],[2,500,0.6,300]))
    print(f' (c0) same with k=0 (pure admixture, 4 params): rms {rms(b0.fun):.1f} Ohm; D0={b0.x[2]:.3f}')
    # (d) K per curve
    def rd(q):
        R0,c,D0,s=q[:4]; K=np.r_[1.,q[4:]]  # K(0.14)=1 normalisation
        return np.concatenate([K[i]*(R0+c*(D**2+D0**2)*F(T,s))-R for i,((T,R),D) in enumerate(zip(C,Ds))])
    bd=least_squares(rd,np.r_[0.5,17,0.15,58,np.ones(6)],bounds=(np.r_[0,0,0,10,0.3*np.ones(6)],np.r_[2,500,0.6,300,3*np.ones(6)]))
    print(f' (d) K free per curve (10 params): rms {rms(bd.fun):.1f} Ohm; K(D)/K(0.14) = {np.round(np.r_[1,bd.x[4:]],3)}; D0={bd.x[2]:.3f}')
