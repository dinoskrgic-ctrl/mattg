"""Rank-one test, position of the maximum, width exponent (Sec. IV).
(1) SVD of the temperature-dependent parts over 14-55, 14-70 and 14-78 K. (2) Per-curve maximum from a local quadratic.
(3) Exponent n of a width growing as T^n, across fit windows and with or without a shared b T background."""
import sys; sys.path.insert(0,'.')
import numpy as np
from scipy.optimize import least_squares
from common import load, Ds
C,C10=load(12,78)
allc=list(zip(Ds,C))
def interp_matrix(lo,hi):
    Tg=np.linspace(lo,hi,30); M=np.array([np.interp(Tg,T,R)-np.interp(12.5,T,R) for D,(T,R) in allc]); return M
for lo,hi in ((14,55),(14,70),(14,78)):
    s=np.linalg.svd(interp_matrix(lo,hi),compute_uv=False); print(f"{lo}-{hi} K: s2/s1 = {s[1]/s[0]*100:.2f}%, rank-one share = {100*s[0]**2/np.sum(s**2):.3f}%")
for D,(T,R) in allc:
    k=(T>40)&(T<76); p=np.polyfit(T[k],R[k],2); print(f"D={D:.2f}: T_max (quadratic 40-76 K) = {-p[1]/(2*p[0]):.1f} K")
# (ii) exponent n: Gamma_f ~ T^n -> F = y/(1+y^2), y = (T/T*)^n ; shared n, T*, free offset+amp per curve
def fit(lo,hi,bg):
    data=[(T[(T>=lo)&(T<=hi)],R[(T>=lo)&(T<=hi)]) for D,(T,R) in allc]
    def res(p):
        n,Ts=p[0],p[1]; b=p[2] if bg else 0; out=[]
        for i,(T,R) in enumerate(data):
            y=(T/Ts)**n; o,A=p[3+2*i],p[4+2*i]; out.append(R-(o+A*y/(1+y**2)+b*T*1e-3))
        return np.concatenate(out)
    p0=[1,58,0.3]+sum([[0.6,1.0] for _ in data],[])
    r=least_squares(res,p0); return r.x[0],r.x[1]
for lo,hi in ((12,78),(12,60),(15,78),(20,78)):
    for bg in (False,True):
        n,Ts=fit(lo,hi,bg); print(f"window {lo}-{hi} K, background {'on ' if bg else 'off'}: n = {n:.3f}, T* = {Ts:.1f} K")
