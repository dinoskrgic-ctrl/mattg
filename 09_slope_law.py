"""Slope law and every number derived from it (Sec. IV and Outlook). Run 08_slope_errors.py first.
a(1 + D^2/D0^2), free exponent, quartic share, linear law, the D = 0.10 V/nm test, the miss at 0.14 V/nm for D0 = 0.10 and
0.18, the self-doping law a/(1 - D^2/Dc^2)^2, the bound on Dc, and a built-in mirror offset a + c (D - D_off)^2.
Slopes as quoted in the paper; 08_slope_errors.py recomputes them from the data to within 0.05 Ohm/K."""
import numpy as np
from scipy.optimize import curve_fit
Ds=np.array([0.14,0.20,0.24,0.27,0.30,0.34,0.38]); S=np.array([7.26,12.07,15.35,18.16,21.52,27.53,32.83])
sig=np.load('slope_sigma.npy')     # 08_slope_errors.py: 2xSE and window spread in quadrature
cs=lambda f,p: np.sum(((S-f(Ds,*p))/sig)**2)
f2=lambda D,a,D0:a*(1+D**2/D0**2); p,cv=curve_fit(f2,Ds,S,[3.6,0.13],sigma=sig,absolute_sigma=True)
print(f"slope law: a = {p[0]:.2f} +- {np.sqrt(cv[0,0]):.2f}, D0 = {p[1]:.3f} +- {np.sqrt(cv[1,1]):.3f}, chi2 = {cs(f2,p):.1f}/5; predicted D=0.10: {f2(0.10,*p):.2f} (measured 5.48 +- 0.6); D=0.30: {f2(0.30,*p):.1f}")
fp=lambda D,a,c,q:a+c*D**q; q,cq=curve_fit(fp,Ds,S,[3.6,200,2],sigma=sig,absolute_sigma=True); print(f"free exponent p = {q[2]:.2f} +- {np.sqrt(cq[2,2]):.2f}")
f4=lambda D,a,c,e:a+c*D**2+e*D**4; q,cq=curve_fit(f4,Ds,S,[3.6,200,0],sigma=sig,absolute_sigma=True)
print(f"quartic share at 0.38 = {100*q[2]*0.38**4/f4(0.38,*q):+.1f} +- {100*np.sqrt(cq[2,2])*0.38**4/f4(0.38,*q):.1f} %")
fl=lambda D,a,c:a+c*D; q,_=curve_fit(fl,Ds,S,sigma=sig,absolute_sigma=True); print(f"linear law: a = {q[0]:.1f}, chi2 = {cs(fl,q):.0f}, predicted D=0.10: {fl(0.10,*q):.2f}")
for D0 in (0.10,0.18):
    pr=S[4]*(0.14**2+D0**2)/(0.30**2+D0**2); e=S[4]*0+sig[4]*(0.14**2+D0**2)/(0.30**2+D0**2)
    print(f"D0 = {D0}: slope(0.30) held fixed predicts {pr:.2f} +- {e:.2f} at 0.14 vs 7.26 +- {sig[0]}: {abs(pr-7.26)/np.hypot(e,sig[0]):.1f} sigma (measurement error only: {abs(pr-7.26)/sig[0]:.1f})")
fs=lambda D,a,Dc:a/(1-D**2/Dc**2)**2; q,_=curve_fit(fs,Ds,S,[8,0.8],sigma=sig,absolute_sigma=True,maxfev=20000); print(f"self-doping a/(1-D^2/Dc^2)^2: best chi2 = {cs(fs,q):.0f} (Dc = {q[1]:.2f})")
# lower bound on Dc from the quartic bound: a + c D^2 (1 + 2 D^2/Dc^2) expansion -> quartic share 2D^2/Dc^2/(...)
f4b=lambda D,a,c,Dc:a+c*D**2*(1+2*D**2/Dc**2)
dcs=np.linspace(0.4,6,2000); ch=[]
for Dc in dcs:
    X=np.c_[np.ones_like(Ds),Ds**2*(1+2*Ds**2/Dc**2)]; W=1/sig**2; c=np.linalg.solve(X.T@(X*W[:,None]),X.T@(S*W)); ch.append(np.sum(((X@c-S)/sig)**2))
ch=np.array(ch); X=np.c_[np.ones_like(Ds),Ds**2]; W=1/sig**2; c=np.linalg.solve(X.T@(X*W[:,None]),X.T@(S*W)); ch0=np.sum(((X@c-S)/sig)**2)
print(f"quartic from emptying pocket, D^2(1+2D^2/Dc^2): Dc >= {dcs[ch<=ch0+4].min():.2f} V/nm at 2 sigma")
d0s=np.linspace(-0.3,0.15,451); ch=[]
for d0 in d0s:
    X=np.c_[np.ones_like(Ds),(Ds-d0)**2]; c=np.linalg.solve(X.T@(X*W[:,None]),X.T@(S*W)); ch.append(np.sum(((X@c-S)/sig)**2))
ch=np.array(ch); ok=d0s[ch<=ch.min()+4]; print(f"a + c(D-Doff)^2: best Doff = {d0s[ch.argmin()]:+.3f}, 2 sigma range {ok.min():+.3f} .. {ok.max():+.3f} V/nm")
