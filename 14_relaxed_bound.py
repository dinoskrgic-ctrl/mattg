"""Bound on Q_dial = C_b/b_dial over all flat-band weights phi(x) = |x| + dphi(x) with dphi >= 0 (heating adds weight).
  C_b = pi/<1/phi>_w,  w = 1/(4 cosh^2(x/2))   [Kubo: hbar/tau = 2 pi V^2 c T / <1/phi>, lambda_d = 2 V^2 c]
  b_dial = 1 + 3 I0/pi^2,  I0 = Int dphi
Proof sketch: at fixed I0, <1/phi>_w is convex in phi and the constraints are linear, so KKT gives the unique minimiser
  phi* = max(|x|, k sqrt(w)) = max(|x|, k/(2 cosh(x/2))),   k fixed by I0.
Q_dial is then a function of one number k; its maximum over k is the bound."""
import numpy as np
from scipy.optimize import minimize_scalar, brentq
x=np.linspace(-80,80,1600001); w=1/(4*np.cosh(x/2)**2)
def stats(phi):
    Cb=np.pi/np.trapezoid(w/phi,x); I0=np.trapezoid(phi-np.abs(x),x); b=1+3*I0/np.pi**2; return Cb,b,Cb/b
def phistar(k): return np.maximum(np.abs(x),k*np.sqrt(w))
res=minimize_scalar(lambda k:-stats(phistar(k))[2],bounds=(0.1,40),method='bounded',options={'xatol':1e-8})
k=res.x; Cb,b,Q=stats(phistar(k)); xc=brentq(lambda t:t-k/(2*np.cosh(t/2)),1e-6,50)
print(f'optimum k = {k:.5f}, crossing x_c = {xc:.5f}: C_b = {Cb:.4f}, b_dial = {b:.4f}, Q_max = {Q:.4f}')
print('Q along the optimal family:')
for kk in [0.5,1,2,3,k,6,10,20]: c,bb,q=stats(phistar(kk)); print(f'   k={kk:6.3f}: C_b={c:6.3f} b={bb:6.3f} Q={q:.4f}')
# random trial shapes must stay below Q_max
rng=np.random.default_rng(1); worst=0
for t in range(3000):
    A=rng.exponential(1,4)*rng.choice([0.1,1,3],4); s=rng.uniform(0.3,4,4); c=rng.normal(0,2,4)
    d=sum(a*np.exp(-((x-cc)/ss)**2) for a,ss,cc in zip(A,s,c)); d=0.5*(d+d[::-1])
    worst=max(worst,stats(np.abs(x)+d)[2])
print(f'3000 random non-negative excesses: largest Q = {worst:.4f} (must be < Q_max)')
xcoth=np.where(np.abs(x)<1e-8,2.0,x/np.tanh(x/2+1e-300)); c,bb,q=stats(xcoth)
print(f'conformal x coth(x/2): C_b={c:.4f} b={bb:.4f} Q={q:.4f} = {q/Q:.3f} of the bound')
# what if dphi may be negative somewhere (phi >= 0 only, Im Sigma_f thermal deficit)?
print('without dphi>=0: phi = max(|x|-delta, 0)+... lowers C_b; check phi=|x|*tanh-type deficits:')
for a in [0.3,1.0]:
    ph=np.sqrt(x**2+1e-12)*(1-a*np.exp(-x**2))+1e-9; print(f'   deficit a={a}: Q={stats(ph)[2]:.4f}')
