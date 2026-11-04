"""When is the thermal excess dphi = phi - |x| of the flat-band Im Sigma_f non-negative?
phi(x) = Int dy chi(y) [coth(y/2) - tanh((y-x)/2)]/2 for a bath with scaling spectrum chi(y=w/T), chi(+inf)=1.
Monotone baths chi = tanh(y/2s)^(1) for several s, a sharper erf, and a non-monotone (peaked) bath for contrast."""
import numpy as np
from scipy.special import erf
y=np.linspace(-300,300,300001)
def phi_of(chi,xs):
    cth=np.where(np.abs(y)<1e-9,0.0,chi/np.tanh(y/2+1e-300))
    # the coth term needs chi(0)=0 to be finite; all baths here are odd
    return np.array([np.trapezoid((cth-chi*np.tanh((y-x)/2))/2,y) for x in xs])
xs=np.linspace(0,25,126)
baths={'tanh(y/2) conformal':np.tanh(y/2),'tanh(y/0.6)':np.tanh(y/0.6),'tanh(y/8)':np.tanh(y/8),'tanh(y/20)':np.tanh(y/20),
       'erf(y/3)':erf(y/3),'peaked: tanh(y/2)(1+2 y^2 e^{-y^2/8}/8)':np.tanh(y/2)*(1+2*(y**2/8)*np.exp(-y**2/8)),
       'dip: tanh(y/2)(1-0.8 e^{-(|y|-4)^2})':np.tanh(y/2)*(1-0.8*np.exp(-(np.abs(y)-4)**2))}
for n,c in baths.items():
    d=phi_of(c,xs)-xs; print(f'{n:42s} min dphi = {d.min():+.4f} at x={xs[d.argmin()]:.1f}; dphi(0) = {d[0]:.3f}; dphi(25) = {d[-1]:+.2e}')
