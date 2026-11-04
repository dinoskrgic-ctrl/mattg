"""(1) A high-energy offset phi -> |x| + c' (T-linear floor of Im Sigma_f) adds no logarithm: b_dial = 1 + 3 I0/pi^2 with I0 measured from |x| + c'.
(2) Convexity: phi'(x) = (chi * w)(x), so a monotone bath spectrum gives a convex phi >= |x| + c'.
(3) Ceiling Q_max(c') from phi* = max(|x| + c', k sqrt(w)); c' = Int_0^inf [chi(y) coth(y/2) - 1] dy  (conformal: c' = 0)."""
import numpy as np
from scipy.special import digamma
from scipy.optimize import minimize_scalar
def L(e,T): return np.log(1e4/(2*np.pi*T)) - np.real(digamma(0.5+1j*e/(2*np.pi*T)))
def bnum(phi,E=60,n=60001):
    Ts=np.exp(np.linspace(np.log(0.05),np.log(0.6),24)); e=np.linspace(-E,E,n)
    O=np.array([-np.trapezoid(T*phi(e/T)*L(e,T),e) for T in Ts]); S=-np.gradient(O,Ts); CT=np.gradient(S,Ts)
    k=slice(3,-3); return -np.polyfit(np.log(Ts[k]),CT[k],1)[0]/(2*np.pi**2/3)
for c in [0.0,1.0,-0.5]:
    for a in [0.0,1.0]:
        print(f"c'={c:+.1f} a={a}: numeric b = {bnum(lambda x:np.abs(x)+c+a*np.exp(-np.abs(x))):.3f}   formula {1+6*a/np.pi**2:.3f}")
x=np.linspace(-80,80,800001); w=1/(4*np.cosh(x/2)**2)
def Q(phi):
    Cb=np.pi/np.trapezoid(w/phi,x); b=1+3*np.trapezoid(phi-np.abs(x),x)/np.pi**2; return Cb,b,Cb/b
print("ceiling vs offset c' (I0 counted from |x|+c'):")
for c in [-0.5,0.0,0.5,1.0,2.0]:
    def q(k): 
        ph=np.maximum(np.abs(x)+c,k*np.sqrt(w)); Cb=np.pi/np.trapezoid(w/ph,x); b=1+3*np.trapezoid(ph-np.abs(x)-c,x)/np.pi**2; return Cb/b
    r=minimize_scalar(lambda k:-q(k),bounds=(0.01,40),method='bounded'); print(f"   c'={c:+.1f}: Q_max = {q(r.x):.3f} at k = {r.x:.3f}")
