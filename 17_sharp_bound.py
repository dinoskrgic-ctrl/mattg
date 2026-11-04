"""Sharp ceiling on Q_dial over REALIZABLE flat bands: any odd, non-decreasing bath spectrum chi(y), chi(inf)=1, is a
mixture of steps chi_s = sgn(y) theta(|y|-s) with weights p(s) >= 0, Int p = 1. phi is linear in p.
Step s: phi_s'(x) = 1 - n_F(x-s) - n_F(x+s), phi_s(0) = 2 ln coth(s/2), floor c'_s = -2 ln(1-e^-s) - s.
Constraint: no high-energy floor, c' = Int p c'_s = 0. 40 random starts (~10 min). Conformal: p = sech^2(s/2)/2.
Q = pi / (<1/phi>_w (1 + 3 I0/pi^2)), I0 = Int (phi - |x| - c')."""
import numpy as np, warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)   # harmless exp overflow in the softmax start points
from scipy.optimize import minimize
x=np.linspace(0,60,6001); dx=x[1]-x[0]; w=1/(4*np.cosh(x/2)**2)     # even functions: integrate x>=0, double
nF=lambda z:0.5*(1-np.tanh(z/2))
s=np.concatenate([np.linspace(0.02,1,40),np.linspace(1.05,12,90)])
PH=[];CP=[];J=[]
for si in s:
    dphi=1-nF(x-si)-nF(x+si); ph=2*np.log(1/np.tanh(si/2))+np.concatenate([[0],np.cumsum(0.5*(dphi[1:]+dphi[:-1])*dx)])
    c=-2*np.log(1-np.exp(-si))-si; PH.append(ph); CP.append(c); J.append(2*np.trapezoid(ph-x-c,x))
PH=np.array(PH); CP=np.array(CP); J=np.array(J)
print('check step asymptote: phi_s(60)-60-c_s max |.| =',np.abs(PH[:,-1]-x[-1]-CP).max())
def Q(p):
    ph=p@PH; inv=2*np.trapezoid(w/ph,x); I0=p@J; return np.pi/(inv*(1+3*I0/np.pi**2))
# conformal check
pc=0.5/np.cosh(s/2)**2; ds=np.gradient(s); pc=pc*ds; pc/=pc.sum()
print(f"conformal (discretized): c'={pc@CP:+.4f}  Q={Q(pc):.4f}  (exact 2.4566)")
def run(eq):
    best=0;bp=None;rng=np.random.default_rng(0)
    for trial in range(40):
        z0=rng.normal(0,1,len(s))
        f=lambda z:-Q(np.exp(z)/np.exp(z).sum())
        cons=[{'type':'eq' if eq else 'ineq','fun':(lambda z:(-1 if not eq else 1)*(np.exp(z)/np.exp(z).sum())@CP)}]
        r=minimize(f,z0,constraints=cons,method='SLSQP',options={'maxiter':400,'ftol':1e-10})
        p=np.exp(r.x)/np.exp(r.x).sum()
        if abs(p@CP)<1e-4 or (not eq and p@CP<=1e-4):
            if -r.fun>best: best=-r.fun; bp=p
    return best,bp
for eq in (True,):
    q,p=run(eq); I0=p@J; ph=p@PH
    print(f"{'c_prime=0' if eq else 'c_prime<=0'}: sharp max Q = {q:.4f}, b_dial = {1+3*I0/np.pi**2:.4f}, C_b = {q*(1+3*I0/np.pi**2):.3f}; weight centred at s = {p@s:.2f} (spread {np.sqrt(p@s**2-(p@s)**2):.2f}); phi(0) = {ph[0]:.3f}")
    np.save('best_p_eq.npy' if eq else 'best_p_ineq.npy',p)
