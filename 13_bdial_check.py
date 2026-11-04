"""Verify b_dial. Exact O(V^2) result for A_f = c T phi(e/T) with phi = |x| + dphi(x), dphi integrable:
   dOmega = -N_d V^2 c [ Int|e|L de  +  T^2 ( I0 ln(Lam/2piT) - I1 ) ]   =>   C/T log coefficient = B_el + 2 I0
   b_dial = 1 + 3 I0 / pi^2,   I0 = Int dx [phi(x) - |x|]
 (1) numeric check of this formula with window and cutoff scans,
 (2) conformal flat band: Im Sigma_f ~ x coth(x/2)  ->  I0 = 2pi^2/3, b_dial = 3 exactly,
 (3) Q_dial = C_b / b_dial with C_b = (hbar/tau_tr per k_BT)/lambda_d from the Kubo average,
 (4) paper-I arctan bath family (chi'' = (2/pi) arctan(w/gamma T)): dphi has a -1/x tail, so b_dial runs as ln T; quoted per window."""
import numpy as np
from scipy.special import digamma
from scipy.integrate import quad
def L(e,T,Lam=1e4): return np.log(Lam/(2*np.pi*T)) - np.real(digamma(0.5+1j*e/(2*np.pi*T)))
def bnum(phi,E,Tlo,Thi,n=60001):
    Ts=np.exp(np.linspace(np.log(Tlo),np.log(Thi),24))
    def Om(T):
        e=np.linspace(-E,E,n); return -np.trapezoid(T*phi(e/T)*L(e,T),e)
    O=np.array([Om(T) for T in Ts]); S=-np.gradient(O,Ts); CT=np.gradient(S,Ts)
    k=slice(3,-3); B=-np.polyfit(np.log(Ts[k]),CT[k],1)[0]; return B/(2*np.pi**2/3)
xc=lambda x:np.where(np.abs(x)<1e-8,2.0,x/np.tanh(x/2+1e-300))
I0=quad(lambda x:2*(x/np.tanh(x/2)-x),1e-9,200)[0]
print(f'conformal: I0 = {I0:.5f}  (2pi^2/3 = {2*np.pi**2/3:.5f})  -> b_dial = {1+3*I0/np.pi**2:.4f}')
for E,(a,b) in [(60,(0.05,0.6)),(120,(0.05,0.6)),(60,(0.02,0.3)),(200,(0.1,2.0))]:
    print(f'   numeric E={E:4d} T in [{a},{b}]: b_dial = {bnum(xc,E,a,b):.3f}')
def kubo(phi):
    x=np.linspace(-60,60,40001); mf=1/(4*np.cosh(x/2)**2)
    return 2/np.trapezoid(mf/(np.pi*phi(x)),x)
r=kubo(xc); Cb=r/2; print(f'   C_b = {Cb:.4f} (2pi^3/7zeta3 = {2*np.pi**3/(7*1.2020569):.4f}),  Q_dial = C_b/b_dial = {Cb/3:.3f}  (paper I own-channel conformal: {Cb/1.5:.3f})')
# a sharp-tailed, sum-rule-preserving Planckian floor: |x| + a*exp(-|x|) -> I0 = 2a
for a in [0.5,1.0,2.0]:
    f=lambda x,a=a:np.abs(x)+a*np.exp(-np.abs(x))
    print(f'floor |x|+{a}e^-|x|: formula b={1+6*a/np.pi**2:.3f}  numeric b={bnum(f,60,0.05,0.6):.3f}  C_b={kubo(f)/2:.3f}  Q_dial={kubo(f)/2/(1+6*a/np.pi**2):.3f}')
# Shape scan: Q_dial = C_b/b_dial for integrable thermal excesses of several forms and strengths
forms={'exp':lambda x,a:np.abs(x)+a*np.exp(-np.abs(x)),
       'sech':lambda x,a:np.abs(x)+a/np.cosh(x/2),
       'gauss':lambda x,a:np.abs(x)+a*np.exp(-x**2/4),
       'smooth':lambda x,a:np.sqrt(x**2+a**2*np.exp(-np.abs(x)/3))}  # sqrt-like core, exponentially closed tail
print('shape scan (a = floor strength): b_dial from I0 formula, C_b from Kubo, Q_dial = C_b/b_dial')
for nm,f in forms.items():
    row=[]
    for a in [0.1,0.3,1,3,10]:
        g=lambda x,a=a,f=f:f(x,a); xx=np.linspace(-200,200,400001)
        I=np.trapezoid(g(xx)-np.abs(xx),xx); b=1+3*I/np.pi**2; Cb=kubo(g)/2
        row.append(f'a={a}: b={b:.2f} Cb={Cb:.2f} Q={Cb/b:.2f}')
    print(f'  {nm:6s} '+' | '.join(row))
print('NOTE: baths whose spectral weight carries a 1/w tail set by T (paper-I arctan family, Gamma = gamma T) give a')
print('non-integrable excess (dphi ~ -ln x): b_dial then runs with ln T and needs the self-consistent thermal-mass calculation (not done).')
