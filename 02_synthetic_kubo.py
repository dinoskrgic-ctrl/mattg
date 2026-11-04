"""Exact thermal average on synthetic curves (Sec. III, Table I).
sigma ~ Int dw (-df/dw) / Gamma(w), Gamma(w) = Gamma_c + <|V|^2> g(w,T), g = -Im G_f(k_D, w). The harmonic mean is linear in
<|V|^2> only if g is frequency independent over a few k_B T. Synthetic curves W = 17.4 (D^2 + 0.16^2) kOhm, R_el = 0.6 kOhm,
Delta = kappa k_B x 57.8 K, analysed like the data: quartic share of the 12-32 K slopes at 0.38 V/nm and T_max.
Flat-level self-energy Im Sigma_f = -kappa sqrt(w^2 + (pi T)^2)/pi, with or without its marginal real part."""
import numpy as np
Ts=57.8; D0=0.160; Wc=17.37; Rel=0.6
Ds=np.array([0.10,0.14,0.20,0.24,0.27,0.30,0.34,0.38])
w=np.linspace(-1500,1500,30001)
def g(T,kappa,Delta,real,sym):
    Gf=kappa*np.sqrt(w**2+(np.pi*T)**2)/np.pi
    Re= -(2*kappa/np.pi**2)*w*np.log(np.sqrt(w**2+(np.pi*T)**2)/1500.) if real else 0*w  # marginal KK partner, cutoff 1500 K
    out=0
    for s in ((+1,-1) if sym else (+1,)):
        out=out+ Gf/((w-s*Delta-Re)**2+Gf**2)
    return out/len((+1,-1) if sym else (+1,))
def rho(T,D,kappa,real=False,sym=False):
    Delta=kappa*Ts
    # normalise so that the omega=0 approximation reproduces the zero-frequency profile: K*V^2*g(0) = W*x/(1+x^2) with W = Wc(D^2+D0^2)
    V2K = Wc*(D**2+D0**2)*Delta   # K*V^2 in kOhm*K
    gg=g(T,kappa,Delta,real,sym)
    Gam = Rel + V2K*gg            # = K*Gamma(w) in kOhm
    mf=1/(4*T*np.cosh(w/(2*T))**2)
    return 1/np.trapezoid(mf/Gam,w)
Tgrid=np.arange(12,79,1.0)
def analyse(kappa,real,sym,label):
    R=np.array([[rho(T,D,kappa,real,sym) for T in Tgrid] for D in Ds])
    k=(Tgrid>=12)&(Tgrid<=32); sl=np.array([np.polyfit(Tgrid[k],r[k],1)[0]*1e3 for r in R])
    fit=Ds>=0.14; x=Ds[fit]**2
    q2=np.polyfit(x,sl[fit],1); r2=np.std(sl[fit]-np.polyval(q2,x))/sl[fit].mean()
    A=np.c_[np.ones_like(x),x,x**2]; c=np.linalg.lstsq(A,sl[fit],rcond=None)[0]
    quart=c[2]*0.38**4/(c[0]+c[1]*0.38**2+c[2]*0.38**4)
    Tmax=[Tgrid[np.argmax(r)] if np.argmax(r)<len(Tgrid)-1 else np.nan for r in R]
    M=R-R[:,[0]]; S=np.linalg.svd(M,compute_uv=False)
    lowT=np.array([rho(2.0,D,kappa,real,sym) for D in Ds])
    print(f'{label}: slopes12-32 {np.round(sl,1)}')
    print(f'    D^2 fit: a={q2[1]:.2f}, D0={np.sqrt(q2[1]/q2[0]):.3f}, rel rms {r2:.3f}; quartic share at 0.38 = {100*quart:+.1f}%; T_max {Tmax}; s2/s1={S[1]/S[0]:.4f}')
    print(f'    R(2 K) [kOhm] (elastic proxy): {np.round(lowT,3)}  ratio 0.38/0.14 = {lowT[-1]/lowT[1]:.2f}')
for kappa in (0.5,1.0,2.0):
    analyse(kappa,False,False,f'kappa={kappa} single level, no ReSigma_f')
    analyse(kappa,True,False, f'kappa={kappa} single level, marginal ReSigma_f')
    analyse(kappa,False,True, f'kappa={kappa} symmetric levels +-Delta')
