"""Shared data loader and exact thermally averaged Kubo model.
Data: device D3, filling nu = 2.1, R_xx(T) at eight displacement fields, digitized from Supplemental Fig. S15b of
Zhang et al., PRL 137, 046301 (2026). Flat-band rate Gamma_f(0,T) = kappa k_B T; levels at +Delta (single) or +-Delta (sym)."""
import numpy as np, os
H=os.path.dirname(os.path.abspath(__file__))
Ds=np.array([0.14,0.20,0.24,0.27,0.30,0.34,0.38])
def clean(T,R,lo=12,hi=78,deg=6):
    o=np.argsort(T);T,R=T[o],R[o];k=(T>lo)&(T<hi);T,R=T[k],R[k];m=np.ones(len(T),bool)
    for _ in range(8):
        p=np.polyfit(T[m],R[m],deg);r=R-np.polyval(p,T);s=np.std(r[m]);m=np.abs(r)<max(2.5*s,0.008)
    return T[m],R[m]
def load(lo=12,hi=78):
    d=np.genfromtxt(os.path.join(H,'data/S15b_D3_nu2.1_digitized.csv'),delimiter=',',skip_header=2)
    C=[clean(d[d[:,0]==D,1],d[d[:,0]==D,2],lo,hi) for D in Ds]
    e=np.genfromtxt(os.path.join(H,'data/S15b_D0.10_redigitized.csv'),delimiter=',',comments='#')
    C10=clean(e[:,0],e[:,1],lo,hi)
    return C,C10
X=np.linspace(-25,25,1001)            # omega/T grid
MF=1/(4*np.cosh(X/2)**2)              # -df/dx
def gtilde(T,kappa,Delta,level='single',real=True,wc=1500.):
    """Delta * (-Im G_f(k_D, w)) on the grid w = X*T, for each T (array). Gamma_f(0,T) = kappa*T."""
    T=np.atleast_1d(T)[:,None]; w=X[None,:]*T
    G=kappa*np.sqrt(w**2+(np.pi*T)**2)/np.pi
    Re=-(2*kappa/np.pi**2)*w*np.log(np.sqrt(w**2+(np.pi*T)**2)/wc) if real else 0*w
    lv=(1,) if level=='single' else (1,-1)
    return sum(Delta*G/((w-s*Delta-Re)**2+G**2) for s in lv)/len(lv)
def rho(T,Rel,A,kappa,Delta,**kw):
    """exact dc resistivity: 1/<1/(Rel + A*gtilde)> with Fermi window (kOhm)."""
    g=gtilde(T,kappa,Delta,**kw)
    return 1/np.trapezoid(MF[None,:]/(Rel+A*g),X,axis=1)
