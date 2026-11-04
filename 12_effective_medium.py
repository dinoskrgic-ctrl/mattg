"""Smooth mirror-breaking disorder (Sec. IV, Background). Run 06_four_parameter_law.py first.
If h(r) varies slowly on the scale of the mean free path, the sample is a 2D network of local resistivities
rho(r) = R_el + W2 |D + h(r)|^2 F(T/T*), h complex Gaussian with <|h|^2> = D0^2. Bruggeman effective medium in 2D,
< (s - s_e)/(s + s_e) > = 0 with s = 1/rho, compared with the homogeneous law; refit of a + c D^2 + e D^4 to the slopes."""
import numpy as np
from scipy.optimize import brentq
R_el,W2,D0,Ts=np.load('fit4.npy')
rng=np.random.default_rng(0); N=40000
h=(rng.normal(size=N)+1j*rng.normal(size=N))*D0/np.sqrt(2)
F=lambda T:(T/Ts)/(1+(T/Ts)**2)
Ds=np.array([0.10,0.14,0.20,0.24,0.27,0.30,0.34,0.38]); T=np.linspace(12,32,21)
def Rema(D,t):
    rho=R_el+W2*np.abs(D+h)**2*F(t); s=1/rho
    se=brentq(lambda se:np.mean((s-se)/(s+se)),s.min(),s.max()); return 1/se
def fitq(sl):
    A=np.vstack([np.ones_like(Ds),Ds**2,Ds**4]).T; c=np.linalg.lstsq(A,sl,rcond=None)[0]
    return c, c[2]*0.38**4/(c[0]+c[1]*0.38**2+c[2]*0.38**4)
sh=[];se=[]
for D in Ds:
    Rh=R_el+W2*(D**2+D0**2)*F(T); Re=np.array([Rema(D,t) for t in T])
    sh.append(np.polyfit(T,Rh,1)[0]*1e3); se.append(np.polyfit(T,Re,1)[0]*1e3)
sh=np.array(sh); se=np.array(se)
print(f"params: R_el={R_el:.3f} kOhm W2={W2:.2f} kOhm D0={D0:.3f} T*={Ts:.1f} K")
print("D      slope_hom  slope_EMA  ratio")
for D,a,b in zip(Ds,sh,se): print(f"{D:.2f}  {a:8.2f}  {b:8.2f}  {b/a:.3f}")
for lab,sl in (('homogeneous',sh),('EMA',se)):
    c,share=fitq(sl); print(f"{lab}: a={c[0]:.2f} D0_apparent={np.sqrt(c[0]/c[1]) if c[1]>0 else float('nan'):.3f} quartic share at 0.38 = {100*share:+.1f}%")
