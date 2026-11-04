"""Errors of the 12-32 K slopes (Sec. IV, Fig. 3a).
(a) 2 x the least-squares standard error of a robust linear fit; (b) moving-block bootstrap of the residuals;
(c) spread over fit windows 12-30, 14-30, 14-32, 12-34 K. The quoted errors are (a) and (c) in quadrature; written to
slope_sigma.npy. The slope law is refitted with each error set for comparison."""
import numpy as np, os
from scipy.optimize import curve_fit
H=os.path.dirname(os.path.abspath(__file__))
d=np.genfromtxt(os.path.join(H,'data/S15b_D3_nu2.1_digitized.csv'),delimiter=',',skip_header=2)
Ds=np.array([0.14,0.20,0.24,0.27,0.30,0.34,0.38])
def robust(T,R):
    m=np.ones(len(T),bool)
    for _ in range(5):
        p=np.polyfit(T[m],R[m],1); r=R-np.polyval(p,T); s=np.std(r[m]); m=np.abs(r)<max(2.5*s,0.01)
    return T[m],R[m]
def fit(T,R):
    X=np.c_[T,np.ones(len(T))]; c,res,*_=np.linalg.lstsq(X,R,rcond=None); r=R-X@c
    se=np.sqrt(np.sum(r**2)/(len(T)-2)/np.sum((T-T.mean())**2)); return 1e3*c[0],1e3*se,r
rng=np.random.default_rng(0); out=[]
for D in Ds:
    a=d[d[:,0]==D]; o=np.argsort(a[:,1]); T,R=a[o,1],a[o,2]
    k=(T>12)&(T<32); Tw,Rw=robust(T[k],R[k]); s,se,r=fit(Tw,Rw)
    # lag-1 autocorrelation of residuals -> effective inflation sqrt((1+rho)/(1-rho))
    rho1=np.corrcoef(r[:-1],r[1:])[0,1]; infl=np.sqrt((1+rho1)/(1-rho1))
    # moving-block bootstrap on residuals
    n=len(Tw); B=10; fitted=Rw-r; bs=[]
    for _ in range(2000):
        starts=rng.integers(0,n-B,int(np.ceil(n/B))); rr=np.concatenate([r[i:i+B] for i in starts])[:n]
        bs.append(fit(Tw,fitted+rr)[0])
    sb=np.std(bs)
    win=[]
    for lo,hi in ((14,30),(12,30),(14,32),(12,34)):
        kk=(T>lo)&(T<hi); win.append(fit(*robust(T[kk],R[kk]))[0])
    sw=np.std(np.r_[s,win],ddof=1)
    out.append((D,s,2*se,se*infl,sb,sw,len(Tw),rho1))
out=np.array(out)
print(" D     slope  2xSE   SE*AR1  block-bs  window-sd  quoted   n   rho1")
sig_q=np.round(np.hypot(np.round(out[:,2],2),np.round(out[:,5],2)),2)   # quoted errors, from the two columns at two decimals
np.save(os.path.join(H,"slope_sigma.npy"),sig_q)
for (D,s,a,b,c,w,n,rh),t in zip(out,sig_q): print(f"{D:.2f}  {s:6.2f}  {a:5.2f}  {b:5.2f}   {c:5.2f}    {w:5.2f}    {t:5.2f}  {int(n):3d}  {rh:+.2f}")
S=out[:,1]; f2=lambda D,a,D0:a*(1+D**2/D0**2)
for name,sg in (("2xSE",out[:,2]),("block bootstrap",out[:,4]),("2xSE (+) window sd",np.hypot(out[:,2],out[:,5]))):
    p,cv=curve_fit(f2,Ds,S,[3.6,0.13],sigma=sg,absolute_sigma=True); chi=np.sum(((S-f2(Ds,*p))/sg)**2)
    f4=lambda D,a,c,e:a+c*D**2+e*D**4; q,cq=curve_fit(f4,Ds,S,[3.6,200,0],sigma=sg,absolute_sigma=True)
    qs=q[2]*0.38**4/f4(0.38,*q); qse=np.sqrt(cq[2,2])*0.38**4/f4(0.38,*q)
    print(f"{name:20s}: a = {p[0]:.2f} +- {np.sqrt(cv[0,0]):.2f}, D0 = {p[1]:.3f} +- {np.sqrt(cv[1,1]):.3f}, chi2 = {chi:.1f}/5, quartic share at 0.38 = {100*qs:+.1f} +- {100*qse:.1f}%")
fp=lambda D,a,c,pp:a+c*D**pp
for name,sg in (("2xSE",out[:,2]),("2xSE (+) window sd",np.hypot(out[:,2],out[:,5]))):
    q,cq=curve_fit(fp,Ds,S,[3.6,200,2],sigma=sg,absolute_sigma=True); print(f"{name:20s}: free exponent p = {q[2]:.2f} +- {np.sqrt(cq[2,2]):.2f}")
