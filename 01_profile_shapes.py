"""Factorization and profile shape (Sec. IV, Table II).
(1) One shared shape parameter with a free offset and amplitude per curve, for seven peaked profiles and one monotonic
saturating profile (Table II). (2) The same with one common elastic offset. (3) Offsets, amplitudes and 12-32 K slopes
per curve. (4) SVD amplitudes of R - R(12 K) and D0 from slopes, profile amplitudes and SVD."""
import numpy as np, os
from scipy.optimize import least_squares
h=os.path.dirname(os.path.abspath(__file__))
d=np.genfromtxt(os.path.join(h,'data/S15b_D3_nu2.1_digitized.csv'),delimiter=',',skip_header=2)
Ds=[0.14,0.20,0.24,0.27,0.30,0.34,0.38]
def clean(T,R,lo=12,hi=78,deg=6):
    o=np.argsort(T);T,R=T[o],R[o];k=(T>lo)&(T<hi);T,R=T[k],R[k];m=np.ones(len(T),bool)
    for _ in range(8):
        p=np.polyfit(T[m],R[m],deg);r=R-np.polyval(p,T);s=np.std(r[m]);m=np.abs(r)<max(2.5*s,0.008)
    return T[m],R[m]
C=[clean(d[d[:,0]==D,1],d[d[:,0]==D,2]) for D in Ds]
shapes={
 'x/(1+x^2)  [Lorentzian level]': lambda T,s: (T/s)/(1+(T/s)**2),
 'x/(1+x)^2  [generic]':                lambda T,s: (T/s)/(1+T/s)**2,
 'x exp(-x)  [generic]':                lambda T,s: (T/s)*np.exp(-T/s),
 'x/(1+x^3)^(1/3)*... tanh-type: tanh(x)/(1+x^2)^0.5': lambda T,s: np.tanh(T/s)/np.sqrt(1+(T/s)**2),
 'arctan(x)-x/2-ish: x/(1+x^2)^(3/4)':  lambda T,s: (T/s)/(1+(T/s)**2)**0.75,
 'log-Kondo-like: ln(1+x)/(1+x)':        lambda T,s: np.log1p(T/s)/(1+T/s),
 'x/(1+x)  [monotonic, no maximum]':     lambda T,s: (T/s)/(1+T/s),
}
def fit(shape,common_offset=False):
    def res(q):
        s=q[0]; out=[]; 
        if common_offset:
            r0=q[1]
            for T,R in C:
                f=shape(T,s); W=np.dot(f,R-r0)/np.dot(f,f); out.append(R-r0-W*f)
        else:
            for T,R in C:
                f=shape(T,s); X=np.c_[np.ones_like(f),f]; c=np.linalg.lstsq(X,R,rcond=None)[0]; out.append(R-X@c)
        return np.concatenate(out)
    x0=[[a] if not common_offset else [a,0.6] for a in (10,20,40,60,90,150)]
    b=min((least_squares(res,q,bounds=([1]+([0] if common_offset else []),[1000]+([2] if common_offset else []))) for q in x0),key=lambda z:z.cost)
    return b, 1e3*np.sqrt(np.mean(b.fun**2))
print('(1)+(2) one shared shape parameter, free offset+amplitude per curve (15 params), 12-78 K')
for k,sh in shapes.items():
    b,r=fit(sh); print(f'   {k:45s} scale = {b.x[0]:7.1f} K  rms = {r:5.1f} Ohm')
print('    -> model prediction: offset R_el independent of D. Same shapes with ONE common offset (9 params):')
for k,sh in list(shapes.items())[:3]:
    b,r=fit(sh,True); print(f'   {k:45s} scale = {b.x[0]:7.1f} K  R_el = {b.x[1]:.3f} kOhm  rms = {r:5.1f} Ohm')
# per-curve offsets with the Lorentzian profile, and restricted to the reliable-at-low-T curves 0.20-0.30
sh=shapes['x/(1+x^2)  [Lorentzian level]']; b,_=fit(sh); s=b.x[0]
offs=[];Ws=[]
for T,R in C:
    f=sh(T,s); X=np.c_[np.ones_like(f),f]; c=np.linalg.lstsq(X,R,rcond=None)[0]; offs.append(c[0]); Ws.append(c[1])
offs=np.array(offs);Ws=np.array(Ws)
print('(3) Lorentzian offsets R_el [kOhm]:',np.round(offs,3),' amplitudes W:',np.round(Ws,3))
print(f'    over 0.14-0.30 (low-T digitization reliable for 0.20-0.30): R_el rises x{offs[4]/offs[0]:.2f}; W rises x{Ws[4]/Ws[0]:.2f}')
# raw low-T resistance (model: nearly elastic since F(5K) ~ 0.09)
raw=[]
for D in Ds:
    k=d[:,0]==D;T=d[k,1];R=d[k,2]; raw.append(np.median(R[(T>4.3)&(T<5.7)]) if np.any((T>4.3)&(T<5.7)) else np.nan)
print('    raw R(5 K) [kOhm]:',np.round(raw,3),'(0.34/0.38 unreliable below 12 K per digitization header)')
# slope ratios vs amplitude ratios
sl=[]
for T,R in C:
    k=(T>=12)&(T<=32); sl.append(np.polyfit(T[k],R[k],1)[0]*1e3)
sl=np.array(sl)
print('    12-32 K slopes [Ohm/K]:',np.round(sl,2))
print(f'    slope(0.38)/slope(0.14) = {sl[-1]/sl[0]:.2f}  vs  W(0.38)/W(0.14) = {Ws[-1]/Ws[0]:.2f}  (exact factorization => equal)')
# model slope over 12-32 from fitted curves
ms=np.array([W*(sh(32,s)-sh(12,s))/20*1e3 for W in Ws]); print('    model slopes 12-32 K from the fit [Ohm/K]:',np.round(ms,2))
# (4) SVD on R - R(12K), 14-55 K
Tg=np.linspace(14,55,60); M=[]
for T,R in C:
    Ri=np.interp(Tg,T,np.convolve(R,np.ones(5)/5,'same')); R12=np.interp(12.5,T,np.convolve(R,np.ones(5)/5,'same')); M.append(Ri-R12)
M=np.array(M); U,S,Vt=np.linalg.svd(M,full_matrices=False); amp=U[:,0]*S[0]; amp=amp/amp[0]
print('(4) SVD amplitudes of R-R(12K), normalized:',np.round(amp,2),' s2/s1 =',round(S[1]/S[0],4))
# D^2 fit to each amplitude set
for name,A in (('slopes',sl),('profile W',Ws),('SVD',amp)):
    q=np.polyfit(np.array(Ds)**2,A,1); print(f'    {name:10s}: D0 = {np.sqrt(q[1]/q[0]):.3f} V/nm, rel rms {np.std(A-np.polyval(q,np.array(Ds)**2))/np.mean(A):.3f}')
