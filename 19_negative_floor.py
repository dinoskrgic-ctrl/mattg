"""The ceiling for baths slower than conformal (Appendix C).
Monotone baths with a negative high-energy floor c' < 0: single-step scan for s >= s0 and 40 000 random mixtures of 1-4
steps with c' <= 0, using the step basis of 17_sharp_bound.py and I0 counted from |x| + c'."""
import numpy as np
x=np.linspace(0,60,6001); dx=x[1]-x[0]; w=1/(4*np.cosh(x/2)**2); nF=lambda z:0.5*(1-np.tanh(z/2))
s=np.exp(np.linspace(np.log(0.02),np.log(12),200))
PH=[];CP=[];J=[]
for si in s:
    d=1-nF(x-si)-nF(x+si); ph=2*np.log(1/np.tanh(si/2))+np.concatenate([[0],np.cumsum(0.5*(d[1:]+d[:-1])*dx)])
    c=-2*np.log(1-np.exp(-si))-si; PH.append(ph); CP.append(c); J.append(2*np.trapezoid(ph-x-c,x))
PH=np.array(PH); CP=np.array(CP); J=np.array(J)
def Q(p): ph=p@PH; return np.pi/(2*np.trapezoid(w/ph,x)*(1+3*(p@J)/np.pi**2))
s0=2*np.log((1+np.sqrt(5))/2)
sing=[(si,CP[i],Q(np.eye(len(s))[i])) for i,si in enumerate(s) if si>=s0*0.999]
print("single steps with c'<=0: max Q = %.4f at s = %.3f (c' = %+.3f)"%max(((q,si,c) for si,c,q in sing)))
for c_t in [-0.1,-0.25,-0.5]:
    i=np.argmin(abs(CP-c_t)); print(f"   single step with c' = {CP[i]:+.2f}: Q = {Q(np.eye(len(s))[i]):.3f}")
rng=np.random.default_rng(1); best=(0,None)
for t in range(40000):
    k=rng.integers(1,5); idx=rng.choice(len(s),k,replace=False); p=np.zeros(len(s)); p[idx]=rng.exponential(1,k); p/=p.sum()
    if p@CP>0: continue
    q=Q(p)
    if q>best[0]: best=(q,p@CP)
print("random mixtures with c'<=0: max Q = %.4f (its c' = %+.4f)"%best)
