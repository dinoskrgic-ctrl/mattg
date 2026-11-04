"""Figure 5: Q_dial against b_dial for realizable flat bands (Sec. V).
Every non-decreasing odd bath spectrum is a mixture of steps chi_s = sgn(y) theta(|y| - s), and phi is linear in the weights.
Dots: 1500 random step mixtures with unit weight and zero high-energy floor. Diamond: the golden-ratio gap
s0 = 2 ln[(1+sqrt5)/2], the maximum (17_sharp_bound.py). Star: conformal phi = x coth(x/2). Thin line: relaxed envelope
phi* = max(|x|, k/(2cosh(x/2))), which keeps only phi >= |x| and is not realizable. Dashed: 4.91, the conformal ratio in its
own channel."""
import numpy as np, os, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, scienceplots
plt.style.use(['science','no-latex']); plt.rcParams.update({'font.size':8,'axes.labelsize':8,'legend.fontsize':6.5})
here=os.path.dirname(os.path.abspath(__file__)); out=os.path.join(here,'figures')
x=np.linspace(0,70,280001); dx=x[1]-x[0]; w=1/(4*np.cosh(x/2)**2); nF=lambda z:0.5*(1-np.tanh(z/2))
def phi_step(s):
    d=1-nF(x-s)-nF(x+s); return 2*np.log(1/np.tanh(s/2))+np.concatenate([[0],np.cumsum(0.5*(d[1:]+d[:-1])*dx)])
cfl=lambda s:-2*np.log(1-np.exp(-s))-s
def st(ph,c=0.0):   # even phi on x>=0; returns (b_dial, Q_dial)
    Cb=np.pi/(2*np.trapezoid(w/ph,x)); b=1+3*2*np.trapezoid(ph-x-c,x)/np.pi**2; return b,Cb/b
s0=2*np.log((1+np.sqrt(5))/2); star=st(phi_step(s0))
rng=np.random.default_rng(4); pts=[]
while len(pts)<1500:
    n1,n2=rng.integers(1,4,2)
    sl=np.exp(rng.uniform(np.log(0.05),np.log(s0),n1)); sh=np.exp(rng.uniform(np.log(s0),np.log(8.0),n2))
    pl=rng.exponential(1,n1); ph_=rng.exponential(1,n2); pl/=pl.sum(); ph_/=ph_.sum()
    Fl=pl@cfl(sl); Fh=ph_@cfl(sh)            # Fl>0, Fh<0; mix u*low + (1-u)*high with zero floor
    u=-Fh/(Fl-Fh); p=np.r_[u*pl,(1-u)*ph_]; s=np.r_[sl,sh]
    ph=sum(pi*phi_step(si) for pi,si in zip(p,s)); c=p@cfl(s)
    pts.append(st(ph,c))
pts=np.array(pts)
X=np.r_[-x[:0:-1],x]; W=1/(4*np.cosh(X/2)**2)
def stf(phi): return 1+3*np.trapezoid(phi-np.abs(X),X)/np.pi**2, np.pi/np.trapezoid(W/phi,X)/(1+3*np.trapezoid(phi-np.abs(X),X)/np.pi**2)
env=np.array([stf(np.maximum(np.abs(X),k*np.sqrt(W))) for k in np.exp(np.linspace(np.log(0.05),np.log(60),800))])
conf=st(np.where(x<1e-9,2.0,x/np.tanh(np.maximum(x,1e-9)/2)))
fig,ax=plt.subplots(figsize=(3.375,2.5))
ax.plot(env[:,0],env[:,1],'-',lw=0.6,color='0.45',label='relaxed envelope')
ax.plot(pts[:,0],pts[:,1],'.',ms=1.5,color='tab:blue',alpha=0.5,label='random monotonic spectra')
ax.plot(*star,'kD',ms=3.5,label='golden-ratio gap')
ax.plot(*conf,'*',ms=8,color='tab:red',label='conformal flat band')
ax.axhline(2*np.pi**3/(7*1.2020569)/1.5,ls='--',lw=0.7,color='0.4'); ax.text(28,5.08,r'$Q_{\rm conf}$ (own channel)',fontsize=6.5,ha='right',color='0.3')
ax.set_xscale('log'); ax.set_xlim(1,30); ax.set_xticks([1,2,5,10,20]); ax.set_xticklabels(['1','2','5','10','20']); ax.minorticks_off(); ax.set_ylim(0,5.6)
ax.set_xlabel(r'$b_{\rm dial}$'); ax.set_ylabel(r'$Q_{\rm dial}$')
ax.legend(loc='upper right',bbox_to_anchor=(1.0,0.86),frameon=False)
fig.savefig(os.path.join(out,'fig5_bound.pdf')); fig.savefig(os.path.join(out,'fig5_bound.png'),dpi=300)
i=env[:,1].argmax(); j=pts[:,1].argmax()
print('golden step (b,Q) =',star,' conformal =',conf); print('relaxed envelope max =',env[i]); print('random mixtures: max Q =',pts[j],' b range',pts[:,0].min(),pts[:,0].max())
