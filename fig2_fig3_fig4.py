"""Figures 2-4 (SciencePlots). Run 06_four_parameter_law.py and 08_slope_errors.py first.
Fig. 2: (a) all eight curves with the four-parameter law fitted on D = 0.20-0.38 (solid) and extrapolated to 0.14 and 0.10
with only the elastic offset adjusted (dashed); (b) [R - R_el(D)]/W(D) against T/T*.
Fig. 3: (a) 12-32 K slope against D with the out-of-sample point at 0.10 V/nm; (b) elastic offset and inelastic amplitude,
normalized at D = 0.30 V/nm. Fig. 4: the (beta, eta) map with the field moving the Dirac carriers at fixed beta."""
import numpy as np, os, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, scienceplots
from common import load, Ds
plt.style.use(['science','no-latex']); plt.rcParams.update({'font.size':8,'axes.labelsize':8,'legend.fontsize':6.5})
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),'figures')
C,C10=load(); q=np.load('fit4.npy'); offs=np.load('offsets.npy')
F=lambda T,s:(T/s)/(1+(T/s)**2)
allC=[(C10,0.10)]+list(zip(C,Ds)); offd={0.10:offs[-1]}; offd.update({D:o for D,o in zip(Ds,offs[:-1])})
cm=plt.cm.viridis(np.linspace(0,0.92,8))
fig,ax=plt.subplots(1,2,figsize=(6.8,2.6))
for i,((T,R),D) in enumerate(allC):
    mk='o' if D>=0.2 else 's'; low=D<0.2
    ax[0].plot(T,R,mk,ms=1.2,color=cm[i],mfc='none' if low else cm[i],mew=0.4,label=f'{D:.2f}')
    Tg=np.linspace(12,78,200); W=q[1]*(D**2+q[2]**2)
    ax[0].plot(Tg,offd[D]+W*F(Tg,q[3]),'--' if low else '-',color='k',lw=0.6)
    ax[1].plot(T/q[3],(R-offd[D])/W,mk,ms=1.2,color=cm[i],mfc='none' if low else cm[i],mew=0.4)
xx=np.linspace(0.18,1.38,200); ax[1].plot(xx,xx/(1+xx**2),'k-',lw=0.9)
ax[0].set_xlabel(r'$T$ (K)'); ax[0].set_ylabel(r'$R_{xx}$ (k$\Omega$)')
h_,l_=ax[0].get_legend_handles_labels(); ax[1].legend(h_,l_,title=r'$D$ (V/nm)',title_fontsize=6.5,ncol=2,loc='lower right',markerscale=2.5,handletextpad=0.2,columnspacing=0.8)
ax[1].set_xlabel(r'$T/T^*$'); ax[1].set_ylabel(r'$[R_{xx}-R_{\rm el}(D)]/W(D)$')
for a,l in zip(ax,'ab'): a.text(0.04,0.92,f'({l})',transform=a.transAxes)
fig.savefig(os.path.join(out,'fig2_collapse.pdf')); fig.savefig(os.path.join(out,'fig2_collapse.png'),dpi=300); plt.close(fig)
# fig 2
S=np.array([7.26,12.07,15.35,18.16,21.52,27.53,32.83]); sig=np.load('slope_sigma.npy')   # 08_slope_errors.py
from scipy.optimize import curve_fit
(a,D0),_=curve_fit(lambda D,a,D0:a*(1+D**2/D0**2),Ds,S,[3.6,0.13],sigma=sig,absolute_sigma=True)
(la,lc),_=curve_fit(lambda D,a,c:a+c*D,Ds,S,sigma=sig,absolute_sigma=True); DD=np.linspace(0,0.42,200)
fig,ax=plt.subplots(1,2,figsize=(6.8,2.6))
ax[0].errorbar(Ds,S,sig,fmt='o',ms=3,color='k',label='fitted')
ax[0].errorbar([0.10],[5.48],[0.6],fmt='s',ms=3.5,mfc='w',color='C3',label='not fitted')
ax[0].plot(DD,a*(1+DD**2/D0**2),'k-',lw=0.9,label=r'$a(1+D^2/D_0^2)$')
ax[0].plot(DD,la+lc*DD,'--',color='C0',lw=0.8,label=r'$a+c|D|$')
ax[0].set_xlabel(r'$D$ (V/nm)'); ax[0].set_ylabel(r'd$R_{xx}$/d$T$, 12–32 K ($\Omega$/K)'); ax[0].set_ylim(0,38); ax[0].set_xlim(0,0.42); ax[0].legend(loc='upper left')
Dall=np.r_[0.10,Ds]; oall=np.r_[offs[-1],offs[:-1]]; Wall=q[1]*(Dall**2+q[2]**2)
n=list(Dall).index(0.30)
ax[1].plot(Dall,oall/oall[n],'s-',ms=3,color='C1',lw=0.8,label=r'elastic $R_{\rm el}$')
ax[1].plot(Dall,Wall/Wall[n],'o-',ms=3,color='C0',lw=0.8,label=r'inelastic $W$')
ax[1].plot(Dall,(Dall**2+q[2]**2)/(0.30**2+q[2]**2),':',color='C0',lw=0.6)
ax[1].set_xlabel(r'$D$ (V/nm)'); ax[1].set_ylabel(r'normalized at $D=0.30$ V/nm'); ax[1].set_xlim(0.07,0.41); ax[1].legend(loc='upper left')
for a_,l in zip(ax,'ab'): a_.text(0.9,0.06,f'({l})',transform=a_.transAxes)
fig.savefig(os.path.join(out,'fig3_slopes.pdf')); fig.savefig(os.path.join(out,'fig3_slopes.png'),dpi=300); plt.close(fig)
# fig 3: (beta, eta) map schematic with Q = C_b(beta) eta, C_b ~ 3.6 beta (paper I)
fig,ax=plt.subplots(figsize=(3.4,2.6))
b=np.linspace(1.5,6,200); e=np.linspace(0.005,1.0,300); B,E=np.meshgrid(b,e); Qm=3.6*B*E
cs=ax.contour(B,E,Qm,levels=[0.5,1,2,4.91,10,15],colors='0.6',linewidths=0.5,linestyles='--')
ax.clabel(cs,fmt=lambda v:f'Q={v:g}',fontsize=5.5)
ax.plot(2,2/3,'*',color='C2',ms=7,label='conformal')
bf=3.4; Ds_=np.array([0,0.14,0.20,0.27,0.38]); eta0=0.012; etas=eta0*(Ds_**2+0.15**2)/0.15**2
ax.plot([bf]*len(Ds_),etas,'o-',color='C3',ms=3,lw=0.8,label=r'Dirac carriers, $D=0\to0.38$ V/nm (schematic)')
ax.annotate('',xy=(bf,etas[-1]*1.6),xytext=(bf,etas[-1]),arrowprops=dict(arrowstyle='->',color='C3',lw=0.8))
ax.set_yscale('log'); ax.set_ylim(0.005,1.2); ax.set_xlim(1.5,6)
ax.set_xlabel(r'$\beta$ (bath slowness, inherited from the flat band)'); ax.set_ylabel(r'$\eta$ (entropy share)'); ax.legend(loc='lower right')
fig.savefig(os.path.join(out,'fig4_map.pdf')); fig.savefig(os.path.join(out,'fig4_map.png'),dpi=300); plt.close(fig)
print('figures written')
