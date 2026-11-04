"""The maximizing bath of 17_sharp_bound.py at fine resolution (Sec. V, Eq. for the ceiling).
Single step chi = sgn(y) theta(|y| - s0) with s0 = 2 ln[(1+sqrt5)/2] = ln[2/(3-sqrt5)], the one gap with zero high-energy
floor c'; C_b, b_dial and Q_dial there, and the conformal reference phi = x coth(x/2)."""
import numpy as np
x=np.linspace(0,80,400001); dx=x[1]-x[0]; w=1/(4*np.cosh(x/2)**2); nF=lambda z:0.5*(1-np.tanh(z/2))
def phi_step(s):
    d=1-nF(x-s)-nF(x+s); return 2*np.log(1/np.tanh(s/2))+np.concatenate([[0],np.cumsum(0.5*(d[1:]+d[:-1])*dx)])
def stats(ph,c):
    inv=2*np.trapezoid(w/ph,x); I0=2*np.trapezoid(ph-x-c,x); b=1+3*I0/np.pi**2; Cb=np.pi/inv; return Cb,b,Cb/b
s0=np.log(2/(3-np.sqrt(5))); c0=-2*np.log(1-np.exp(-s0))-s0
Cb,b,Q=stats(phi_step(s0),c0); print(f"s0 = {s0:.5f} (golden: 2 ln phi = {2*np.log((1+np.sqrt(5))/2):.5f}), c' = {c0:.1e}: C_b = {Cb:.4f}, b_dial = {b:.4f}, Q = {Q:.4f}")
xc=np.where(x<1e-9,2.0,x/np.tanh(np.maximum(x,1e-9)/2)); print("conformal: C_b=%.4f b=%.4f Q=%.4f"%stats(xc,0.0))
print("ratio sharp/relaxed = %.3f, conformal/sharp = %.3f"%(Q/3.2337, 2.4566/Q))
