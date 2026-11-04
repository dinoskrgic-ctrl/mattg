"""How a field-independent background moves the apparent maximum at low field (Sec. IV).
Eight-curve background fit (b = 1.8 Ohm/K, T* = 52.6 K, D0 = 0.104 V/nm) and the D >= 0.20 fit (b = 0.27, T* = 58.1,
D0 = 0.136): apparent T_max of R = R_el + W2 (D^2 + D0^2) x/(1+x^2) + b T, located like the data (quadratic, 40-76 K)."""
import numpy as np
T=np.linspace(40,76,200)
for b,Ts,D0,W2 in ((1.8e-3,52.6,0.104,15.9),(0.27e-3,58.1,0.136,15.9)):
    out=[]
    for D in (0.10,0.14,0.20,0.30,0.38):
        R=0.6+W2*(D**2+D0**2)*(T/Ts)/(1+(T/Ts)**2)+b*T; p=np.polyfit(T,R,2); out.append(f"{D:.2f}:{-p[1]/(2*p[0]):.0f}K")
    print(f"b={b*1e3:.2f} Ohm/K, T*={Ts}: "+"  ".join(out))
