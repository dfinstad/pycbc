import numpy as np

# phase of h(f)
def Phif3hPN(f, M, eta, s1x=0.0, s1y=0.0, s1z=0.0, s2x=0.0, s2y=0.0, s2z=0.0, Lam=0.0, dLam=0.0):
    gt = 4.92549094830932e-6         # GN*Msun/c^3 in seconds
    EulerGamma = 0.57721566490153286060
    vlso = 1.0/np.sqrt(6.0)
    delta = np.sqrt(1.0 - 4.0*eta)
    v = (np.pi*M*(f + 1e-100)*gt)**(1.0/3.0)
    v2 = v*v
    v3 = v2*v
    v4 = v2*v2
    v5 = v4*v
    v6 = v3*v3
    v7 = v3*v4
    v10 = v5*v5
    v12 = v10*v2
    eta2 = eta**2
    eta3 = eta**3

    m1M = 0.5*(1.0 + delta)
    m2M = 0.5*(1.0 - delta)
    chi1L = s1z
    chi2L = s2z
    chi1sq = s1x*s1x + s1y*s1y + s1z*s1z
    chi2sq = s2x*s2x + s2y*s2y + s2z*s2z
    chi1dotchi2 = s1x*s2x + s1y*s2y + s1z*s2z
    SL = m1M*m1M*chi1L + m2M*m2M*chi2L
    dSigmaL = delta*(m2M*chi2L - m1M*chi1L)

    # phase correction due to spins
    sigma = eta*(721.0/48.0*chi1L*chi2L - 247.0/48.0*chi1dotchi2)
    sigma += 719.0/96.0*(m1M*m1M*chi1L*chi1L + m2M*m2M*chi2L*chi2L)
    sigma -= 233.0/96.0*(m1M*m1M*chi1sq + m2M*m2M*chi2sq)
    phis_15PN = 188.0*SL/3.0 + 25.0*dSigmaL
    ga = (554345.0/1134.0 + 110.0*eta/9.0)*SL + (13915.0/84.0 - 10.0*eta/3.0)*dSigmaL
    pn_ss3 =  (326.75/1.12 + 557.5/1.8*eta)*eta*chi1L*chi2L
    pn_ss3 += ((4703.5/8.4 + 2935.0/6.0*m1M - 120.0*m1M*m1M) + (-4108.25/6.72 - 108.5/1.2*m1M + 125.5/3.6*m1M*m1M))*m1M*m1M*chi1sq
    pn_ss3 += ((4703.5/8.4 + 2935.0/6.0*m2M - 120.0*m2M*m2M) + (-4108.25/6.72 - 108.5/1.2*m2M + 125.5/3.6*m2M*m2M))*m2M*m2M*chi2sq
    phis_3PN = np.pi*(3760.0*SL + 1490.0*dSigmaL)/3.0 + pn_ss3
    phis_35PN = ( -8980424995.0/762048.0 + 6586595.0*eta/756.0 - 305.0*eta2/36.0)*SL  \
                     - (170978035.0/48384.0 - 2876425.0*eta/672.0 - 4735.0*eta2/144.0)*dSigmaL

    # tidal correction to phase
    # Lam is the reduced tidal deformation parameter \tilde\Lam
    # dLam is the asymmetric reduced tidal deformation parameter, which is usually negligible
    tidal = Lam*v10*(- 39.0/2.0 - 3115.0/64.0*v2) + dLam*6595.0/364.0*v12

    return 3.0/128.0/eta/v5*(1.0 + 20.0/9.0*(743.0/336.0 + 11.0/4.0*eta)*v2 + (phis_15PN - 16.0*np.pi)*v3 \
    + 10.0*(3058673.0/1016064.0 + 5429.0/1008.0*eta + 617.0/144.0*eta2 - sigma)*v4  \
    + (38645.0/756.0*np.pi - 65.0/9.0*eta*np.pi - ga)*(1.0 + 3.0*np.log(v/vlso))*v5  \
    + (11583231236531.0/4694215680.0 - 640.0/3.0*np.pi**2 - 6848.0/21.0*(EulerGamma + np.log(4.0*v)) + \
      (-15737765635.0/3048192.0 + 2255.0*np.pi**2/12.0)*eta + 76055.0/1728.0*eta2 - 127825.0/1296.0*eta3 + phis_3PN)*v6 \
    + (np.pi*(77096675.0/254016.0 + 378515.0/1512.0*eta - 74045.0/756.0*eta**2) + phis_35PN)*v7 + tidal)

# correction to modulus of h(f)
# normalization should be C * (GN*Mchirp/c^3)^5/6 * f^-7/6 / D / pi^2/3 * (5/24)^1/2
def Af3hPN(f, M, eta, s1x=0.0, s1y=0.0, s1z=0.0, s2x=0.0, s2y=0.0, s2z=0.0, Lam=0.0, dLam=0.0):
    gt = 4.92549094830932e-6         # GN*Msun/c^3 in seconds
    EulerGamma = 0.57721566490153286060
    delta = np.sqrt(1.0 - 4.0*eta)
    v = (np.pi*M*(f + 1e-100)*gt)**(1.0/3.0)
    v2 = v*v
    v3 = v2*v
    v4 = v2*v2
    v5 = v4*v
    v6 = v3*v3
    v7 = v3*v4
    eta2 = eta**2
    eta3 = eta**3

    # modulus correction due to aligned spins
    chis = 0.5*(s1z + s2z)
    chia = 0.5*(s1z - s2z)
    be = 113.0/12.0*(chis + delta*chia - 76.0/113.0*eta*chis)
    sigma = chia**2*(81.0/16.0 - 20.0*eta) + 81.0/8.0*chia*chis*delta + chis**2*(81.0/16.0 - eta/4.0)
    eps = delta*chia*(502429.0/16128.0 - 907.0/192.0*eta) + chis*(5.0/48.0*eta2 - 73921.0/2016.0*eta + 502429.0/16128.0)

    return 1.0 + v2*(11.0/8.0*eta + 743.0/672.0) + v3*(be/2.0 - 2.0*np.pi) + v4*(1379.0/1152.0*eta2 + 18913.0/16128.0*eta + 7266251.0/8128512.0 - sigma/2.0)  \
           + v5*(57.0/16.0*np.pi*eta - 4757.0*np.pi/1344.0 + eps)  \
           + v6*(856.0/105.0*EulerGamma + 67999.0/82944.0*eta3 - 1041557.0/258048.0*eta2 - 451.0/96.0*np.pi**2*eta + 10.0*np.pi**2/3.0  \
                 + 3526813753.0/27869184.0*eta - 29342493702821.0/500716339200.0 + 856.0/105.0*np.log(4.0*v))  \
           + v7*(- 1349.0/24192.0*eta2 - 72221.0/24192.0*eta - 5111593.0/2709504.0)*np.pi

# combine the modulus and the phase of h(f)
def hf3hPN(f, M, eta, s1x=0.0, s1y=0.0, s1z=0.0, s2x=0.0, s2y=0.0, s2z=0.0, Lam=0.0, dLam=0.0,  Deff=1.0):

    pre = 3.6686934875530996e-19     # (GN*Msun/c^3)^(5/6)/Hz^(7/6)*c/Mpc/sec
    Mchirp = M*eta**0.6
    A0 = Mchirp**(5.0/6.0)/(f + 1e-100)**(7.0/6.0)/Deff/np.pi**(2.0/3.0)*np.sqrt(5.0/24.0)

    Phi = Phif3hPN(f, M, eta, s1x=s1x, s1y=s1y, s1z=s1z, s2x=s2x, s2y=s2y, s2z=s2z, Lam=Lam, dLam=dLam)
    A = Af3hPN(f, M, eta, s1x=s1x, s1y=s1y, s1z=s1z, s2x=s2x, s2y=s2y, s2z=s2z, Lam=Lam, dLam=dLam)

    # note the convention for the sign in front of the phase
    return pre*A0*A*np.exp(-1.0j*Phi)
