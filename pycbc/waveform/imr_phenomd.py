#!/usr/bin/env python

#import pyximport; pyximport.install()
import numpy as np
import numexpr as ne
import math
from scipy.interpolate import CubicSpline
from qnm_tab_data import qnmdata
from matplotlib import pyplot as plt

# define constants
mtsun = 4.925491025543575903411922162094833998e-6
mrsun = 1.476625061404649406193430731479084713e3
mpc_to_m = 3.086e22
lal_pi = 3.141592653589793238462643383279502884
lal_pi_4 = 0.785398163397448309615660845819875721
lal_gamma = 0.577215664901532860606512090082402431
pi_m_sixth = 0.8263074871107581108331125856317241299
amp_fjoin_ins = 0.014
phi_fjoin_ins = 0.018
f_cut = 0.2
powers_of_pi = {i/3.: lal_pi**(i/3.) for i in range(-5, 6)}

def generate(f, mass1=36., mass2=32., chi1=0., chi2=0.,
             distance=1., fref=20.):
    M = mass1 + mass2
    eta = (mass1*mass2)/(M*M)
    mf = M * mtsun * f
    mfref = M * mtsun * fref
    dl_meters = distance * mpc_to_m
    amp0 = 2. * np.sqrt(5./(64.*lal_pi)) * M * mrsun * M * mtsun / dl_meters
    # calculate factors
    factors = compute_factors(mf, mfref, fref, mass1, mass2, eta, chi1, chi2)
    # compute amplitude series
    amp = amp_series(mf, factors)
    # compute phase series
    phi = phase_series(mf, factors)
    exp_phi = ne.evaluate('cos(phi) - 1.0j*sin(phi)')
    #exp_phi = ne.evaluate('exp(-1.0j*phi)')
    # construct template from amplitude and phase
    template = amp0 * amp * exp_phi
    return template

def amp_series(f, fac):
    pmf = fac['powers_of_mf']
    int_start = np.sum(f < amp_fjoin_ins)
    mrd_start = np.sum(f < fac['fmaxcalc'])
    #print("transition indices: {} {}".format(int_start, mrd_start))
    amp_prefac = fac['amp0'] * pmf[-7./6]
    # inspiral
    amp_ins_ansatz = 1. + pmf[2./3][:int_start] * fac['apf23'] \
                     + pmf[4./3][:int_start] * fac['apf43'] \
                     + pmf[5./3][:int_start] * fac['apf53'] \
                     + pmf[7./3][:int_start] * fac['apf73'] \
                     + pmf[8./3][:int_start] * fac['apf83'] \
                     + pmf[1.][:int_start]*(fac['apf1'] + pmf[1.][:int_start] \
                     * fac['apf2'] + pmf[2.][:int_start] * fac['apf3'])
    # intermediate
    amp_int_ansatz = fac['delta0']+pmf[1.][int_start:mrd_start]*fac['delta1'] \
                     +pmf[2.][int_start:mrd_start] \
                     *(fac['delta2']+pmf[1.][int_start:mrd_start] \
                       *fac['delta3']+pmf[2.][int_start:mrd_start] \
                       *fac['delta4'])
    # merger-ringdown
    amp_mrd_ansatz = ampmrd(f[mrd_start:], fac)
    return amp_prefac * np.concatenate([amp_ins_ansatz, amp_int_ansatz,
                                        amp_mrd_ansatz])

def singlefreq_ampins(f, powers, fac):
    a = 1. + powers[2./3] * fac['apf23'] + powers[4./3] * fac['apf43'] \
        + powers[5./3] * fac['apf53'] + powers[7./3] * fac['apf73'] \
        + powers[8./3] * fac['apf83'] + powers[1.] \
        * (fac['apf1'] + powers[1.] * fac['apf2'] + powers[2.] * fac['apf3'])
    return a

def singlefreq_dampins(f, powers, fac):
    da = ((-969 + 1804*fac['eta'])*powers_of_pi[2./3])/(1008.*powers[1./3]) \
         + ((fac['chi1']*(81*fac['setaplus1'] - 44*fac['eta']) + fac['chi2']*(81 - 81*fac['seta'] - 44*fac['eta']))*np.pi)/48. \
         + ((-27312085 - 10287648*fac['chi22'] - 10287648*fac['chi12']*fac['setaplus1'] \
         + 10287648*fac['chi22']*fac['seta'] + 24*(-1975055 + 857304*fac['chi12'] - 994896*fac['chi1']*fac['chi2'] + 857304*fac['chi22'])*fac['eta'] \
         + 35371056*fac['eta2'])*powers[1./3]*powers_of_pi[4./3])/6.096384e6 \
         + (5*powers[2./3]*powers_of_pi[5./3]*(fac['chi2']*(-285197*(-1 + fac['seta']) \
         + 4*(-91902 + 1579*fac['seta'])*fac['eta'] - 35632*fac['eta2']) + fac['chi1']*(285197*fac['setaplus1'] \
         - 4*(91902 + 1579*fac['seta'])*fac['eta'] - 35632*fac['eta2']) + 42840*(-1 + 4*fac['eta'])*np.pi))/96768. \
         - (f*fac['pi2']*(-336*(-3248849057.0 + 2943675504*fac['chi12'] - 3339284256*fac['chi1']*fac['chi2'] + 2943675504*fac['chi22'])*fac['eta2'] - 324322727232*fac['eta3'] \
         - 7*(-177520268561 + 107414046432*fac['chi22'] + 107414046432*fac['chi12']*fac['setaplus1'] - 107414046432*fac['chi22']*fac['seta'] \
         + 11087290368*(fac['chi1'] + fac['chi2'] + fac['chi1']*fac['seta'] - fac['chi2']*fac['seta'])*np.pi) \
         + 12*fac['eta']*(-545384828789.0 - 176491177632*fac['chi1']*fac['chi2'] + 202603761360*fac['chi22'] + 77616*fac['chi12']*(2610335 + 995766*fac['seta']) \
         - 77287373856*fac['chi22']*fac['seta'] + 5841690624*(fac['chi1'] + fac['chi2'])*np.pi + 21384760320*fac['pi2'])))/3.0042980352e10 \
         + (7.0/3.0)*powers[4./3]*fac['rho1'] + (8.0/3.0)*powers[5./3]*fac['rho2'] + 3.*powers[2.]*fac['rho3']
    return da

def ampmrd(f, fac):
    fminfrd = f - fac['frd']
    fdmgamma3 = fac['fdm'] * fac['gamma3']
    gamma2 = fac['gamma2']
    exp_factor = ne.evaluate('exp(-fminfrd*gamma2/fdmgamma3)')
    #a = np.exp(-fminfrd*fac['gamma2']/fdmgamma3) \
    a = exp_factor \
        *fdmgamma3*fac['gamma1']/(pow_2_of(fminfrd) + pow_2_of(fdmgamma3))
    return a

def phase_series(f, fac):
    pmf = fac['powers_of_mf']
    int_start = np.sum(f < phi_fjoin_ins)
    mrd_start = np.sum(f < fac['fmrdjoin'])
    # inspiral
    v = pmf[1./3][:int_start] * powers_of_pi[1./3]
    logv = np.log(v)
    ins_p = np.ones(len(v)) * fac['init_phasing']
    ins_p += fac['ppf23'] * pmf[2./3][:int_start]
    ins_p += fac['ppf13'] * pmf[1./3][:int_start]
    ins_p += fac['ppf13wlogv'] * logv * pmf[1./3][:int_start]
    ins_p += fac['ppflogv'] * logv
    ins_p += fac['ppfm13'] * pmf[-1./3][:int_start]
    ins_p += fac['ppfm23'] * pmf[-2./3][:int_start]
    ins_p += fac['ppfm1'] * pmf[-1.][:int_start]
    ins_p += fac['ppfm43'] / pmf[4./3][:int_start]
    ins_p += fac['ppfm53'] * pmf[-5./3][:int_start]
    ins_p += (fac['ppf1'] * pmf[1.][:int_start] + fac['ppf43']
                * pmf[4./3][:int_start] + fac['ppf53'] * pmf[5./3][:int_start]
                + fac['ppf2'] * pmf[2.][:int_start]) * fac['etainv']
    # intermediate
    int_p = fac['beta1'] * pmf[1.][int_start:mrd_start]
    int_p += -fac['beta3'] / (3. * pmf[3.][int_start:mrd_start])
    int_p += fac['beta2'] * np.log(pmf[1.][int_start:mrd_start])
    int_p *= fac['etainv']
    int_p += fac['c1int'] + fac['c2int'] * pmf[1.][int_start:mrd_start]
    # merger-ringdown
    rholm = 1.0
    taulm = 1.0
    mrd_p = -(fac['alpha2'] * pmf[-1.][mrd_start:])
    mrd_p += 4./3 * (fac['alpha3'] * pmf[3./4][mrd_start:])
    mrd_p += fac['alpha1'] * pmf[1.][mrd_start:]
    mrd_p += fac['alpha4'] * rholm \
             * np.arctan((pmf[1.][mrd_start:] - fac['alpha5'] \
             * fac['frd']) / (rholm * fac['fdm'] * taulm))
    mrd_p *= fac['etainv']
    mrd_p += fac['c1mrd'] + fac['c2mrd'] * pmf[1.][mrd_start:]

    return np.concatenate([ins_p, int_p, mrd_p]) - fac['phi_precalc'] -fac['t0fac']

def singlefreq_phase(f, powers, fac):
    if f < phi_fjoin_ins:  # inspiral
        v = powers[1./3] * powers_of_pi[1./3]
        logv = np.log(v)
        phi = fac['init_phasing']
        phi += fac['ppf23'] * powers[2./3]
        phi += fac['ppf13'] * powers[1./3]
        phi += fac['ppf13wlogv'] * logv * powers[1./3]
        phi += fac['ppflogv'] * logv
        phi += fac['ppfm13'] * powers[-1./3]
        phi += fac['ppfm23'] * powers[-2./3]
        phi += fac['ppfm1'] * powers[-1.]
        phi += fac['ppfm43'] / powers[4./3]
        phi += fac['ppfm53'] * powers[-5./3]
        phi += (fac['ppf1'] * powers[1.] + fac['ppf43']
                    * powers[4./3] + fac['ppf53'] * powers[5./3]
                    + fac['ppf2'] * powers[2.]) * fac['etainv']
    elif f < 0.5 * fac['frd']:  # intermediate
        phi = fac['beta1'] * powers[1.]
        phi += -fac['beta3'] / (3. * powers[3.])
        phi += fac['beta2'] * np.log(powers[1.])
        phi *= fac['etainv']
        phi += fac['c1int'] + fac['c2int'] * powers[1.]
    else:  # merger-ringdown
        rholm = 1.0
        taulm = 1.0
        phi = -(fac['alpha2'] * powers[-1.])
        phi += 4./3 * (fac['alpha3'] * powers[3./4])
        phi += fac['alpha1'] * powers[1.]
        phi += fac['alpha4'] * rholm \
               * np.arctan((powers[1.] - fac['alpha5'] \
               * fac['frd']) / (rholm * fac['fdm'] * taulm))
        phi *= fac['etainv']
        phi += fac['c1mrd'] + fac['c2mrd'] * powers[1.]
    return phi

def compute_factors(mf, mfref, fref, m1_msun, m2_msun, eta, chi1, chi2):
    fac = {'eta': eta, 'chi1': chi1, 'chi2': chi2,
           'chi12': chi1*chi1, 'chi22': chi2*chi2, 'eta2': eta*eta,
           'seta': np.sqrt(1.-4.*eta), 'etainv': 1./eta}
    fac['eta3'] = fac['eta2'] * eta
    fac['amp0'] = np.sqrt((2. * eta) / 3.) * pi_m_sixth
    fac['setaplus1'] = fac['seta'] + 1.
    fac['q'] = 0.5 * (1. + fac['seta'] - 2. * eta) * fac['etainv']
    fac['m1'] = 0.5 * (1. + fac['seta'])
    fac['m2'] = 0.5 * (1. - fac['seta'])
    fac['m12'] = fac['m1']*fac['m1']
    fac['m22'] = fac['m2']*fac['m2']

    #labels = ['eta', 'chi1', 'chi2', 'eta2', 'eta3', 'chi12', 'chi22',
    #          'seta', 'setaplus1', 'etainv', 'q', 'm1', 'm2', 'm12', 'm22',
    #          'amp0', 'erad']
    #vals = factorcalc(eta, chi1, chi2, np.zeros(len(labels)))
    #fac = {l: v for l, v in zip(labels, vals)}
    fac['mf'] = mf
    fac['mfref'] = mfref
    fac['fref'] = fref
    fac['pi'] = lal_pi
    fac['pi2'] = pow_2_of(lal_pi)
    fac['chi'] = chi_pn(fac['seta'], eta, chi1, chi2)
    fac['xi'] = -1. + fac['chi']
    fac['finspin'] = finspin(fac['m12'], fac['m22'], eta, fac['eta2'],
                             fac['eta3'], chi1, chi2)
    s_erad = (fac['m12'] * chi1 + fac['m22'] * chi2) / (fac['m12'] + fac['m22'])
    fac['erad'] = (eta*(0.055974469826360077 + 0.5809510763115132*eta \
                        - 0.9606726679372312*fac['eta2'] + 3.352411249771192*fac['eta3']) \
                   * (1. + (-0.0030302335878845507 - 2.0066110851351073*eta \
                            + 7.7050567802399215*fac['eta2'])*s_erad)) \
                   / (1. + (-0.6714403054720589 - 1.4756929437702908*eta \
                            + 7.304676214885011*fac['eta2'])*s_erad)
    fac['frd'] = fring(fac['finspin'], fac['erad'])
    fac['fdm'] = fdamp(fac['finspin'], fac['erad'])

    # amplitude factors
    fac.update(compute_gammas(eta, fac['eta2'], fac['xi']))
    fac['fmaxcalc'] = fmaxcalc(fac)
    fac.update(compute_rhos(eta, fac['eta2'], fac['xi']))
    fac.update(compute_amp_prefactors(fac))
    fac.update(compute_deltas(fac))

    # phase factors
    fac.update(compute_phase_fit_factors(fac['eta'], fac['eta2'], fac['xi']))
    fac['pn'] = compute_pn_phasing(m1_msun, m2_msun, eta, fac['eta2'],
                                   fac['eta3'], chi1, chi2,
                                   fac['pi2'])
    fac.update(compute_phase_prefactors(fac))
    # compute phi at fref
    mfref_third = ne.evaluate('mfref ** (1./3)')
    mfref_pow = {1./3: mfref_third}
    mfref_pow[2./3] = mfref_pow[1./3] * mfref_pow[1./3]
    mfref_pow[3./3] = mfref
    mfref_pow[4./3] = mfref * mfref_pow[1./3]
    mfref_pow[5./3] = mfref * mfref_pow[2./3]
    mfref_pow[6./3] = mfref * mfref
    mfref_pow[9./3] = mfref * mfref_pow[6./3]
    mfref_pow[3./4] = np.sqrt(mfref * np.sqrt(mfref))
    mfref_pow.update({-i/3.: 1./mfref_pow[i/3.] for i in range(1, 6)})
    fac['phi_precalc'] = singlefreq_phase(mfref, mfref_pow, fac)

    # powers of mf
    #mf_third = useful_powers(mf, np.zeros(len(mf)))  # 1/3
    #mf_third = np.cbrt(mf)
    mf_third = ne.evaluate('mf ** (1./3)')
    fac['powers_of_mf'] = {1./3: mf_third}
    fac['powers_of_mf'][2./3] = mf_third * mf_third
    fac['powers_of_mf'][3./3] = mf
    fac['powers_of_mf'][4./3] = mf * mf_third
    fac['powers_of_mf'][5./3] = mf * fac['powers_of_mf'][2./3]
    fac['powers_of_mf'][6./3] = mf * mf
    fac['powers_of_mf'][7./3] = mf * fac['powers_of_mf'][4./3]
    fac['powers_of_mf'][8./3] = mf * fac['powers_of_mf'][5./3]
    fac['powers_of_mf'][9./3] = mf * fac['powers_of_mf'][6./3]
    fac['powers_of_mf'][12./3] = mf * fac['powers_of_mf'][9./3]
    fac['powers_of_mf'][-7./6] = 1./mf * 1./np.sqrt(mf_third)
    fac['powers_of_mf'][3./4] = np.sqrt(mf * np.sqrt(mf))
    fac['powers_of_mf'].update({-i/3.: 1./fac['powers_of_mf'][i/3.]
                                for i in range(1, 6)})
    #print("final spin is {}".format(fac['finspin']))
    #print("dimensionless m1 is {}".format(fac['m1']))
    #print("dimensionless m2 is {}".format(fac['m2']))
    #print("f-ringdown is {}".format(fac['frd']))
    #print("f-damp is {}".format(fac['fdm']))
    #print("fmaxcalc is {}".format(fac['fmaxcalc']))
    return fac

def compute_amp_prefactors(fac):
    eta = fac['eta']
    chi1 = fac['chi1']
    chi2 = fac['chi2']
    chi12 = fac['chi12']
    chi22 = fac['chi22']
    eta2 = fac['eta2']
    eta3 = fac['eta3']
    pi = fac['pi']
    pi2 = fac['pi2']
    seta = fac['seta']
    setaplus1 = fac['setaplus1']
    # build prefactors dict
    pf = {}
    pf['apf23'] = ((-969 + 1804*eta)*powers_of_pi[2./3])/672.
    pf['apf1'] = ((chi1*(81*setaplus1 - 44*eta) + chi2*(81 - 81*seta - 44*eta))*pi)/48.
    pf['apf43'] = ((-27312085.0 - 10287648*chi22 - 10287648*chi12*setaplus1 + 10287648*chi22*seta \
                  + 24*(-1975055 + 857304*chi12 - 994896*chi1*chi2 + 857304*chi22)*eta \
                  + 35371056*eta2) * powers_of_pi[4./3]) / 8.128512e6
    pf['apf53'] = (powers_of_pi[5./3] * (chi2*(-285197*(-1 + seta) + 4*(-91902 + 1579*seta)*eta - 35632*eta2) \
                  + chi1*(285197*setaplus1 - 4*(91902 + 1579*seta)*eta - 35632*eta2) \
                  + 42840*(-1.0 + 4*eta)*pi)) / 32256.
    pf['apf2'] = - (pi2*(-336*(-3248849057.0 + 2943675504*chi12 - 3339284256*chi1*chi2 + 2943675504*chi22)*eta2 \
                 - 324322727232*eta3 - 7*(-177520268561 + 107414046432*chi22 + 107414046432*chi12*setaplus1 \
                 - 107414046432*chi22*seta + 11087290368*(chi1 + chi2 + chi1*seta - chi2*seta)*pi) \
                 + 12*eta*(-545384828789 - 176491177632*chi1*chi2 + 202603761360*chi22 \
                 + 77616*chi12*(2610335 + 995766*seta) - 77287373856*chi22*seta \
                 + 5841690624*(chi1 + chi2)*pi + 21384760320*pi2)))/6.0085960704e10
    pf['apf73'] = fac['rho1']
    pf['apf83'] = fac['rho2']
    pf['apf3'] = fac['rho3']
    return pf

def compute_phase_fit_factors(eta, eta2, xi):
    sigma1 = 2096.551999295543 + 1463.7493168261553*eta \
             + (1312.5493286098522 + 18307.330017082117*eta - 43534.1440746107*eta2 \
             + (-833.2889543511114 + 32047.31997183187*eta - 108609.45037520859*eta2)*xi \
             + (452.25136398112204 + 8353.439546391714*eta - 44531.3250037322*eta2)*xi*xi)*xi
    #sigma1 *= 1.
    sigma2 = -10114.056472621156 - 44631.01109458185*eta \
             + (-6541.308761668722 - 266959.23419307504*eta + 686328.3229317984*eta2 \
             + (3405.6372187679685 - 437507.7208209015*eta + 1.6318171307344697e6*eta2)*xi \
             + (-7462.648563007646 - 114585.25177153319*eta + 674402.4689098676*eta2)*xi*xi)*xi
    #sigma2 *= 1.
    sigma3 = 22933.658273436497 + 230960.00814979506*eta \
             + (14961.083974183695 + 1.1940181342318142e6*eta - 3.1042239693052764e6*eta2 \
             + (-3038.166617199259 + 1.8720322849093592e6*eta - 7.309145012085539e6*eta2)*xi \
             + (42738.22871475411 + 467502.018616601*eta - 3.064853498512499e6*eta2)*xi*xi)*xi
    #sigma3 *= 1.
    sigma4 = -14621.71522218357 - 377812.8579387104*eta \
             + (-9608.682631509726 - 1.7108925257214056e6*eta + 4.332924601416521e6*eta2 \
             + (-22366.683262266528 - 2.5019716386377467e6*eta + 1.0274495902259542e7*eta2)*xi \
             + (-85360.30079034246 - 570025.3441737515*eta + 4.396844346849777e6*eta2)*xi*xi)*xi
    #sigma4 *= 1.
    beta1 = 97.89747327985583 - 42.659730877489224*eta \
            + (153.48421037904913 - 1417.0620760768954*eta + 2752.8614143665027*eta2 \
            + (138.7406469558649 - 1433.6585075135881*eta + 2857.7418952430758*eta2)*xi \
            + (41.025109467376126 - 423.680737974639*eta + 850.3594335657173*eta2)*xi*xi)*xi
    #beta1 *= 1.
    beta2 = -3.282701958759534 - 9.051384468245866*eta \
            + (-12.415449742258042 + 55.4716447709787*eta - 106.05109938966335*eta2 \
            + (-11.953044553690658 + 76.80704618365418*eta - 155.33172948098394*eta2)*xi \
            + (-3.4129261592393263 + 25.572377569952536*eta - 54.408036707740465*eta2)*xi*xi)*xi
    #beta2 *= 1.
    beta3 = -0.000025156429818799565 + 0.000019750256942201327*eta \
            + (-0.000018370671469295915 + 0.000021886317041311973*eta + 0.00008250240316860033*eta2 \
            + (7.157371250566708e-6 - 0.000055780000112270685*eta + 0.00019142082884072178*eta2)*xi \
            + (5.447166261464217e-6 - 0.00003220610095021982*eta + 0.00007974016714984341*eta2)*xi*xi)*xi
    #beta3 *= 1.
    alpha1 = 43.31514709695348 + 638.6332679188081*eta \
             + (-32.85768747216059 + 2415.8938269370315*eta - 5766.875169379177*eta2 \
             + (-61.85459307173841 + 2953.967762459948*eta - 8986.29057591497*eta2)*xi \
             + (-21.571435779762044 + 981.2158224673428*eta - 3239.5664895930286*eta2)*xi*xi)*xi
    #alpha1 *= 1.
    alpha2 = -0.07020209449091723 - 0.16269798450687084*eta \
             + (-0.1872514685185499 + 1.138313650449945*eta - 2.8334196304430046*eta2 \
             + (-0.17137955686840617 + 1.7197549338119527*eta - 4.539717148261272*eta2)*xi \
             + (-0.049983437357548705 + 0.6062072055948309*eta - 1.682769616644546*eta2)*xi*xi)*xi
    #alpha2 *= 1.
    alpha3 = 9.5988072383479 - 397.05438595557433*eta \
             + (16.202126189517813 - 1574.8286986717037*eta + 3600.3410843831093*eta2 \
             + (27.092429659075467 - 1786.482357315139*eta + 5152.919378666511*eta2)*xi \
             + (11.175710130033895 - 577.7999423177481*eta + 1808.730762932043*eta2)*xi*xi)*xi
    #alpha3 *= 1.
    alpha4 = -0.02989487384493607 + 1.4022106448583738*eta \
             + (-0.07356049468633846 + 0.8337006542278661*eta + 0.2240008282397391*eta2 \
             + (-0.055202870001177226 + 0.5667186343606578*eta + 0.7186931973380503*eta2)*xi \
             + (-0.015507437354325743 + 0.15750322779277187*eta + 0.21076815715176228*eta2)*xi*xi)*xi
    #alpha4 *= 1.
    alpha5 = 0.9974408278363099 - 0.007884449714907203*eta \
             + (-0.059046901195591035 + 1.3958712396764088*eta - 4.516631601676276*eta2 \
             + (-0.05585343136869692 + 1.7516580039343603*eta - 5.990208965347804*eta2)*xi \
             + (-0.017945336522161195 + 0.5965097794825992*eta - 2.0608879367971804*eta2)*xi*xi)*xi
    #alpha5 *= 1.
    ff = {'sigma1': sigma1, 'sigma2': sigma2, 'sigma3': sigma3,
          'sigma4': sigma4, 'beta1': beta1, 'beta2': beta2,
          'beta3': beta3, 'alpha1': alpha1, 'alpha2': alpha2,
          'alpha3': alpha3, 'alpha4': alpha4, 'alpha5': alpha5}
    return ff

def compute_phase_prefactors(fac, rholm=1.0, taulm=1.0):
    pn = fac['pn']
    sigma1 = fac['sigma1']
    sigma2 = fac['sigma2']
    sigma3 = fac['sigma3']
    sigma4 = fac['sigma4']
    beta1 = fac['beta1']
    beta2 = fac['beta2']
    beta3 = fac['beta3']
    alpha1 = fac['alpha1']
    alpha2 = fac['alpha2']
    alpha3 = fac['alpha3']
    alpha4 = fac['alpha4']
    alpha5 = fac['alpha5']
    frd = fac['frd']
    fdm = fac['fdm']
    mfref = fac['mfref']
    fmaxcalc = fac['fmaxcalc']
    etainv = fac['etainv']
    # PN phasing series
    pf = {'init_phasing': pn['v5'] - lal_pi_4}
    pf['ppf23'] = pn['v7'] * powers_of_pi[2./3]
    pf['ppf13'] = pn['v6'] * powers_of_pi[1./3]
    pf['ppf13wlogv'] = pn['vlogv6'] * powers_of_pi[1./3]
    pf['ppflogv'] = pn['vlogv5']
    pf['ppfm13'] = pn['v4'] * powers_of_pi[-1./3]
    pf['ppfm23'] = pn['v3'] * powers_of_pi[-2./3]
    pf['ppfm1'] = pn['v2'] * powers_of_pi[-1.]
    pf['ppfm43'] = pn['v1'] / powers_of_pi[4./3]
    pf['ppfm53'] = pn['v0'] * powers_of_pi[-5./3]
    # higher order terms that were calibrated for PhenomD
    pf['ppf1'] = sigma1
    pf['ppf43'] = sigma2 * 0.75
    pf['ppf53'] = sigma3 * 0.6
    pf['ppf2'] = sigma4 * 0.5
    # connection coefficients
    pf['fmrdjoin'] = 0.5 * frd
    vdphi = ne.evaluate('(lal_pi * phi_fjoin_ins) ** (1./3)')
    log_vdphi = np.log(vdphi)
    vdphi2 = pow_2_of(vdphi)
    vdphi3 = pow_3_of(vdphi)
    vdphi4 = pow_4_of(vdphi)
    vdphi5 = vdphi4 * vdphi
    vdphi6 = vdphi5 * vdphi
    vdphi7 = vdphi6 * vdphi
    vdphi8 = vdphi7 * vdphi
    dphi_ins = 2. * pn['v7'] * vdphi7
    dphi_ins += (pn['v6'] + pn['vlogv6'] * (1.0 + log_vdphi)) * vdphi6
    dphi_ins += pn['vlogv5'] * vdphi5
    dphi_ins += -1.0 * pn['v4'] * vdphi4
    dphi_ins += -2.0 * pn['v3'] * vdphi3
    dphi_ins += -3.0 * pn['v2'] * vdphi2
    dphi_ins += -4.0 * pn['v1'] * vdphi
    dphi_ins += -5.0 * pn['v0']
    dphi_ins /= vdphi8 * 3.0
    dphi_ins *= lal_pi
    dphi_ins += (sigma1 + sigma2 * vdphi * powers_of_pi[-1./3]
                 + sigma3 * vdphi2 * powers_of_pi[-2./3]
                 + (sigma4*powers_of_pi[-1.]) * vdphi3) * etainv
    dphi_int = (beta1 + beta3/pow_4_of(phi_fjoin_ins) + beta2/phi_fjoin_ins) * etainv
    pf['c2int'] = dphi_ins - dphi_int
    fjoin_pow_third = ne.evaluate('phi_fjoin_ins ** (1./3)')
    fjoin_pow = {1./3: fjoin_pow_third}
    fjoin_pow[2./3] = fjoin_pow[1./3] * fjoin_pow[1./3]
    fjoin_pow[3./3] = phi_fjoin_ins
    fjoin_pow[4./3] = phi_fjoin_ins * fjoin_pow[1./3]
    fjoin_pow[5./3] = phi_fjoin_ins * fjoin_pow[2./3]
    fjoin_pow[6./3] = phi_fjoin_ins * phi_fjoin_ins
    fjoin_pow.update({-i/3.: 1./fjoin_pow[i/3.] for i in range(1, 6)})
    v = fjoin_pow[1./3] * powers_of_pi[1./3]
    logv = np.log(v)
    c1int = pf['init_phasing']
    c1int += pf['ppf23'] * fjoin_pow[2./3]
    c1int += pf['ppf13'] * fjoin_pow[1./3]
    c1int += pf['ppf13wlogv'] * logv * fjoin_pow[1./3]
    c1int += pf['ppflogv'] * logv
    c1int += pf['ppfm13'] * fjoin_pow[-1./3]
    c1int += pf['ppfm23'] * fjoin_pow[-2./3]
    c1int += pf['ppfm1'] * fjoin_pow[-1.]
    c1int += pf['ppfm43'] / fjoin_pow[4./3]
    c1int += pf['ppfm53'] * fjoin_pow[-5./3]
    c1int += (pf['ppf1'] * fjoin_pow[1.] + pf['ppf43'] * fjoin_pow[4./3]
              + pf['ppf53'] * fjoin_pow[5./3] + pf['ppf2'] * fjoin_pow[2.]) * etainv
    c1int += -etainv * (beta1*phi_fjoin_ins - beta3/(3.*pow_3_of(phi_fjoin_ins)) \
             + beta2 * np.log(phi_fjoin_ins))
    c1int += -pf['c2int'] * phi_fjoin_ins
    pf['c1int'] = c1int
    phi_int_fmrd = beta1 * pf['fmrdjoin'] - beta3/(3.*pow_3_of(pf['fmrdjoin'])) \
                   + beta2 * np.log(pf['fmrdjoin'])
    phi_int_temp = etainv * phi_int_fmrd + pf['c1int'] + pf['c2int'] * pf['fmrdjoin']
    dphi_int_temp = pf['c2int'] + (beta1 + beta3/pow_4_of(pf['fmrdjoin']) \
                    + beta2/pf['fmrdjoin']) * etainv
    dphi_mrd_val = (alpha1 + alpha2/pow_2_of(pf['fmrdjoin']) \
                   + alpha3/(pf['fmrdjoin'] **(0.25)) + alpha4 \
                   /(fdm * taulm * (1. + pow_2_of(pf['fmrdjoin'] \
                   - alpha5 * frd)/(pow_2_of(fdm * taulm * rholm))))) * etainv
    pf['c2mrd'] = dphi_int_temp - dphi_mrd_val
    phimrd_fmrd = -alpha2/pf['fmrdjoin']
    phimrd_fmrd += 4./3 * alpha3 * math.sqrt(pf['fmrdjoin'] * math.sqrt(pf['fmrdjoin']))
    phimrd_fmrd += alpha1 * pf['fmrdjoin']
    phimrd_fmrd += alpha4 * rholm * np.arctan((pf['fmrdjoin'] - alpha5*frd) \
                   /(rholm * fdm * taulm))
    pf['c1mrd'] = phi_int_temp - etainv * phimrd_fmrd - pf['c2mrd'] * pf['fmrdjoin']
    # compute time shift to t0
    t0 = (alpha1 + alpha2/pow_2_of(fmaxcalc) \
               + alpha3/(fmaxcalc **(0.25)) + alpha4 \
               /(fdm * taulm * (1 + pow_2_of(fmaxcalc \
               - alpha5 * frd)/(pow_2_of(fdm * taulm * rholm))))) * etainv
    pf['t0fac'] = t0 * (fac['mf'] - fac['mfref'])
    return pf

def compute_pn_phasing(m1, m2, eta, eta2, eta3, chi1z, chi2z,
                       pi2):
    mtot = m1 + m2
    m1M = m1 / mtot
    m2M = m2 / mtot
    chi1dotchi2 = chi1z * chi2z
    chi1sq = chi1z**2.
    chi2sq = chi2z**2.
    pfaN = np.float128(3.) / (np.float128(128.) * eta)
    pn = {'v0': np.float128(1.)}
    pn['v1'] = np.float128(0.)
    pn['v2'] = 5. * (74.3 / 8.4 + 11. * eta) / 9.
    pn['v3'] = -16. * lal_pi
    pn['v4'] = 5. * (3058.673 / 7.056 + 5429. / 7. * eta
                     + 617. * eta * eta) / 72.
    pn['v5'] = 5. / 9. * (772.9/8.4 - 13. * eta) * lal_pi
    pn['vlogv5'] = 5. / 3 * (772.9/8.4 - 13. * eta) * lal_pi
    pn['vlogv6'] = -684.8 / 2.1
    pn['v6'] = 11583.231236531/4.694215680 - 640./3.*pi2 - 684.8/2.1*lal_gamma \
               + eta*(-15737.765635/3.048192 + 225.5/1.2*pi2) \
               + eta2*76.055/1.728 - eta3*127.825/1.296 \
               + pn['vlogv6']*np.log(4.)
    pn['v7'] = lal_pi*(770.96675/2.54016 + 378.515/1.512*eta - 740.45/7.56*eta2)
    # I think dQuadMonds should be zero for non-testingGR circumstances
    qm_def1 = qm_def2 = 1.
    # add 3.5PN spin term
    so35 = lambda x: x*(-17097.8035/4.8384+eta*28764.25/6.72+eta2*47.35/1.44 \
                     + x*(-7189.233785/1.524096+eta*458.555/3.024-eta2*534.5/7.2))
    pn['v7'] += so35(m1M) * chi1z + so35(m2M) * chi2z
    # add 3PN spin term WITHOUT spin-spin parts (LALsuite subtracts these off)
    so3 = lambda x: lal_pi * x * (1490./3 + x * 260.)
    pn['v6'] += so3(m1M) * chi1z + so3(m2M) * chi2z
    # add 2.5PN spin term
    so25 = lambda x: -x*(1391.5/8.4-x*(1.-x)*10./3.+ x*(1276./8.1+x*(1.-x)*170./9.))
    so25_term = so25(m1M) * chi1z + so25(m2M) * chi2z
    pn['v5'] += so25_term
    pn['vlogv5'] += 3. * (so25_term)
    # add 2PN spin terms
    pn['v4'] += 247./4.8*eta*chi1dotchi2 -721./4.8*eta * chi1z * chi2z
    pn['v4'] += (-720./9.6*m1M*m1M*qm_def1 + 1./9.6*m1M*m1M)*chi1z*chi1z
    pn['v4'] += (-720./9.6*m2M*m2M*qm_def2 + 1./9.6*m2M*m2M)*chi2z*chi2z
    pn['v4'] += (240./9.6*m1M*m1M*qm_def1 + -7./9.6*m1M*m1M) * chi1sq
    pn['v4'] += (240./9.6*m2M*m2M*qm_def2 + -7./9.6*m2M*m2M) * chi2sq
    # add 1.5PN spin term
    so15 = lambda x: x*(25.+38./3.*x)
    pn['v3'] += so15(m1M) * chi1z + so15(m2M) * chi2z

    for key in pn:
        pn[key] *= pfaN
    return pn

def compute_deltas(fac):
    eta = fac['eta']
    eta2 = fac['eta2']
    xi = fac['xi']
    f1 = amp_fjoin_ins  # defined at top of file
    f3 = fac['fmaxcalc']
    dfx = 0.5 * (f3 - f1)
    f2 = f1 + dfx
    # need powers of f1
    #powers_of_f1 = {i/3.: f1**(1/3.) for i in range(1, 10)}
    #f1_third = onefreq_powers(f1)
    f1_third = ne.evaluate('f1 ** (1./3)')
    powers_of_f1 = {1./3: f1_third}
    powers_of_f1[2./3] = f1_third * f1_third
    powers_of_f1[3./3] = f1
    powers_of_f1[4./3] = f1 * f1_third
    powers_of_f1[5./3] = f1 * powers_of_f1[2./3]
    powers_of_f1[6./3] = f1 * f1
    powers_of_f1[7./3] = f1 * powers_of_f1[4./3]
    powers_of_f1[8./3] = f1 * powers_of_f1[5./3]
    powers_of_f1[9./3] = f1 * powers_of_f1[6./3]
    #  v1, d1 are inspiral amp and derivative at f1
    v1 = singlefreq_ampins(f1, powers_of_f1, fac)
    d1 = singlefreq_dampins(f1, powers_of_f1, fac)
    # v3 is merger-ringdown amp at f3
    v3 = ampmrd(f3, fac)
    # d2 is merger-ringdown amp derivative at f3
    fdmgamma3 = fac['fdm'] * fac['gamma3']
    gamma2 = fac['gamma2']
    fminfrd = f3 - fac['frd']
    #expfactor = np.exp(((fminfrd)*fac['gamma2'])/(fdmgamma3))
    expfactor = ne.evaluate('exp(fminfrd*gamma2/fdmgamma3)')
    pow2pluspow2 = fminfrd*fminfrd + fdmgamma3*fdmgamma3
    d2 = ((-2*fac['fdm']*(fminfrd)*fac['gamma3']*fac['gamma1']) / pow2pluspow2 -
          (fac['gamma2']*fac['gamma1'])) / (expfactor * pow2pluspow2)
    # v2 is amplitude at f2 (in intermediate region)
    v2 = 0.8149838730507785 + 2.5747553517454658*eta \
         + (1.1610198035496786 - 2.3627771785551537*eta + 6.771038707057573*eta2 \
            + (0.7570782938606834 - 2.7256896890432474*eta + 7.1140380397149965*eta2)*xi \
            + (0.1766934149293479 - 0.7978690983168183*eta + 2.1162391502005153*eta2)*xi*xi)*xi
    # calculate convenience quantities
    f12 = f1*f1
    f13 = f12*f1
    f14 = f13*f1
    f15 = f14*f1
    f22 = f2*f2
    f23 = f22*f2
    f24 = f23*f2
    f32 = f3*f3
    f33 = f32*f3
    f34 = f33*f3
    f35 = f34*f3
    # calculate deltas
    delta0 = -((d2*f15*f22*f3 - 2*d2*f14*f23*f3 + d2*f13*f24*f3 - d2*f15*f2*f32 + d2*f14*f22*f32 \
             - d1*f13*f23*f32 + d2*f13*f23*f32 + d1*f12*f24*f32 - d2*f12*f24*f32 + d2*f14*f2*f33 \
             + 2*d1*f13*f22*f33 - 2*d2*f13*f22*f33 - d1*f12*f23*f33 + d2*f12*f23*f33 - d1*f1*f24*f33 \
             - d1*f13*f2*f34 - d1*f12*f22*f34 + 2*d1*f1*f23*f34 + d1*f12*f2*f35 - d1*f1*f22*f35 \
             + 4*f12*f23*f32*v1 - 3*f1*f24*f32*v1 - 8*f12*f22*f33*v1 + 4*f1*f23*f33*v1 + f24*f33*v1 \
             + 4*f12*f2*f34*v1 + f1*f22*f34*v1 - 2*f23*f34*v1 - 2*f1*f2*f35*v1 + f22*f35*v1 - f15*f32*v2 \
             + 3*f14*f33*v2 - 3*f13*f34*v2 + f12*f35*v2 - f15*f22*v3 + 2*f14*f23*v3 - f13*f24*v3 \
             + 2*f15*f2*f3*v3 - f14*f22*f3*v3 - 4*f13*f23*f3*v3 + 3*f12*f24*f3*v3 - 4*f14*f2*f32*v3 \
             + 8*f13*f22*f32*v3 - 4*f12*f23*f32*v3) / (pow_2_of(f1 - f2)*pow_3_of(f1 - f3)*pow_2_of(f3-f2)))
    delta1 = -((-(d2*f15*f22) + 2*d2*f14*f23 - d2*f13*f24 - d2*f14*f22*f3 + 2*d1*f13*f23*f3 \
             + 2*d2*f13*f23*f3 - 2*d1*f12*f24*f3 - d2*f12*f24*f3 + d2*f15*f32 - 3*d1*f13*f22*f32 \
             - d2*f13*f22*f32 + 2*d1*f12*f23*f32 - 2*d2*f12*f23*f32 + d1*f1*f24*f32 + 2*d2*f1*f24*f32 \
             - d2*f14*f33 + d1*f12*f22*f33 + 3*d2*f12*f22*f33 - 2*d1*f1*f23*f33 - 2*d2*f1*f23*f33 \
             + d1*f24*f33 + d1*f13*f34 + d1*f1*f22*f34 - 2*d1*f23*f34 - d1*f12*f35 + d1*f22*f35 \
             - 8*f12*f23*f3*v1 + 6*f1*f24*f3*v1 + 12*f12*f22*f32*v1 - 8*f1*f23*f32*v1 - 4*f12*f34*v1 \
             + 2*f1*f35*v1 + 2*f15*f3*v2 - 4*f14*f32*v2 + 4*f12*f34*v2 - 2*f1*f35*v2 - 2*f15*f3*v3 \
             + 8*f12*f23*f3*v3 - 6*f1*f24*f3*v3 + 4*f14*f32*v3 - 12*f12*f22*f32*v3 + 8*f1*f23*f32*v3) \
             / (pow_2_of(f1 - f2)*pow_3_of(f1 - f3)*pow_2_of(-f2 + f3)))
    delta2 = -((d2*f15*f2 - d1*f13*f23 - 3*d2*f13*f23 + d1*f12*f24 + 2*d2*f12*f24 - d2*f15*f3 \
             + d2*f14*f2*f3 - d1*f12*f23*f3 + d2*f12*f23*f3 + d1*f1*f24*f3 - d2*f1*f24*f3 - d2*f14*f32 \
             + 3*d1*f13*f2*f32 + d2*f13*f2*f32 - d1*f1*f23*f32 + d2*f1*f23*f32 - 2*d1*f24*f32 - d2*f24*f32 \
             - 2*d1*f13*f33 + 2*d2*f13*f33 - d1*f12*f2*f33 - 3*d2*f12*f2*f33 + 3*d1*f23*f33 + d2*f23*f33 \
             + d1*f12*f34 - d1*f1*f2*f34 + d1*f1*f35 - d1*f2*f35 + 4*f12*f23*v1 - 3*f1*f24*v1 + 4*f1*f23*f3*v1 \
             - 3*f24*f3*v1 - 12*f12*f2*f32*v1 + 4*f23*f32*v1 + 8*f12*f33*v1 - f1*f34*v1 - f35*v1 - f15*v2 \
             - f14*f3*v2 + 8*f13*f32*v2 - 8*f12*f33*v2 + f1*f34*v2 + f35*v2 + f15*v3 - 4*f12*f23*v3 + 3*f1*f24*v3 \
             + f14*f3*v3 - 4*f1*f23*f3*v3 + 3*f24*f3*v3 - 8*f13*f32*v3 + 12*f12*f2*f32*v3 - 4*f23*f32*v3) \
             / (pow_2_of(f1 - f2)*pow_3_of(f1 - f3)*pow_2_of(-f2 + f3)))
    delta3 = -((-2*d2*f14*f2 + d1*f13*f22 + 3*d2*f13*f22 - d1*f1*f24 - d2*f1*f24 + 2*d2*f14*f3 \
             - 2*d1*f13*f2*f3 - 2*d2*f13*f2*f3 + d1*f12*f22*f3 - d2*f12*f22*f3 + d1*f24*f3 + d2*f24*f3 \
             + d1*f13*f32 - d2*f13*f32 - 2*d1*f12*f2*f32 + 2*d2*f12*f2*f32 + d1*f1*f22*f32 - d2*f1*f22*f32 \
             + d1*f12*f33 - d2*f12*f33 + 2*d1*f1*f2*f33 + 2*d2*f1*f2*f33 - 3*d1*f22*f33 - d2*f22*f33 \
             - 2*d1*f1*f34 + 2*d1*f2*f34 - 4*f12*f22*v1 + 2*f24*v1 + 8*f12*f2*f3*v1 - 4*f1*f22*f3*v1 \
             - 4*f12*f32*v1 + 8*f1*f2*f32*v1 - 4*f22*f32*v1 - 4*f1*f33*v1 + 2*f34*v1 + 2*f14*v2 \
             - 4*f13*f3*v2 + 4*f1*f33*v2 - 2*f34*v2 - 2*f14*v3 + 4*f12*f22*v3 - 2*f24*v3 + 4*f13*f3*v3 \
             - 8*f12*f2*f3*v3 + 4*f1*f22*f3*v3 + 4*f12*f32*v3 - 8*f1*f2*f32*v3 + 4*f22*f32*v3) \
             / (pow_2_of(f1 - f2)*pow_3_of(f1 - f3)*pow_2_of(-f2 + f3)))
    delta4 = -((d2*f13*f2 - d1*f12*f22 - 2*d2*f12*f22 + d1*f1*f23 + d2*f1*f23 - d2*f13*f3 + 2*d1*f12*f2*f3 \
             + d2*f12*f2*f3 - d1*f1*f22*f3 + d2*f1*f22*f3 - d1*f23*f3 - d2*f23*f3 - d1*f12*f32 + d2*f12*f32 \
             - d1*f1*f2*f32 - 2*d2*f1*f2*f32 + 2*d1*f22*f32 + d2*f22*f32 + d1*f1*f33 - d1*f2*f33 + 3*f1*f22*v1 \
             - 2*f23*v1 - 6*f1*f2*f3*v1 + 3*f22*f3*v1 + 3*f1*f32*v1 - f33*v1 - f13*v2 + 3*f12*f3*v2 - 3*f1*f32*v2 \
             + f33*v2 + f13*v3 - 3*f1*f22*v3 + 2*f23*v3 - 3*f12*f3*v3 + 6*f1*f2*f3*v3 - 3*f22*f3*v3) \
             / (pow_2_of(f1 - f2)*pow_3_of(f1 - f3)*pow_2_of(-f2 + f3)))
    return {'delta0': delta0, 'delta1': delta1, 'delta2': delta2,
            'delta3': delta3, 'delta4': delta4}

def compute_gammas(eta, eta2, xi):
    gamma1 = 0.006927402739328343 + 0.03020474290328911*eta \
             + (0.006308024337706171 - 0.12074130661131138*eta + 0.26271598905781324*eta2 \
             + (0.0034151773647198794 - 0.10779338611188374*eta + 0.27098966966891747*eta2)*xi \
             + (0.0007374185938559283 - 0.02749621038376281*eta + 0.0733150789135702*eta2)*xi*xi)*xi
    gamma2 = 1.010344404799477 + 0.0008993122007234548*eta \
             + (0.283949116804459 - 4.049752962958005*eta + 13.207828172665366*eta2 \
             + (0.10396278486805426 - 7.025059158961947*eta + 24.784892370130475*eta2)*xi \
             + (0.03093202475605892 - 2.6924023896851663*eta + 9.609374464684983*eta2)*xi*xi)*xi
    gamma3 = 1.3081615607036106 - 0.005537729694807678*eta \
             + (-0.06782917938621007 - 0.6689834970767117*eta + 3.403147966134083*eta2 \
             + (-0.05296577374411866 - 0.9923793203111362*eta + 4.820681208409587*eta2)*xi \
             + (-0.006134139870393713 - 0.38429253308696365*eta + 1.7561754421985984*eta2)*xi*xi)*xi
    return {'gamma1': gamma1, 'gamma2': gamma2, 'gamma3': gamma3}

def compute_rhos(eta, eta2, xi):
    rho1 = 3931.8979897196696 - 17395.758706812805*eta \
           + (3132.375545898835 + 343965.86092361377*eta - 1.2162565819981997e6*eta2 \
           + (-70698.00600428853 + 1.383907177859705e6*eta - 3.9662761890979446e6*eta2)*xi \
           + (-60017.52423652596 + 803515.1181825735*eta - 2.091710365941658e6*eta2)*xi*xi)*xi
    rho2 = -40105.47653771657 + 112253.0169706701*eta \
           + (23561.696065836168 - 3.476180699403351e6*eta + 1.137593670849482e7*eta2 \
           + (754313.1127166454 - 1.308476044625268e7*eta + 3.6444584853928134e7*eta2)*xi \
           + (596226.612472288 - 7.4277901143564405e6*eta + 1.8928977514040343e7*eta2)*xi*xi)*xi
    rho3 = 83208.35471266537 - 191237.7264145924*eta \
           + (-210916.2454782992 + 8.71797508352568e6*eta - 2.6914942420669552e7*eta2 \
           + (-1.9889806527362722e6 + 3.0888029960154563e7*eta - 8.390870279256162e7*eta2)*xi \
           + (-1.4535031953446497e6 + 1.7063528990822166e7*eta - 4.2748659731120914e7*eta2)*xi*xi)*xi
    return {'rho1': rho1, 'rho2': rho2, 'rho3': rho3}

def chi_pn(seta, eta, chi1, chi2):
    chi_s = chi1 + chi2
    chi_a = chi1 - chi2
    return 0.5 * (chi_s * (1. - eta * 76. / 113.) + seta * chi_a)

def finspin(m12, m22, eta, eta2, eta3, chi1, chi2):
    s = m12 * chi1 + m22 * chi2
    fspin = eta*(3.4641016151377544 - 4.399247300629289*eta +
                 9.397292189321194*eta2 - 13.180949901606242*eta3 +
                 s*((1.0/eta - 0.0850917821418767 - 5.837029316602263*eta) +
                    (0.1014665242971878 - 2.0967746996832157*eta)*s +
                    (-1.3546806617824356 + 4.108962025369336*eta)*s*s +
                    (-0.8676969352555539 + 2.064046835273906*eta)*s*s*s))
    return fspin


def fring(finspin, erad):
    # initialize spline interpolator
    cs = CubicSpline(qnmdata['a'], qnmdata['fring'])
    return cs(finspin) / (1. - erad)

def fdamp(finspin, erad):
    # initialize spline interpolator
    cs = CubicSpline(qnmdata['a'], qnmdata['fdamp'])
    return cs(finspin) / (1. - erad)

def fmaxcalc(fac):
    if not fac['gamma2'] > 1:
        return abs(fac['frd']+(fac['fdm']*(-1.+np.sqrt(1.-pow_2_of(fac['gamma2']))) \
                               *fac['gamma3'])/fac['gamma2'])
    else:
        return abs(fac['frd']+(-fac['fdm']*fac['gamma3'])/fac['gamma2'])

def pow_2_of(x):
    return x*x

def pow_3_of(x):
    return x*x*x

def pow_4_of(x):
    return x*x*x*x

if __name__ == "__main__":
    from pycbc.waveform import get_fd_waveform
    from matplotlib import pyplot as plt
    m1 = 36.
    m2 = 32.
    s1z = 0.
    s2z = 0.
    # generate reference from pycbc
    hp, _ = get_fd_waveform(
        approximant='IMRPhenomD', mass1=m1, mass2=m2,
        spin1z=s1z, spin2z=s2z,
        distance=1., f_lower=20., f_upper=500.,
        spin_order=-1, phase_order=-1,
        delta_f=1./2048, f_ref=20.)
    kmin = np.sum(hp.sample_frequencies < 20.)
    kmax = np.sum(hp.sample_frequencies < 500.)
    # generate template from this code
    n = int(480*2048)
    f = np.linspace(20, 500, num=n)
    t = generate(f, mass1=m1, mass2=m2,
                 chi1=s1z, chi2=s2z, fref=20.)

    #fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    #ax[0].loglog(hp.sample_frequencies, abs(hp), label='lalsuite')
    #ax[0].loglog(f, abs(t), label='dansuite')
    #ax[0].set(xlabel='frequency', ylabel='strain amplitude',
    #       xlim=(15, 800))
    #ax[0].legend()
    #ax[1].plot(abs(t)/abs(hp[kmin:kmax]), label='waveform ratio')
    #ax[1].legend()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(hp.sample_frequencies[kmin:kmax],
               np.unwrap(np.angle(hp[kmin:kmax])), label='lalsuite')
    ax[0].plot(f, np.unwrap(np.angle(t)), label='dansuite')
    ax[0].set(xlabel='frequency', ylabel='phase')
    ax[0].legend()
    plt.show()

