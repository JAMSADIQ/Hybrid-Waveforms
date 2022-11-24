#given fmin, fref get NRSur and PN modes PN modes will have fminPN =fminNR - 3 to have a longer waveform
#For mismatch we need SEOB or longer NR waveforms.
#python get_data.py --fmin 15 --fref 20 --q 2  --approximant SpinTaylorT4 --output-prefix PN

import numpy as np
import lalsimulation as lalsim
import  RIFT.lalsimutils as lalsimutils
import lal
import gwsurrogate
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fmin",default=20, type=float, help="minimum start frequency (Hz).depending on NRSur model there is a limit on fmin")
parser.add_argument("--fref",default=20,type=float, help="reference frequency (Hz) >= fmin ")
parser.add_argument("--q",default=1.001,type=float, help="mass ratio: m1/m2 >=1.")
parser.add_argument("--approximant",default='SpinTaylorT4',help="model-approximant")
parser.add_argument("--surrogatemodel",default='NRSur7dq4',help="model-approximant")
parser.add_argument("--output-prefix",default="PN",help="Prefix of output file")
parser.add_argument("--pathplot",default="/home/jam.sadiq/public_html/",help="Prefix of output file")
opts = parser.parse_args() 
#parameters for each waveform
#Intrinsic change spins or make them in argparser choices
Mtot = 70
dist = 300 #300Mpc Mpc will add inside function
spin1x = 0.6025
spin1y = 0.445
spin1z = 0.282874586434922
spin2x = 0.7467
spin2y = 0.2778
spin2z = -0.0790504878105789
#fixed can be changed
T_window = 8.
fmaxSNR = 2000.
deltaT = 1. / 16384.


def get_PNmodes_without_RIFT(qratio, Mtotal, s1x, s1y, s1z, s2x, s2y, s2z, fref, fmin, dist=300., ampO=-1, approximant=lalsim.SpinTaylorT4, lmax=4, modesname='PN'):
    """
    solely using lalsimualtion to get PN modes
    RIFT code may have some issues in tapering
    the modes
    """
    m1 = qratio / ( 1. + qratio) * Mtotal
    m2 = 1. / (1. + qratio) * Mtotal
    m1_SI = m1 * lal.MSUN_SI
    m2_SI = m2 * lal.MSUN_SI
    dist_SI=dist*lal.PC_SI*1.e6
    phiRef=0.0
    rate=8192.*2.
    dT =1.0/rate
    modearray = lalsim.SimInspiralCreateModeArray()
    for idxl in range(lmax-1):
       l=idxl+2
       for idxm in range(2*l+1):
            m=l-idxm
            lalsim.SimInspiralModeArrayActivateMode(modearray, l, m);
    lalParsNR=lal.CreateDict()
    err_code =lalsim.SimInspiralWaveformParamsInsertModeArray(lalParsNR,modearray)
    lalParsTay = lal.CreateDict()
    err_code += lalsim.SimInspiralWaveformParamsInsertPNAmplitudeOrder(lalParsTay,ampO)
    hlmsSTT4 = lalsim.SimInspiralChooseTDModes(phiRef, dT, m1_SI, m2_SI, s1x, s1y, s1z, s2x, s2y, s2z, fmin, fref, dist_SI, lalParsTay, lmax, approximant);
    hlmSTT4_tmp=lalsim.SphHarmTimeSeriesGetMode(hlmsSTT4,2,2)
    Tvals =  float(hlmSTT4_tmp.epoch)+np.arange(len(hlmSTT4_tmp.data.data))*hlmSTT4_tmp.deltaT
    Time = np.arange(len(hlmSTT4_tmp.data.data))*hlmSTT4_tmp.deltaT
    #np.savetxt("PNTime.txt", Tvals)
    for l in range(2, LMAX+1):
        for m in range(-l, l+1):
            mode = lalsim.SphHarmTimeSeriesGetMode(hlmsSTT4,l,m).data.data
            plt.plot(Tvals, mode.real, label='l{0}m{1}'.format(l,m))
            plt.legend()
            plt.show()
            np.savetxt('PN{0}{1}'.format(l,m), np.stack((Tvals, mode.real,mode.imag), axis=-1))
    return 0
    
def get_hlm_modes(qratio, Mtotal, s1x, s1y, s1z, s2x, s2y, s2z, fref, fmin, dist=300., approximant=lalsim.GetApproximantFromString('SpinTaylorT4'), lmax=4, modesname='PN'):
    """
    get hlm modes in .txt format for given mass ratio, total mass, spins, refrence and minimum frequencies.
    """
    #get the m1, m2 from qratio and Mtotal
    m1 = qratio / ( 1. + qratio) * Mtotal
    m2 = 1. / (1. + qratio) * Mtotal
    m1_SI = m1 * lal.MSUN_SI
    m2_SI = m2 * lal.MSUN_SI
    #get a dictionary 
    P = lalsimutils.ChooseWaveformParams()
    P.m1 = m1_SI
    P.m2 = m2_SI
    P.s1x = s1x
    P.s1y = s1y
    P.s1z = s1z
    P.s2x = s2x
    P.s2y = s2y
    P.s2z = s2z
    P.fmin = fmin
    P.fref = fref
    P.deltaT = deltaT
    P.deltaF = 1. / T_window
    P.dist = dist * lal.PC_SI *1.e6 
    P.lambda1 = 0.0
    P.lambda2 = 0.0
    P.ampO = -1 #highest PN order
    P.phaseO = 7
    P.approx = lalsim.GetApproximantFromString(approximant)
    print("P.approx = ", P.approx)
    P.theta = 0.0
    P.phi = 0.0
    P.psi = 0.0
    P.tref = 0.0
    P.radec = False
    P.detector = 'H1'
    P.extra_params = {}
    hlmT_lal = lalsimutils.hlmoft(P, Lmax=lmax)
    #Now we have complex Time series for modes 
    #get time using dT and epoch 
    h22_key = hlmT_lal[(2,2)] #just key of 22 mode
    #remove wrapped zeros try to find better strategy
    argzeros = np.argwhere(h22_key.data.data)
     
    Tvals =  float(h22_key.epoch)+np.arange(len(h22_key.data.data))*h22_key.deltaT
    Tvals = Tvals[argzeros][:,0]
    #save data in  T, Re, Im for each mode
    #and plot mode by mode
    for mode in hlmT_lal.keys():
        hlm_data = hlmT_lal[mode].data.data[...][argzeros][:,0]
        Re, Im = hlm_data.real, hlm_data.imag
        np.savetxt("{0}{1}{2}".format(modesname, mode[0], mode[1]), np.stack((Tvals, Re, Im), axis=-1))
        plt.figure()
        plt.plot(Tvals, Re, label="l{0}m{1}".format(mode[0], mode[1]))
        plt.title(modesname)
        plt.legend()
        plt.show()
    return 0

def get_NRsur(qratio, Mtotal, s1x, s1y, s1z, s2x, s2y, s2z, fref, fmin, dist=300., surogate_model='NRSur7dq4', lmax=4, modesname='NRsur'):
    """
    get NRsur modes if one istalled gwsurrogate from conda 
    see https://pypi.org/project/gwsurrogate/
    and https://github.com/sxs-collaboration/gwsurrogate/blob/master/tutorial/website/NRSur7dq4.ipynb
    """
    #pull model 
    gwsurrogate.catalog.pull(surogate_model)#('NRSur7dq4')
    sur =  gwsurrogate.LoadSurrogate(surogate_model)#('NRSur7dq4')
    chiA = [s1x, s1y, s1z]
    chiB = [s2x, s2y, s2z]
    dt = deltaT
    t, h, dyn = sur(qratio, chiA, chiB, dt=dt, f_low=fmin, f_ref=fref, ellMax=lmax, M=Mtotal, dist_mpc=dist, units='mks')
    for mode in h.keys():
        Re, Im = h[mode].real, h[mode].imag
        np.savetxt("{0}{1}{2}".format(modesname, mode[0], mode[1]), np.stack((t, Re, Im), axis=-1))
        plt.figure()
        plt.plot(t, Re, label="l{0}m{1}".format(mode[0], mode[1]))
        plt.title(modesname)
        plt.legend()
        plt.show()
    return 0

#test
fminNRsur = 0.0
get_NRsur(opts.q, Mtot, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, opts.fref, fminNRsur, surogate_model=opts.surrogatemodel, lmax=4, modesname='NRsur')

fminPN = opts.fref - 5.0 #for longer PN the code use backward evolution
get_hlm_modes(opts.q, Mtot, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, opts.fref, fminPN, approximant=opts.approximant, lmax=4, modesname=opts.output_prefix)
