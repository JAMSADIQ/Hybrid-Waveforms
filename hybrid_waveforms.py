#asisuming whe have data SXS[lm]. form SXS22 SXS21 SXS43 et and PN22 PN44 etc
#each of these files have time real_hlm imag_hlm

#python hybridwaveform.py T-for-RigRot -1.3 T-Hybrid-Interval-Start -1.3 T-Hybrid-Interval-End -0.8
import matplotlib
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import argparse
import numpy as np
from math import *
import scipy
from scipy.optimize import minimize
import wavform_utils as utils




########### These need to be chosen manually ##################
T_junk,T_ringdown = -5.2, 0.0
T_for_RigRot, T_Hybrid_Interval_Start, T_Hybrid_Interval_End = -1.3, -1.3, -0.8
####################### Data Getting ##########################################
parser = argparse.ArgumentParser()
parser.add_argument("--lmax",default=4,help="Max l mode (with all corresponding m modes).")
parser.add_argument("--T_for_RigRot",default=-1.5,help="Time at which we do Rigid Rotation to Coprecess Frame.")
parser.add_argument("--T_Hybrid_Interval_Start",default=-1.5,help="t1 = time at which hybrid interval start. (usually t1= t_rot) ")
parser.add_argument("--T_Hybrid_Interval_End",default=-0.8,help="t2 = time at which hybrid interval ends.  ")
parser.add_argument("--T_Exact_Interval_Start",default=-0.4,help="t3 = time at which exact waveform tries to align with  hybrid waveform.  we tried it to align close to merger")
parser.add_argument("--T_junk",default=-1.5,help="tjunk= time at which numerical data has no junk radiation left. Already taken into account in Richard code  ")
parser.add_argument("--T_ringdown",default=0.01,help="tringdown = tapered data after merger. Already taken into account in Richard code ")
parser.add_argument("--T_window",default=32.,help="time window for data")
parser.add_argument("--fmaxSNR", default=1700,help= "maxSNR of LIGO 2000 Hz ")
opts = parser.parse_args()

lmax = opts.lmax

# Open data and save into dictionary 
hlm_NR    = {}
T_NR      = {}
Angles_NR = {}
for l in range(2, lmax+1):
    for m in range(-l, l+1):
        DataNR = np.loadtxt("SXS{0}{1}".format(l,m))
        hlm_NR[(l,m)] = DataNR[:,1] + DataNR[:, 2]*1j
        if (l==2 and m==2):
            T_NR = DataNR[:,0]

hlm_MD ={}
T_MD = {}
Angles_MD ={}
for l in range(2, lmax+1):
    for m in range(-l,l+1):
        DataMD = np.loadtxt("PN{0}{1}".format(l,m))
        hlm_MD[(l,m)] = DataMD[:,1] + DataMD[:, 2]*1j
        if (l==2 and m==2):
            T_MD = DataMD[:,0]

### assuming we also have true full waveform to test accuracy of hybrid
hlm_EX    = {}
T_EX      = {}
Angles_EX = {} #Coprecessing angles for rigid rotation
##
### Maybe SEOB model have only some modes not all so we make rest as zero modes
for l in range(2, lmax+1):
    for m in range(-l,l+1):
            DataEX = np.loadtxt("SEOB{0}{1}".format(l,m))
            hlm_EX[(l,m)] = DataEX[:,1] + DataEX[:, 2]*1j
        else:
            data22 = np.loadtxt("SEOB22")
            data22 = data22.T
            for i in range(len(data22[0])):
                hlm_EX[(l,m)] = 0.0*(data22[1] + data22[2]*1j)
        if (l==2 and m==2):
            T_EX = DataEX[:,0]

############################################################################
##Step I: Coprecess Angles Needed for Precessing Case Only 
#Coprecess_Angles gives the three Euler rotations at each time of waveform.
#%alphanr, betanr, gammanr = utils.Coprecess_Angles(lmax, hlm_NR)
alphanr, betanr, gammanr = utils.Coprecess_Angles(lmax, hlm_NR)
Angles_NR['Alpha_NR'] = alphanr.copy()
Angles_NR['Beta_NR'] = betanr.copy()
Angles_NR['Gamma_NR'] = gammanr.copy()


alpha_md, beta_md, gamma_md = utils.Coprecess_Angles( lmax, hlm_MD)
Angles_MD['Alpha_MD'] = alpha_md.copy()
Angles_MD['Beta_MD'] = beta_md.copy()
Angles_MD['Gamma_MD'] = gamma_md.copy()
#############################################################################

##Step II  Optimization
#(a)Data of Two waveforms in a hybrid Interval ########################
def get_data_for_hybrid_interval(l, m, t_rigrot, thyb_str, thyb_end, waveNR, waveModel, tnumeric, tmodel,initial_t0):
    """
    This function output  the data 
    of two waveform modes in hybrid interval.
    This data will be used for optimization for
    best choice of t, phi, Psi shifts to align 
    waveforms in hybrid interval [thyb_str, thyb_end]
    t_rigrot = time for rigid rotation of two waveforms
    t_rigrot can be chosen arbitrarily but must be
    the time in NR  waveform.
    initial_t0: Based on matching the freq of
    two waveforms at a fix time in hybrid interval.
    """
    startnr, endnr = np.searchsorted(tnumeric, (thyb_str, thyb_end))
    startnr = startnr -1
    tnumeric = tnumeric[startnr:endnr]
    alpha_nr = np.interp(t_rigrot , tnumeric , Angles_NR['Alpha_NR'][startnr:endnr])
    beta_nr  = np.interp(t_rigrot , tnumeric , Angles_NR['Beta_NR'][startnr:endnr])
    gamma_nr = np.interp(t_rigrot , tnumeric , Angles_NR['Gamma_NR'][startnr:endnr])

    alpha_model  =  np.interp(t_rigrot - initial_t0, tmodel , Angles_MD['Alpha_MD'])
    beta_model   =  np.interp(t_rigrot - initial_t0, tmodel , Angles_MD['Beta_MD'])
    gamma_model  =  np.interp(t_rigrot - initial_t0, tmodel , Angles_MD['Gamma_MD'])
    #For non-precessing case angles = 0
    #alpha_nr, beta_nr,gamma_nr , alpha_model, beta_model, gamma_model = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    NRhlm_hyb = utils.Rigid_rotate_waveform(l, m, alpha_nr, beta_nr, gamma_nr, waveNR)[startnr:endnr]
    print(len(NRhlm_hyb), len(tnumeric))

    MDhlm_hyb = utils.T_interp_Rigid_rotate_waveform(l, m, alpha_model, beta_model, gamma_model, waveModel,  initial_t0, tnumeric, tmodel)

    return  NRhlm_hyb, MDhlm_hyb 

#(b) Defining Optimization Function for Scipy Optimize Algorithm ##################

def OptimizeFunction(x):
    """
    Given the x = np.array( [t0guess, phi0guess, Psiguess])
    and get get_data_for_hybrid_interval(l, m, t_rigrot, \
    thyb_str, thyb_end, waveNR, waveModel, tnumeric, tmodel, initial_t0)
    , which are actual data  sets from dictionary used,
    this function will find the time and phase shifts and Psi (0 or pi/2).
    time shifts and phase shifts can be chosen in many number of ways
    phi = -2pi to +2pi
    t0 as we are using seconds so must must be larger than 0.001? 
    Psi related to the sign convention of waveform spherical
    harmonics decomposition.
    """
    t0 = x[0]
    phi0 = x[1]
    psi0 = x[2]
    Sum = 0
    for l in range(2, lmax+1):
        for m in range(-l, l+1):
            partial = get_data_for_hybrid_interval(l, m, T_for_RigRot, T_Hybrid_Interval_Start, T_Hybrid_Interval_End, hlm_NR, hlm_MD, T_NR, T_MD,t0)
            factor = 1; #20 if m%2==1 else  for m=1 mode dominance
            #Removing contributions from m=0 modes
            if m == 0:
                factor = 0
            Sum += np.sum(factor* np.absolute(partial[0] - partial[1] * np.exp( (m*phi0  + 2*psi0) *1j)))
    return  Sum *1e17
#This function requires OptimizeFunction type function

def Get_Optimized_Values(xstart):
    """
    Given initial guesses, xstart in array, this function 
    use the above  defined OptimizeFunction to get optimized
    values for t0, phi0, Psi.
    One can experiment with different methods 
    check like numpy minimize function for other choices
    tolerance can also be changed. We are using Nelder Mead 
    method from scipy.minimize function.
    The best min val and optimized vals will be the output 
    of this function
    The output will be function value with optimized parameters
    as well as the values of these parameters that gives best result (minima).
    """
    opt_result = minimize(OptimizeFunction, xstart, method='Nelder-Mead' , tol=1e-30)
    return opt_result.fun,  opt_result.x[0], opt_result.x[1] , opt_result.x[2]

### Trying different initial choices for a better minimum value
def get_best_OptimizedResults(tnr, tmd, hlmNR22, hlmMD22, tchoose):
    """
    This function will try to get the 
    best Optimized results for best minimum
    of the two waveforms in hybrid interval.
    This function takes numeric, model time and 
    only 22 mode of two waveforms for matching
    frequency at fixed time.
    We will iterate over different initial guesses
    phi [-2pi, 2pi], 
    t0 based on matching freq of two waveforms at 
    a fixed time probably t-rigidrotate.
    Return will be t0, phi0  for the
    best minimum function
    """
    phi0guess = np.linspace(-2*np.pi,2*np.pi, 10) 
    #timeshift choices  check get_t0guess
    t0guess =  get_t0guess(hlmNR22, hlmMD22,tnr ,tmd,tchoose)
    Del_t0 = 0.001 #can change this 
    tmin, tmax = t0guess + Del_t0, t0guess - Del_t0
    alen = len(phi0guess)
    t0guess = np.linspace(tmin, tmax,alen)
    Psi0 = np.array([0, np.pi/2])
    psi0guess= np.tile(Psi0, int(alen/2)) #one must have alen to be even

    Func = np.zeros(alen, dtype=np.float64)
    t0_val  = np.zeros(alen, dtype=np.float64)
    phi0_val = np.zeros(alen, dtype=np.float64)
    psi0_val = np.zeros(alen, dtype=np.float64)

    for i in range(alen):
        Func[i] , t0_val[i] , phi0_val[i] , psi0_val[i]= Get_Optimized_Values( np.array([ t0guess[i], phi0guess[i], psi0guess[i]]))
    print( "array of optimized func vals =" ,Func)
    idx = np.argmin(Func)
    print( "best values, func, t0, phi0, Psi =", Func[idx], t0_val[idx], phi0_val[idx], psi0_val[idx])
    return t0_val[idx], phi0_val[idx] , psi0_val[idx]

########################################################################################
##############Step III Resulting Hybrid Data Getting Code#################################
#Best Optimize Results
t0, phi0, psi0 = get_best_OptimizedResults(T_NR, T_MD, hlm_NR[(2,2)], hlm_MD[(2,2)], T_for_RigRot)
print("The best optimized vals = ", t0, phi0, psi0)
#Rotation at fixed time 
alpha_Numeric =  np.interp(T_for_RigRot , T_NR , Angles_NR['Alpha_NR'])
beta_Numeric  =  np.interp(T_for_RigRot , T_NR , Angles_NR['Beta_NR'])
gamma_Numeric =  np.interp(T_for_RigRot , T_NR , Angles_NR['Gamma_NR'])

t_model = T_MD + np.ones_like(T_MD)*t0  #shifted Time Model
#I need this shifted time for finding the rigid rotating angles.
alpha_Model =  np.interp(T_for_RigRot -t0, t_model , Angles_MD['Alpha_MD'])
beta_Model  =  np.interp(T_for_RigRot -t0, t_model , Angles_MD['Beta_MD'])
gamma_Model  =  np.interp(T_for_RigRot -t0, t_model , Angles_MD['Gamma_MD'])

########### Get Hybrid Data

hlm_Hyb = {}
T_Hyb = {}
Interp_Exc ={}
for l in range(2, lmax+1):
    for m in range(-l, l+1):
        #Only if we want data to be saved in files as output
        #outfile1=open("RigrotatedNR{0}{1}".format(l,m), "a")
        #outfile2=open("RigrotatedshiftedMD{0}{1}".format(l,m), "a")
        #outfile3=open("MD_and_NR_Hybrid{0}{1}".format(l,m), "a")

        newdataNR = utils.Rigid_rotate_waveform(l, m, alpha_Numeric, beta_Numeric, gamma_Numeric, hlm_NR)
        Re_newNR, Im_newNR = newdataNR.real ,  newdataNR.imag

        newdataMD = utils.Rigid_rotate_waveform(l, m, alpha_Model, beta_Model, gamma_Model, hlm_MD)*(np.exp((m*phi0+2*psi0)*1j))
        Re_newMD, Im_newMD  = newdataMD.real ,  newdataMD.imag

        Thyb, Re_hyb, Im_hyb= utils.Analytic_NR_Hybrid_Data(T_NR, t_model, newdataNR,  newdataMD, T_Hybrid_Interval_Start, T_Hybrid_Interval_End)
        hlm_Hyb[(l,m)] = Re_hyb + Im_hyb *1j
        if (l==2 and m== -2):
            T_Hyb = Thyb.copy()

        #Saving to Files if needed
#        for i in range(len(T_NR)):
#            outfile1.write("{0:20.16e} ".format(T_NR[i]))
#            outfile1.write("{0:20.16e} ".format(Re_newNR[i]))
#            outfile1.write("{0:20.16e} ".format(Im_newNR[i]))
#            outfile1.write("\n")
#        outfile1.close()
#        for i in range(len(t_model)):
#            outfile2.write("{0:20.16e} ".format(t_model[i]))
#            outfile2.write("{0:20.16e} ".format(Re_newMD[i]))
#            outfile2.write("{0:20.16e} ".format(Im_newMD[i]))
#            outfile2.write("\n")
#        outfile2.close()
##     
##
#        for i in range(len(Thyb)):
#            outfile3.write("{0:20.16e} ".format(Thyb[i]))
#            outfile3.write("{0:20.16e} ".format(Re_hyb[i]))
#            outfile3.write("{0:20.16e} ".format(Im_hyb[i]))
#            outfile3.write("\n")
#        outfile3.close()
#


######################################################################
######################################################################
#####################  Mismatch ######################################
alpha_ex, beta_ex, gamma_ex = utils.Coprecess_Angles(lmax, hlm_EX)
Angles_EX['Alpha_EX'] = alpha_ex.copy()
Angles_EX['Beta_EX'] = beta_ex.copy()
Angles_EX['Gamma_EX'] = gamma_ex.copy()
############# Exact data only "ROTATED"  no tphi shifts  ############### 
Rot_alpha_EX  =  np.interp(T_for_RigRot , T_EX , Angles_EX['Alpha_EX'])
Rot_beta_EX   =  np.interp(T_for_RigRot , T_EX , Angles_EX['Beta_EX'])
Rot_gamma_EX  =  np.interp(T_for_RigRot , T_EX , Angles_EX['Gamma_EX'])


Interp_Exp ={}
for l in range(2, lmax+1):
    for m in range(-l, l+1):
#%        RotdataEX = utils.Rigid_rotate_waveform(l, m, Rot_alpha_EX ,Rot_beta_EX, Rot_gamma_EX , hlm_EX)
 
        RotdataEX = Rigid_rotate_waveform(l, m, Rot_alpha_EX ,Rot_beta_EX, Rot_gamma_EX , hlm_EX)
        Interp_Exc[(l,m)] = np.interp(T_Hyb,T_EX, RotdataEX)


for l in range(2,lmax+1):
    for m in range(-l,l+1):
        if  m <= 0  :
            print("m=0 or neg")
        else:
            wave1 = utils.Waveform_afterWindowing(T_Hyb, Interp_Exc[(l,m)], T_junk,T_ringdown)
            wave2 = utils.Waveform_afterWindowing(T_Hyb, hlm_Hyb[(l,m)], T_junk,T_ringdown)

            l, m, Tptotalmass, Tptotmismatch = utils.get_plots_mismatch__freq_over_TotMass(T_Hyb, wave1, wave2 ,70 , l, m,  T_Hybrid_Interval_Start, T_Hybrid_Interval_End)
            print ("l=", l , "m=", m, "Mtot", Tptotalmass, "mismatch for l{0},m{1}".format(l,m), Tptotmismatch)

#plt.legend(fontsize=20, loc='upper center', bbox_to_anchor=(0.5, 1.15),ncol=3, fancybox=True, shadow=True)
plt.legend(loc=3,fontsize=20)# bbox_to_anchor=(1, 0.5), fontsize=20)
plt.semilogy()
plt.grid()
#plt.savefig("Mismatchplotsmodebymode.png")
#plt.show()
plt.figure()
#Strain Plot
theta= np.linspace(0, np.pi, 32)
phi = np.linspace(0, 2*np.pi, 64)
import lalsimulation as lalsim
psd=lalsim.SimNoisePSDaLIGOZeroDetHighPower
f_Low, f_Max = 20, 2000
Mstandard = 70.
for M in range(40, 60, 20):
    utils.Mismatchplot(hlm_Hyb,Interp_Exc,T_Hyb, theta, phi, M, Mstandard, psd, f_Low, f_Max)
plt.show()
