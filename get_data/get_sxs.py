#script to get sxs modes after installing sxs package https://sxs.readthedocs.io/en/main/
import h5py
import numpy as np
from sympy.physics.units import G, kg, s, m, au, year
import scipy.constants 
import matplotlib.pyplot as plt
import sxs
from numpy import pi
import lal
lmax = 2
#units
parsec = 648000/pi*au
megaparsec = 10**6 *lal.PC_SI #*parsec
msun = 4*pi**2*au**3/G/year**2
c = scipy.constants.speed_of_light*m/s

#specific distance (300Mpc) and total mass (70Msun)
DISTANCE=300*megaparsec
MASS = 70 * msun

#we will convert sxs waveforms in MKS units
def GeometricTime_To_MKS_Time(t_over_m, mass):
    """
    t_over_m = numpy array with t/M in geometric units
    mass = total mass. Should have units recognized by sympy
    Return value is in units of seconds but the units are removed.
    """
    factor = lal.G_SI*mass*lal.MSUN_SI/lal.C_SI**3 #((G*mass/c**3/s))
    print ("factor= ",factor)
    return t_over_m * factor

def GeometricStrain_TO_Observer_Strain(r_h_over_m, mass, distance):
    """
    r_h_over_m = numpy array with strain in geometric units
    mass = total mass. Should have units recognized by sympy
    distance = distance to source. Should have units recognized by sympy
    Return value is strain at observer location
    """
    factor = lal.G_SI*mass*lal.MSUN_SI/lal.C_SI**2/distance 
    return r_h_over_m  * factor

def get_modes_from_sxs(waveformnumber, MASS=70, DISTANCE=300*megaparsec):
    """
    use sxs catalog and get sxs modes in from we need
    """
    import sxs
    extrapolation_order = 4
    #waveformnumber = 0123, 0058
    metadata = sxs.load("SXS:BBH:"+waveformnumber+"/Lev/metadata.json")
    w = sxs.load("SXS:BBH:"+waveformnumber+"/Lev/rhOverM", extrapolation_order=extrapolation_order)
    index_junk_end =  w.index_closest_to(metadata.reference_time)
    #get sliced data after junk
    w_sliced = w[index_junk_end:]
    time = w_sliced.t
    print(len(time), len(w_sliced[:, w_sliced.index(2, 2)].data.view(float)))
    time_sec = GeometricTime_To_MKS_Time(time, MASS)
    h22 = w_sliced[:, w_sliced.index(2, 2)].data.view(float)
    print(len(time_sec), len(h22))
    real22, imag22 = h22.real, h22.imag
    newreal22 = GeometricStrain_TO_Observer_Strain(real22, MASS, DISTANCE)
    newimag22 = GeometricStrain_TO_Observer_Strain(imag22, MASS, DISTANCE)
    max_amp_indx = np.argmax(np.absolute( newreal22 +  newimag22 *1j ))
    print(max_amp_indx)
    quit()
    #Tshift using 22
    tshift = time_sec[max_amp_indx]
    dt = 1e0/2**14   #time_sec[-1] - time_sec[-2] can be reasonable
    N = int((time_sec[-1] - time_sec[0])/dt)
    Tarray = np.empty(N)
    for i in range(N):
        Tarray[i] = time_sec[0]+ i*dt
    NewT = np.interp(Tarray, time_sec, time_sec) - tshift
    for l in(2, lmax+1):
        for m in range(-l, l+1):
            hlmZ = w_sliced[:, w_sliced.index(l, m)].data.view(float)
            phys_real = GeometricStrain_TO_Observer_Strain(hlmZ.real, MASS, DISTANCE)       
            print(len(phys_real), len(time_sec))
            quit()
            NewRe = np.interp(Tarray, time_sec, phys_real)
            phys_imag = GeometricStrain_TO_Observer_Strain(hlmZ.imag, MASS, DISTANCE)
            NewIm = np.interp(Tarray, time_sec, phys_imag)
            plt.plot(Tarray, NewRe)
            plt.title('{0}{1}'.format(l, m))
            plt.show()
            np.savetxt('SXS{0}{1}'.format(l,m), np.stack((NewT,NewRe,NewIm), axis=-1))
    return 0

get_modes_from_sxs('0058', MASS=70, DISTANCE=300*megaparsec)
quit()

#extras
#If you downloaded .h5 file from sxs catalo use this function
def get_modes_from_file(filename, MASS, DISTANCE):
    """
     given .h5 file (rhOverM_Asymptotic_GeometricUnits_CoM.h5)
     in sxs format this function will save 
     modes from outermost extrapolation "N4" and return modes
     in files with time, real and imaginary parts of mode.
     file name sxslm (SXS22, SXS33)
     for system with Mtot= MASS and distance at DISTANCE
    """
    file = h5py.File(filename, "r")
    hlm=file[u'Extrapolated_N4.dir']
    #first get 22 mode for time such that merger is set at t = 0.0 sec
    time22, real22, imag22 =hlm[u'Y_l{0}_m{1}.dat'.format(2,2)][:].T
    time_sec = GeometricTime_To_MKS_Time(time22, MASS)
    newreal22 = GeometricStrain_TO_Observer_Strain(real22, MASS, DISTANCE)
    newimag22 = GeometricStrain_TO_Observer_Strain(imag22, MASS, DISTANCE)
    max_amp_indx = np.argmax(np.absolute( real22 +  imag22 *1j ))
    #Tshift using 22
    tshift = time_sec[max_amp_indx]
    dt = 1e0/2**14   #time_sec[-1] - time_sec[-2] can be reasonable
    N = int((time_sec[-1] - time_sec[0])/dt)
    Tarray = np.empty(N)
    for i in range(N):
        Tarray[i] = time_sec[0]+ i*dt
    for l in range(2,5):
        for m in range(-l,l+1):
            time, real, imag =hlm[u'Y_l{0}_m{1}.dat'.format(l,m)][:].T
            phys_time = GeometricTime_To_MKS_Time(time, MASS)
            #uniform grid with dt step and shift time such that merger is at t=0.0
            NewT = np.interp(Tarray, phys_time, phys_time) - tshift
            phys_real = GeometricStrain_TO_Observer_Strain(real, MASS, DISTANCE)
            NewRe = np.interp(Tarray, phys_time, phys_real)
            phys_imag = GeometricStrain_TO_Observer_Strain(imag, MASS, DISTANCE)
            NewIm = np.interp(Tarray, phys_time, phys_imag)
            np.savetxt('SXS{0}{1}'.format(l,m), np.stack((NewT,NewRe,NewIm), axis=-1))
    file.close()

#to remove junk radiation
def Planktaper(t, t1, t2, t3, t4):
    """
    Given time series of waveform compute the function that can deal with gibbs
    phenominon for ringdown and initial junk
    """
    assert (t1 < t2 and t2 < t3 and t3 < t4)
    if (t <= t1):
        return 0
    if (t1 < t< t2):
        y = (t2 - t1)/(t-t1) + (t2-t1)/(t-t2)
        if y > 0:
            return np.exp(-y)/(1+ np.exp(-y))
        else:
            return 1.0/(1+ np.exp(y))
    if(t2 <= t <= t3):
        return 1
    if(t3 < t < t4):
        z = (t3-t4)/(t-t3) +  (t3-t4)/(t-t4)  #removes negative sign in first term
        return 1e0/(1+ np.exp(z))
    if (t >= t4):
        return 0
    return t
