#The Purpose of this code is to get angles for going to coprecessing frame and time
#And define functions that rotate waveforms 
import numpy as np
import matplotlib.pyplot as plt
from math import *
from numpy import linalg as LA


def SphericalPolarAngles(v1):
   """
   Given a 3D vector v1, calculates its polar and azimuthal
   orientation. Returns (Theta, Phi)
   Theta = arccos (v1[2]);
   Phi = atan2(v1[1], v1[0]);
   """

   norm = np.sqrt(np.dot(v1,v1))
   Theta = np.arccos(v1[2] / norm) # theta ==> -beta
   Phi = np.arctan2(v1[1], v1[0]);
   return Theta, Phi


#The average orientation tensor for preferred direction
def clm(l,m):
  """
  Prefactors in definitions of average orientation tensor given in Appendex of arXive 1205.2287v1 
  """
  if (m > l or m < -l ):
    return  0
  return sqrt(l*(l+1) - m*(m+1))

#Lab  Richard O'Shaughnessy Paper  #waveform from a dictionary "saved data"
def I0(lmax, waveform): 
  """
  One of the terms in average orientation tensor given in Appendex of arXive 1205.2287v1
  This will be a real quantity
  """
  Sum = 0.
  for l in range(2, lmax + 1):
    for m in range(-l, l+1):
      Sum += (l*(l+1) -m**2)* (abs(waveform[(l,m)])**2)
  return (1./2.)*Sum

def I1(lmax, waveform):
  """
  One of the terms in average orientation tensor given in Appendex of arXive 1205.2287v1
  This will be a complex quantity
  """
  Sum =0.
  for l in range(2,lmax + 1):
    for m in range(-l, l): #to avoid psi(l, m+1 , m+1 > l case)
      Sum += clm(l,m)*(m+1./2)*waveform[(l,m)]* waveform[(l,m+1)].conjugate()
  return Sum

def I2(lmax, waveform):
  """
 One of the terms in average orientation tensor given in Appendex of arXive 1205.2287v1
  This will be a complex quantity
  """
  Sum = 0.
  for l in range(2,lmax + 1):
    for m in range(-l, l-1): #to avoid psi(l, m+1 , m+2 > l case)
      Sum += clm(l,m)*clm(l,m+1)*waveform[(l,m)]* waveform[(l,m+2)].conjugate()
  return (1./2.)*Sum


def Izz(lmax, waveform):
  """
 One of the terms in average orientation tensor given in Appendex of arXive 1205.2287v1
  This will be a real quantity
  """
  Sum = 0.
  for l in range(2,lmax + 1):
    for m in range(-l, l+1):
      Sum += m**2 * (abs(waveform[(l,m)])**2)
  return Sum

def Lab(lmax, waveform):

  """
    Purpose of Lab is to construct a 3 by 3 matrix whose dominant \
    eigen vector provides two Euler angles that can be used to  
    rotate the waveform into a frame where radiation is emitted 
    along z direction such that waveform behave in this frame essentially
    as a non-precessing wavefvorm. 
    The matrix is symmetric where it has all components like lxx etc are scalars
  """

  denom = 0.
  for l in range(2,lmax + 1):
    for m in range(-l, l+1):
      denom += (abs(waveform[(l,m)])**2)


  lxx = 1.0/denom *(I0(lmax, waveform) + I2(lmax, waveform).real)
  lyy = 1.0/denom *(I0(lmax, waveform) - I2(lmax, waveform).real)
  lzz = 1.0/denom * Izz(lmax, waveform)
  lxy = 1.0/denom * I2(lmax, waveform).imag
  lxz = 1.0/denom * I1(lmax, waveform).real
  lyz = 1.0/denom * I1(lmax, waveform).imag

  M = [[0]*3 for i in range(3)]
  M[0][0] = lxx
  M[0][1] = lxy
  M[0][2] = lxz
  M[1][0] = lxy
  M[1][1] = lyy
  M[1][2] = lyz
  M[2][0] = lxz
  M[2][1] = lyz
  M[2][2] = lzz

  return M

#######################################################
def Coprecess_Angles( lmax, waveform): 
  """
  Using the principle direction of Lab matrix 
  we will compute the two Euler angles for rotating waveform
  into a co-rotating frame.
  The third Euler angle is obtained using the two given Euler angles\
   using Boyle et al integral 
  """

# remove unused variables

  if 'length' in waveform:
    #alen = waveform('length')
    alen = waveform['length']
  else:
    alen = len(waveform[(2,2)])

  Alp = np.zeros(alen, dtype=np.float64)
  Bta = np.zeros(alen, dtype=np.float64)
  Gma = np.zeros(alen, dtype=np.float64)

  lab = Lab(lmax, waveform)

  axx = lab[0][0]
  axy = lab[0][1]
  axz = lab[0][2]
  ayy = lab[1][1]
  ayz = lab[1][2]
  azz = lab[2][2]
  
  mat =  [[0]*3 for i in range(3)]
  x = [[0]*3 for i in range(3)]
  d = [0]*3
  v2 = [0]*3
  v3 = [0]*3
  vold = None
 
  for i in range(alen):
    mat[0][0] = axx[i]
    mat[0][1] = axy[i]
    mat[1][0] = axy[i]
    mat[2][0] = axz[i]
    mat[0][2] = axz[i]
    mat[1][1] = ayy[i]
    mat[1][2] = ayz[i]
    mat[2][1] = ayz[i]
    mat[2][2] = azz[i]

    eigenValues, eigenVectors =  LA.eig(np.array(mat))

# Explain sorting

    idx = eigenValues.argsort()[::-1]   
    d = eigenValues[idx]
    x = eigenVectors[:,idx]  


    # Note eigenVector[:,0] is the first eigenvector, etc.

    v1 = np.array((x[0][0], x[1][0], x[2][0]))
    v2 = np.array((x[0][1], x[1][1], x[2][1]))
    v3 = np.array((x[0][2], x[1][2], x[2][2]))


    # Note: Want the z-component of the waveform direction initiall to
    # be positive. Otherwise. This is an arbitrary choice. For later
    # times, choose sign of the eigenvector to maximize overlap with
    # previous time's eigenvector.
    if ((i==0 and v1[[2]] < 0) or (i> 0 and np.dot(vold, v1)< 0)):
      v1 = -v1


    vold = v1.copy()  #in numpy it points to address so vold is v[i]t v[i-1]


    # Please explain ..
    beta, gamma = SphericalPolarAngles(v1)
    if (i == 0):
      alpha = - gamma * cos(beta)
      alphabar = 0
    else:
      # Cite and equation
      #alpha  = alphaold - (beta -betaold)*(cos(0.5*(betaold + beta)))
      alphabar = alphabarold - 0.5*(gammaold+gamma)*((beta - betaold))*sin(0.5*(betaold+beta))
      alpha  =  -gamma *cos(beta) + alphabar
    alphabarold = alphabar
    betaold = beta
    gammaold = gamma

    Alp[i] = alpha
    Bta[i] = beta
    Gma[i] = gamma

  return  Alp, Bta, Gma


def wigner(l, m, mp, alpha, beta, gamma):
  """
  For rotating waveforms Wigner rotations are used. factorial from numpy.
  Based on Patricia Schmidth paper. Verified via M-Boyle numpy module
  """
  fac = np.math.factorial
  term = sqrt(fac(l+m) * fac(l-m)* fac(l+mp) * fac(l-mp))
  MinK = max(0, m-mp)
  MaxK = min(l+m , l-mp)
  Sum = 0.

  for k in range(MinK, MaxK+1):
    Sum = Sum + (sin(beta/2.0)**(2*k +mp-m))*(cos(beta/2.0)**(2*l-2*k-mp+m))*((-1)**(k+m-mp))/(fac(k)*fac(l+m-k)*fac(l-mp-k)*fac(mp-m+k))
  Sum = Sum * term * (cos(mp * gamma + m * alpha) + (sin(mp * gamma + m * alpha) * 1j))
  return Sum


def unwrap_phase(Z):
  """
  Given complex hlm a+ib
  Function will return 
  unwrapped phase
  """
  phase =  np.angle(Z)
  return np.unwrap(phase)

def Amplitude(Z):
  """
  Given complex waveform a+ib
  the function will return 
  amplitude = sqrt(a^2 + b^2) 
  """
  return np.absolute(Z)

def angular_frequency(Z, T):
  """
  Given a+ib and T for delta_t 
  the function will compute
  smooth phase, get the omega
  by central differencing 
  of frequency.
  NOTE: This is angular frequency
  freq = omega/2pi if one need
  """
  phase = np.angle(Z)
  smoothphase = np.unwrap(phase)#-1123.73 
  # angular frequency using centered differencing not a good method though
  delt = T[1] -T[0]
  omega = np.empty()
  omega = 1e0/delt*np.diff(smoothphaseSXS)


