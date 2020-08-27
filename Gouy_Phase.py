import numpy as np
import matplotlib.pyplot as plt
import beam as beam
from scipy.optimize import curve_fit as fit
from scipy.fft import fft #as fft
from scipy.fft import ifft #as ifft

def step(Uin, k_z, D): # Propagate input Beam object over distance, D; return Beam object. Fourier algorithm. 
        Pin = fft(Uin)                       # FFT of amplitude distribution, U, gives spatial frequency distribution, Pin at initial position, z.
        Pout = Pin * np.exp(1j * k_z * D)   # Multiply Spatial frequency distribution by phase-factor corresponding to propagation through distance, D, to give new spatial frequency distribution, Pout. 
        Uout = ifft(Pout)                       # IFFT of spatial frequency distribution gives amplitude distribution, Uout, at plane z = z + D
        return Uout

def Gaussian(space, offset, height, width): # Defines Gaussian function for fitting; space is a 1D array. 
    return height * np.exp(-((space-offset)/width)**2)

def Gaussfit(space,Array,init_width=30):# Fit Gaussian to magnitude of Amplitude array. Return fit parameters. 
    init_params = [0.1,1,init_width]    # Use initial width parameter smaller than expected width for beam profile. Otherwise, use initial width parameter larger than expected width - avoids bad fit in k-space. 
    est_params, est_err = fit(Gaussian, space, Array, p0 = init_params)
    return est_params # [offset, amplitude, width]

wav = 1.064e-3                                                      # wavelength in mm
z0 = 0                                                          # input waist location
b = 1000                                                            # input Rayleigh range
w0 = np.sqrt(b * wav / np.pi)                                       # input waist size - specified by Rayleigh range
space_0 = 5000                                                      # Start - M1

W = 200                                                             # Width of window in mm
xres = 0.01                                                         # 1D array representing kz-space. Derived from condition for monochromaticity.
N = int(W / xres)                                                   # Number of x-bins (keeps resolution the same)  
x = np.linspace(-W/2, W/2, N)                                       # 1D array representing x-space
kneg = np.linspace(-(np.pi * N)/W, -(2 * np.pi)/W, int(N/2))        # 1D array representing kx-space from max -ve value to min -ve value
kpos = np.linspace((2 * np.pi)/W, (np.pi * N)/W, int(N/2))          # 1D array representing kx-space from min +ve value to max +ve value
kx = np.concatenate((kpos,kneg), axis = 0)                          # 1D array representing kx-space. Order of values matches that spatial frequency distribution derived from FFT of amlitude distribution
kwav = 2 * np.pi / wav                                              # Magnitude of wave-vector
kz = np.sqrt(kwav**2 - kx**2)                                       # 1D array representing kz-space. Derived from condition for monochromaticity. 
zres = 5000                                                         # z-resolution in mm: 3000 for most; 50 for beam profile. 
q0 = z0 - 1j * np.pi * w0**2 / wav                                  # Input beam parameter
U = (1/q0) * np.exp(1j * kwav * x**2 / (2 * q0))                    # Input array
w = [Gaussfit(x,abs(U),1)[2]]                                       # Initialise width list

N = len(U)
idx = int(N/2)
d = np.linspace(-3000,3000,100)
plane = []
phase = []
gouy = []
U0 = U[idx]
phase_0 = np.angle(U0) % (2 * np.pi)
U = U * np.exp(-1j * phase_0)
U1 = U
U0 = U1[idx]
pl = 0
ph = (np.angle(U0)) % (2 * np.pi)
gy_prev = pl - ph
for i in range(len(d)):
    U1 = step(U,kz,d[i])
    U0 = U1[idx]
    pl = (kwav * d[i]) % (2 * np.pi)
    ph = (np.angle(U0)) % (2 * np.pi)
    gy = pl - ph
    if (gy - gy_prev) > np.pi:
        gy = gy - 2 * np.pi
    if (gy - gy_prev) < -np.pi:
        gy = gy + 2 * np.pi
    plane.append(pl)
    phase.append(ph)
    gouy.append(gy)
    gy_prev = gy
    #print(plane, '\t', phase, '\t', gouy)

plane = np.asarray(plane)
phase = np.asarray(phase)
gouy = np.asarray(gouy)
#plt.figure(1)
#plt.plot(d,plane)
#plt.figure(3)
#plt.plot(d,phase)
plt.figure(5)
plt.plot(d,gouy)
plt.show()

'''
start = 0
size = 10
end = 1000
m = int((end - start) / size)
n = int(len(x)/2)
U0 = U[n]
d = [start]
p = [np.angle(U0)]
for i in range(m):
    U = step(U,size)
    U0 = U[n]
    d.append(d[-1]+size)
    p.append(np.angle(U0))

p = np.asarray(p)
plt.figure()
plt.plot(d,p)
plt.show()
'''