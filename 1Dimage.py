import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft #as fft
from scipy.fft import ifft #as ifft
from scipy.fft import fftshift

# Program description
'''
Uses Fourier method of Sziklas and Seigman (1975) [see also Leavey and Courtial, Young TIM User Guide (2017).] for beam propagation.
'''

# Inputs
wav = 1e-3                                                 # wavelength in mm
F1 = 100                                                       # Lens 1 in mm
S1 = 100                                                       # Space 1 in mm

class Wave: 
    '''Represents Wave in x, propagating along z. 
    Attributes: 
    W: width of window in mm
    res: resolution in x and y in mm
    N: number of bins along each axis
    x: 1D array, x-co-ordinate
    kwav: constant, magnitude of wavevector
    kneg/kpos: 1D arrays, negative/positive components of k-vactor along x-axis (spatial frequencies)
    kx: 1D array, component of k-vector along x-axis (spatial frequency)
    kz: 1D array, component of k-vector along z-axis
    zres: constant: resolution in z
    U: 1D array, Complex amplitude
    '''
    def __init__(self, *args): # Initialises amplitude array.
        self.W = 10                                                                                 # Width of window in mm
        self.res = 0.01                                                                             # 1D array representing kz-space. Derived from condition for monochromaticity.
        self.N = int(self.W / self.res)                                                                  # Number of x-bins (keeps resolution the same)  
        self.x = np.linspace(-self.W/2, self.W/2, self.N)                                           # 1D array representing x-space
        self.kwav = 2 * np.pi / wav                                                                 # Magnitude of wave-vector        
        self.kx_noshift = np.linspace(-(np.pi * self.N)/self.W, (np.pi * self.N)/self.W, self.N)    # Define kx-array
        self.kx = fftshift(self.kx_noshift)                                                         # fftshift so that kz array matches (un-shifted) spatial frequency distibution (P) in step method
        self.kz = np.sqrt(self.kwav**2 - self.kx**2)                                                # 1D array representing kz-space. Derived from condition for monochromaticity. 
        self.zres = 3000                                                                            # z-resolution in mm: 3000 for most; 50 for transverse profile as function of z. 
        self.U = np.ones(len(self.x))                                                               # Input array: plane wave.

    def aperture(self,width):
        Uin = self.U
        frame = (self.W - width) / 2
        frame_bins = int(round(self.N * (frame / self.W)))
        width_bins = self.N - 2 * frame_bins
        mask = np.concatenate((np.zeros(frame_bins),np.ones(width_bins),np.zeros(frame_bins)), axis = 0)
        Uout = Uin * mask
        self.U = Uout
        return self

    def supergaussian(self,width,exponent):
        Uin = self.U
        Uout = Uin *  np.exp(-((self.x)**2 / (2 * (width**2)))**exponent)       # super gaussian function
        self.U = Uout
        return self

    def step(self, D): # Propagate input Wave object over distance, D; return Wave object. Fourier algorithm. 
        Pin = fft(self.U)                       # FFT of amplitude distribution, U, gives spatial frequency distribution, Pin at initial position, z.
        Pout = Pin * np.exp(1j * self.kz * D)   # Multiply Spatial frequency distribution by phase-factor corresponding to propagation through distance, D, to give new spatial frequency distribution, Pout. 
        Uout = ifft(Pout)                       # IFFT of spatial frequency distribution gives amplitude distribution, Uout, at plane z = z + D
        self.U = Uout
        return self

    def propagate(self,distance): # Propagate Wave object through distance with resolution, zres; return Wave object. 
        Uprev = self
        res = self.zres                 
        num = distance // res           # number of steps: divide distance by resolution. 
        rem = distance % res            # remainder of division: final step size. If num = 0, i.e. zres > distance, single step taken, equal to distance. 
        for i in range(num):            # num steps
            Unext = Uprev.step(res)
            Uprev = Unext
        Unext = Uprev.step(rem)         # Final step of size rem. 
        return Unext

    def tilt(self,angle): # Applies linear phase-gradient, simulating effect of tilting mirror. Input angle in mrad. 
        Uin = self.U
        a = angle / 1000
        Uout = Uin * np.exp(-1j * self.kwav * self.x * np.sin(a))
        self.U = Uout
        return self

    def lens(self,f): # Lens element of focal length, f. 
        Uin = self.U
        Uout = Uin * np.exp(-1j * self.kwav * self.x**2 / (2 * f))
        self.U = Uout
        return self

    def mirror(self,R): # Mirror element of radius of curvature, R. 
        Uin = self.U
        Uout = Uin * np.exp(-1j * self.kwav * self.x**2 / R)
        self.U = Uout
        return self

    def amp_plot(self,n=1): # Plot magnitude of Amplitude array in x-space. 
        Uplot = abs(self.U)/max(abs(self.U))
        plt.figure(n)
        plt.plot(self.x,Uplot,'-')
        axes = plt.gca()
        axes.set_xlim([-5, 5])
        axes.set_ylim([0, 1.1])
        plt.xlabel('x / mm')
        plt.ylabel('Normalised spatial distribution')

    def freq_plot(self,n=2): # Plot magnitude of Spatial frequency array in k-space. 
        P = fft(self.U)
        kxplot = fftshift(self.kx)   
        Pshift = fftshift(P)
        Pabs = abs(Pshift)
        Pplot = Pabs/max(Pabs)
        plt.figure(n)
        plt.plot(kxplot,Pplot,'-')
        axes = plt.gca()
        axes.set_xlim([-300, 300])
        axes.set_ylim([0, 1.1])
        plt.xlabel('k_x / mm^-1')
        plt.ylabel('Normalised spatial frequency distribution')

def system(): 
    # Runs series of methods corresponding to wave propagation through various elements in system. 
    U = Wave()
    #U = U.aperture(0.6)
    U = U.supergaussian(0.3,10)
    U.amp_plot(1)
    U = U.propagate(S1)
    U = U.lens(F1)
    U = U.propagate(S1)
    U.freq_plot(2)
    U = U.propagate(S1)
    U = U.lens(F1)
    U = U.propagate(S1)
    U.amp_plot(3)
    #U.freq_plot(6)

def main():
    system()
    plt.show()
    
if __name__ == "__main__":
    main()

# Redundant or stored code
# ------------------------
# Mask
'''
mask = np.concatenate((np.zeros(int(0.475*R)),np.ones(int(0.01 * R)),np.zeros(int(0.03 * R)),np.ones(int(0.01 * R)),np.zeros(int(0.475 * R))), axis = 0)    # double slit
mask = np.concatenate((np.zeros(int(0.25*self.N)),np.ones(int(0.5 * self.N)),np.zeros(int(0.25 * self.N))), axis = 0)                                       # single slit
'''
# fftshift 'by hand'
'''
#Pneg = P[int(len(P)/2):]
#Ppos = P[:int(len(P)/2)]
#Pswap = np.concatenate((Pneg,Ppos), axis = 0)
'''