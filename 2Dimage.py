import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft #as fft
from scipy.fft import ifft #as ifft
from scipy.fft import fft2 #as fft
from scipy.fft import ifft2 #as ifft

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
    U: 1D array, Complex amplitude
    x: 1D array, x-co-ordinate
    kneg/kpos: 1D arrays, negative/positive components of k-vactor along x-axis (spatial frequencies)
    kx: 1D array, component of k-vector along x-axis (spatial frequency)
    kz: 1D array, component of k-vector along z-axis
    kwav: constant, magnitude of wavevector
    zres: constant: resolution in z
    '''
    def __init__(self, *args): # Initialises amplitude array.
        self.W = 3                                                                             # Width of window in mm
        xres = 0.001                                                                            
        yres = xres
        self.N = int(self.W / xres)                                                             # Number of x-bins (keeps resolution the same)  
        #self.x, self.y = np.ogrid[0:(self.N),0:(self.N)]
        #self.x = np.arange[0:(self.N)]
        #self.x = xres * (self.x - self.N / 2) + xres / 2
        #self.y = yres * (self.y - self.N / 2) + yres / 2                                      
        self.x = np.linspace(-self.W/2,self.W/2,self.N)
        self.y = np.linspace(-self.W/2,self.W/2,self.N)
        self.x = np.reshape(self.x,(len(self.x),1))
        self.y = np.reshape(self.y,(1,len(self.y)))
        self.kwav = 2 * np.pi / wav                                                             # Magnitude of wave-vector        
        self.kxneg = np.linspace(-(np.pi * self.N)/self.W, -(2 * np.pi)/self.W, int(self.N/2))   # 1D array representing kx-space from max -ve value to min -ve value
        self.kxpos = np.linspace((2 * np.pi)/self.W, (np.pi * self.N)/self.W, int(self.N/2))     # 1D array representing kx-space from min +ve value to max +ve value
        self.kx = np.concatenate((self.kxpos,self.kxneg), axis = 0)                               # 1D array representing kx-space. Order of values matches that spatial frequency distribution derived from FFT of amlitude distribution                                    # 1D array representing x-space
        self.kyneg = np.linspace(-(np.pi * self.N)/self.W, -(2 * np.pi)/self.W, int(self.N/2))   # 1D array representing kx-space from max -ve value to min -ve value
        self.kypos = np.linspace((2 * np.pi)/self.W, (np.pi * self.N)/self.W, int(self.N/2))     # 1D array representing kx-space from min +ve value to max +ve value
        self.ky = np.concatenate((self.kypos,self.kyneg), axis = 0)                               # 1D array representing kx-space. Order of values matches that spatial frequency distribution derived from FFT of amlitude distribution    
        self.kx = np.reshape(self.kx,(len(self.kx),1))
        self.ky = np.reshape(self.ky,(1,len(self.ky)))
        self.kz = np.sqrt(self.kwav**2 - self.kx**2 - self.ky**2)                                            # 1D array representing kz-space. Derived from condition for monochromaticity. 
        self.zres = 3000                                                                        # z-resolution in mm: 3000 for most; 50 for transverse profile as function of z. 
        U0 = np.ones((len(self.x), len(self.y)))                                                               # 2D array representing plane wave.
        self.U = U0                                                                             # Initialise amplitude array.

    def aperture(self,width):
        Uin = self.U
        frame = (self.W - width) / 2
        frame_bins = int(round(self.N * (frame / self.W)))
        width_bins = self.N - 2 * frame_bins
        mx = np.concatenate((np.zeros((frame_bins,1)),np.ones((width_bins,1)),np.zeros((frame_bins,1))), axis = 0)
        my = np.concatenate((np.zeros((1,frame_bins)),np.ones((1,width_bins)),np.zeros((1,frame_bins))), axis = 1)
        mask = mx * my
        Uout = Uin * mask
        self.U = Uout
        return self

    def supergaussian(self,width,exponent):
        Uin = self.U
        sgx = np.exp(-((self.x)**2 / (2 * (width**2)))**exponent)
        sgy = np.exp(-((self.y)**2 / (2 * (width**2)))**exponent)
        mask = sgx * sgy
        Uout = Uin *  mask       # super gaussian function
        self.U = Uout
        return self

    def step(self, D): # Propagate input Wave object over distance, D; return Wave object. Fourier algorithm. 
        Pin = fft2(self.U)                       # FFT of amplitude distribution, U, gives spatial frequency distribution, Pin at initial position, z.
        Pout = Pin * np.exp(1j * self.kz * D)   # Multiply Spatial frequency distribution by phase-factor corresponding to propagation through distance, D, to give new spatial frequency distribution, Pout. 
        Uout = ifft2(Pout)                       # IFFT of spatial frequency distribution gives amplitude distribution, Uout, at plane z = z + D
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
        Uout = Uin * np.exp(-1j * self.kwav * (self.x**2 + self.y**2) / (2 * f))
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
        #axes.set_xlim([-10, 10])
        axes.set_ylim([0, 1.1])
        plt.grid(which = 'major', axis = 'both')
        plt.xlabel('x / mm')
        plt.ylabel('Normalised amplitude distribution')
        #plt.legend()
        plt.tight_layout()

    def freq_plot(self,n=2): # Plot magnitude of Spatial frequency array in k-space. 
        P = fft(self.U)
        kxplot = np.concatenate((self.kxneg,self.kxpos), axis = 0)     
        Pneg = P[int(len(P)/2):]
        Ppos = P[:int(len(P)/2)]
        Pswap = np.concatenate((Pneg,Ppos), axis = 0)
        Pabs = abs(Pswap)
        Pplot = Pabs/max(Pabs)
        plt.figure(n)
        plt.plot(kxplot,Pplot,'-')
        axes = plt.gca()
        #axes.set_xlim([-500, 500])
        axes.set_ylim([0, 1.1])
        plt.grid(which = 'major', axis = 'both')
        plt.xlabel('k_x / mm^-1')
        plt.ylabel('Normalised spatial frequency distribution')
        #plt.legend()
        plt.tight_layout()

    def plot_2D(self,n=1): # Plot magnitude of Amplitude array in x-space. 
        #xplot = np.reshape(self.kx,(len(self.kx),1))
        #yplot = np.reshape(self.ky,(len(self.ky),))
        Uplot = abs(self.U)#/max(abs(self.U))
        plt.figure(n)
        plt.pcolormesh(Uplot, cmap = 'gray') # plot vs x and y
        #plt.contour(self.U)

def system(): 
    # Runs series of methods corresponding to wave propagation through various elements in system. 
    U = Wave()
    U = U.aperture(0.005)
    #U = U.supergaussian(0.05,10)
    U.plot_2D(1)
    U = U.propagate(S1)
    U = U.lens(F1)
    U = U.propagate(S1)
    U.plot_2D(2)
    U = U.propagate(S1)
    U = U.lens(F1)
    U = U.propagate(S1)
    U.plot_2D(3) 

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
# 1D system
'''
#U = U.aperture(10)
U = U.supergaussian(200)
U.amp_plot(1)
#U.freq_plot(2)
U = U.lens(F1)
U = U.propagate(S1)
#U.amp_plot(3)
U.freq_plot(4)
U = U.propagate(S1)
U = U.lens(F1)
U = U.propagate(S1)
U.amp_plot(5)
#U.freq_plot(6)
'''