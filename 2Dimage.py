import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2
from scipy.fft import ifft2
from scipy.fft import fftshift

# Program description
'''
Uses Fourier method of Sziklas and Seigman (1975) [see also Leavey and Courtial, Young TIM User Guide (2017).] for beam propagation.
'''

# Inputs
wav = 0.001   # wavelength in mm
F1 = 100      # Lens 1 in mm
S1 = 100      # Space 1 in mm

class Wave: 
    '''Represents Wave in x and y, propagating along z. 
    Attributes: 
    W: width of window in mm
    res: resolution in x and y in mm
    N: number of bins along each axis
    x: 1D array, x-co-ordinate, row vector
    y: 1D array, y-co-ordinate, column vector
    kwav: constant, magnitude of wavevector
    kx: 1D array, component of k-vector along x-axis (spatial frequency), row vector
    ky: 1D array, component of k-vector along y-axis (spatial frequency), row vector
    kz: 2D array, component of k-vector along z-axis
    zres: constant: resolution in z
    U: 2D array, Complex amplitude
    '''
    def __init__(self, *args): # Initialises amplitude array.
        self.W = 10                                                                                 # Width of window in mm
        self.res = 0.01                                                                            # Resolution in x    
        self.N = int(self.W / self.res)                                                            # Number of x-bins (keeps resolution the same)                                     
        self.x = np.linspace(-self.W/2,self.W/2,self.N)                                             # Define x-array
        self.y = np.reshape(self.x,(len(self.x),1))                                                 # Define y-array to be same as x-array, in column-vector form
        self.x = np.reshape(self.x,(1,len(self.x)))                                                 # Make x-array explicitly into a row-vector
        self.kwav = 2 * np.pi / wav                                                                 # Magnitude of wave-vector        
        self.kx_noshift = np.linspace(-(np.pi * self.N)/self.W, (np.pi * self.N)/self.W, self.N)    # Define kx-array
        self.kx = fftshift(self.kx_noshift)                                                         # fftshift so that kz array matches (un-shifted) spatial frequency distibution (P) in step method
        self.ky = np.reshape(self.kx,(len(self.kx),1))                                              # Define ky-array to be same as kx-array, in column-vector form
        self.kx = np.reshape(self.kx,(1,len(self.kx)))                                              # Make kx-array explicitly into a row-vector
        self.kz = np.sqrt(self.kwav**2 - self.kx**2 - self.ky**2)                                   # 2D array representing kz-space. Derived from condition for monochromaticity. Note: fftshifted to match P in step method.
        self.zres = 3000                                                                            # z-resolution in mm: 3000 for most; 50 for transverse profile as function of z. 
        self.U = np.ones((len(self.x), len(self.y)))                                                # 2D array representing plane wave. 

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
        Uout = Uin *  mask       
        self.U = Uout
        return self

    def step(self, D): # Propagate input Wave object over distance, D; return Wave object. Fourier algorithm. 
        Pin = fft2(self.U)                          # 2D FFT of amplitude distribution, U, gives spatial frequency distribution, Pin at initial position, z.
        Pout = Pin * np.exp(1j * self.kz * D)       # Multiply Spatial frequency distribution by phase-factor corresponding to propagation through distance, D, to give new spatial frequency distribution, Pout. 
        Uout = ifft2(Pout)                          # 2D IFFT of spatial frequency distribution gives amplitude distribution, Uout, at plane z = z + D
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

    def lens(self,f): # Lens element of focal length, f. 
        Uin = self.U
        Uout = Uin * np.exp(-1j * self.kwav * (self.x**2 + self.y**2) / (2 * f))
        self.U = Uout
        return self

    def plotx2D(self,n=1): # Plot spatial distribution in x-space.
        xplot = self.x
        yplot = self.y
        Uplot = abs(self.U)
        #plt.figure(n)
        plt.figure(n, figsize=(5, 5), dpi=120)
        plt.pcolormesh(xplot,yplot,Uplot, cmap = 'gray') # plot vs x and y
        axes = plt.gca()
        axes.set_xlim([-0.5, 0.5])
        axes.set_ylim([-0.5, 0.5])
        axes.set_xlabel('x / mm')
        axes.set_ylabel('y / mm')
        plt.title('Spatial distribution')

    def plotk2D(self,n=2): # Plot spatial frequency distribution in kx-space. 
        kxplot = fftshift(self.kx)
        kyplot = fftshift(self.ky)
        P = fft2(self.U)
        Pshift = fftshift(P)
        Pplot = abs(Pshift)
        #plt.figure(n)
        plt.figure(n, figsize=(5, 5), dpi=120)
        plt.pcolormesh(kxplot,kyplot,Pplot, cmap = 'gray')
        axes = plt.gca()
        axes.set_xlim([-30, 30])
        axes.set_ylim([-30, 30])
        axes.set_xlabel('k_x / mm^-1')
        axes.set_ylabel('k_y / mm^-1')
        plt.title('Spatial frequency distribution')

def system(): # Runs series of methods corresponding to wave propagation through various elements in system. 
    U = Wave()
    #U = U.aperture(0.06)
    U = U.supergaussian(0.03,10)
    U.plotx2D(1)
    U = U.propagate(S1)
    U = U.lens(F1)
    U = U.propagate(S1)
    U.plotk2D(2)
    U = U.propagate(S1)
    U = U.lens(F1)
    U = U.propagate(S1)
    U.plotx2D(3) 

def main():
    system()
    plt.show()
    
if __name__ == "__main__":
    main()

# Redundant or stored code
# ------------------------
# ogrid to generate x,y arrays
'''
yres = xres
self.y = np.linspace(-self.W/2,self.W/2,self.N)
self.x, self.y = np.ogrid[0:(self.N),0:(self.N)]
self.x = np.arange[0:(self.N)]
self.x = xres * (self.x - self.N / 2) + xres / 2
self.y = yres * (self.y - self.N / 2) + yres / 2   
'''
# definition of y and ky
'''
self.kyneg = np.linspace(-(np.pi * self.N)/self.W, -(2 * np.pi)/self.W, int(self.N/2))   # 1D array representing kx-space from max -ve value to min -ve value
self.kypos = np.linspace((2 * np.pi)/self.W, (np.pi * self.N)/self.W, int(self.N/2))     # 1D array representing kx-space from min +ve value to max +ve value
self.ky = np.concatenate((self.kypos,self.kyneg), axis = 0)                               # 1D array representing kx-space. Order of values matches that spatial frequency distribution derived from FFT of amlitude distribution 
'''
# fftshift of  kx 'by hand'
'''
self.kxneg = np.linspace(-(np.pi * self.N)/self.W, -(2 * np.pi)/self.W, int(self.N/2))   # 1D array representing kx-space from max -ve value to min -ve value
self.kxpos = np.linspace((2 * np.pi)/self.W, (np.pi * self.N)/self.W, int(self.N/2))     # 1D array representing kx-space from min +ve value to max +ve value
self.kx = np.concatenate((self.kxpos,self.kxneg), axis = 0)                               # 1D array representing kx-space. Order of values matches that spatial frequency distribution derived from FFT of amlitude distribution   
'''
# Mask
'''
mask = np.concatenate((np.zeros(int(0.475*R)),np.ones(int(0.01 * R)),np.zeros(int(0.03 * R)),np.ones(int(0.01 * R)),np.zeros(int(0.475 * R))), axis = 0)    # double slit
mask = np.concatenate((np.zeros(int(0.25*self.N)),np.ones(int(0.5 * self.N)),np.zeros(int(0.25 * self.N))), axis = 0)                                       # single slit
'''
# tilt
'''
def tilt(self,angle): # Applies linear phase-gradient, simulating effect of tilting mirror. Input angle in mrad. 
    Uin = self.U
    a = angle / 1000
    Uout = Uin * np.exp(-1j * self.kwav * self.x * np.sin(a))
    self.U = Uout
    return self
'''
# mirror
'''
def mirror(self,R): # Mirror element of radius of curvature, R. 
    Uin = self.U
    Uout = Uin * np.exp(-1j * self.kwav * self.x**2 / R)
    self.U = Uout
    return self
'''