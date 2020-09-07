import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as fit
from scipy.fft import fft #as fft
from scipy.fft import ifft #as ifft

# Program description
'''
Uses Fourier method of Sziklas and Seigman (1975) [see also Leavey and Courtial, Young TIM User Guide (2017).] for beam propagation.
Calculation of:
1) waist size along beam path. (beam_profile)
2) effect of mirror tilt on position and direction of beam at waist. (simple_correction, simple_corr_dep)
3) effect of mirror tilt on position and direction of beam at waist as a function of mirror separation. (corr_dep)
4) effect of input error (in position and direction) on transverse position of beam as a function of distance from beam waist. (sen_dep)
'''

# General comment: 
# Erroneous results can occur due to poor fit in k-space. If this is the case, use initial width greater than expected width in Gaussfit. 
# Beam profile requires initial width to be smaller than expected width. 

# Inputs
x0 = 0.0                                                            # initial offset in mm
a0 = 0.0                                                            # initial angle in mrad
wav = 1.064e-3                                                      # wavelength in mm
kwav = 2 * np.pi / wav                                              # Magnitude of wave-vector in mm^-1
z0 = -3550                                                          # input waist location
b0 = 1090                                                           # input Rayleigh range
w0 = np.sqrt(b0 * wav / np.pi)                                      # input waist size - specified by Rayleigh range
# WFS42
z1 = 0.0                                                            # focus inbetween TTs
b1 = 261                                                            # Rayleigh Range at focus inbetween TTs
w1 = np.sqrt(b1 * wav / np.pi)                                      # input waist size - specified by Rayleigh range
space_0 = 1120                                                      # SRM - L0
space_1 = 2372                                                      # L0 - L1
space_2 = 195                                                       # L1 - TT1
space_3 = 385                                                       # TT1 - W1
space_4 = 180                                                       # W1 - TT2
space_5 = 425                                                       # TT2 - L2
space_6 = 175                                                       # L2 - L3
var_space = (120, 490)                                              # L3 - WFS1, L3 - WFS2)
space_7 = 305                                                       # L3 - W2
FI = -17100                                                         # L0
L1 = 1000 * (1 / 1.28)                                              # L1
L2 = 1000 * (1 / 3.485)                                             # L2
L3 = 1000 * (1 / (-0.645))                                          # L3

class Beam: 
    '''Represents Gaussian beam in x, propagating along z. 
    Attributes: 
    U: 1D array, Complex amplitude
    w: 1D array, width
    z: 1D array, z-co-ordinate
    x: 1D array, x-co-ordinate
    kneg/kpos: 1D arrays, negative/positive components of k-vactor along x-axis (spatial frequencies)
    kx: 1D array, component of k-vector along x-axis (spatial frequency)
    kz: 1D array, component of k-vector along z-axis
    kwav: constant, magnitude of wavevector
    zres: constant: resolution in z
    '''
    def __init__(self, *args): # Initialises amplitude array.
        x0 = 0
        a0 = 0
        W = 200                                                                 # Width of window in mm
        xres = 0.01                                                             # 1D array representing kz-space. Derived from condition for monochromaticity.
        N = int(W / xres)                                                       # Number of x-bins (keeps resolution the same)  
        self.x = np.linspace(-W/2, W/2, N)                                      # 1D array representing x-space
        self.kneg = np.linspace(-(np.pi * N)/W, -(2 * np.pi)/W, int(N/2))       # 1D array representing kx-space from max -ve value to min -ve value
        self.kpos = np.linspace((2 * np.pi)/W, (np.pi * N)/W, int(N/2))         # 1D array representing kx-space from min +ve value to max +ve value
        self.kx = np.concatenate((self.kpos,self.kneg), axis = 0)               # 1D array representing kx-space. Order of values matches that spatial frequency distribution derived from FFT of amlitude distribution
        self.kwav = 2 * np.pi / wav                                             # Magnitude of wave-vector
        self.kz = np.sqrt(self.kwav**2 - self.kx**2)                            # 1D array representing kz-space. Derived from condition for monochromaticity. 
        self.zres = 3000                                                        # z-resolution in mm: 3000 for most; 50 for beam profile. 
        if len(args) == 2:                                                      # Two arguments: Instantiates Beam object from waist size, w0, and distance to waist, z0. 
            w0 = args[0]
            z0 = args[1]
        elif len(args) == 4:                                                    # Four arguments: Instantiates Beam object from waist size, w0, distance to waist, z0, input offset, x0, and input angle, a0. 
            w0 = args[0]
            z0 = args[1]
            x0 = args[2]
            a0 = args[3]
        a0 = a0 / 1000                                                          # Converts input angle from mrad to rad
        q0 = z0 - 1j * np.pi * w0**2 / wav                                      # Input beam parameter
        U0 = (1/q0) * np.exp(1j * self.kwav * (self.x-x0)**2 / (2 * q0))        # Input array, offset by x0
        U0 = U0 * np.exp(1j * self.kwav * self.x * np.sin(a0))                 # Tilt beam by initial angle, a0
        self.U = U0                                                             # Initialise amplitude array
        self.w = [Gaussfit(self.x,abs(self.U),1)[2]]                            # Initialise width list
        self.z = [0]                                                            # Initialise z-position list.

    def step(self, D): # Propagate input Beam object over distance, D; return Beam object. Fourier algorithm. 
        Pin = fft(self.U)                       # FFT of amplitude distribution, U, gives spatial frequency distribution, Pin at initial position, z.
        Pout = Pin * np.exp(1j * self.kz * D)   # Multiply Spatial frequency distribution by phase-factor corresponding to propagation through distance, D, to give new spatial frequency distribution, Pout. 
        Uout = ifft(Pout)                       # IFFT of spatial frequency distribution gives amplitude distribution, Uout, at plane z = z + D
        self.U = Uout
        return self

    def propagate(self,distance,profile=False): # Propagate Beam object through distance with resolution, zres; return Beam object. 
        Uprev = self
        if profile:
            w = Uprev.w                 # unpack width_list
            z = Uprev.z                 # unpack z-position_list
            res = 50                    # Set res to 50 if generating plot of beam profile.
        else:
            res = self.zres             # Otherwise use global variable, zres
        num = distance // res           # number of steps: divide distance by resolution. 
        rem = distance % res            # remainder of division: final step size. If num = 0, i.e. zres > distance, single step taken, equal to distance. 
        for i in range(num):            # num steps
            Unext = Uprev.step(res)
            Uprev = Unext
            if profile:
                zprev = z[-1]
                z.append(zprev + res)   # Build up z-array as go along. 
                wnext = Gaussfit(Unext.x,abs(Unext.U),1)[2]
                w.append(wnext)
        Unext = Uprev.step(rem)         # Final step of size rem. 
        if profile:
            zprev = z[-1]
            z.append(zprev + rem) 
            wnext = Gaussfit(Unext.x,abs(Unext.U),1)[2]
            w.append(wnext)
            Unext.w = w
            Unext.z = z
        return Unext

    def tilt(self,angle): # Applies linear phase-gradient, simulating effect of tilting mirror. Input angle in mrad. 
        Uin = self.U
        a = angle / 1000
        Uout = Uin * np.exp(1j * self.kwav * self.x * np.sin(a))
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
        plt.plot(self.x,Uplot,'o', label = 'model data', markersize = 3)
        axes = plt.gca()
        axes.set_xlim([-2, 2])
        axes.set_ylim([0, 1.1])
        plt.grid(which = 'major', axis = 'both')
        plt.xlabel('x / mm')
        plt.ylabel('Normalised amplitude distribution')
        #plt.legend()
        plt.tight_layout()

    def amp_fit(self,plot=False,n=1): # Fit (and Plot) magnitude of Amplitude array in x-space. 
        Uplot = abs(self.U)/max(abs(self.U))
        xparams = Gaussfit(self.x,Uplot)
        UFit = Gaussian(self.x, xparams[0], xparams[1], xparams[2])
        if plot == True:
            plt.figure(n)
            plt.plot(self.x,Uplot,'o', label = 'model data', markersize = 3)
            plt.plot(self.x,UFit,'-', label = 'fit')
            axes = plt.gca()
            axes.set_xlim([-1, 1])
            axes.set_ylim([0, 1.1])
            plt.grid(which = 'major', axis = 'both')
            plt.xlabel('x / mm')
            plt.ylabel('Normalised amplitude distribution')
            plt.legend()
            plt.tight_layout()
        return xparams

    def freq_fit(self,plot=False,n=2): # Fit ( and plot) magnitude of Spatial frequency array in k-space. 
        P = fft(self.U)
        kplot = np.concatenate((self.kneg,self.kpos), axis = 0)     
        Pneg = P[int(len(P)/2):]
        Ppos = P[:int(len(P)/2)]
        Pswap = np.concatenate((Pneg,Ppos), axis = 0)
        Pabs = abs(Pswap)
        Pplot = Pabs/max(Pabs)
        kparams = Gaussfit(kplot,Pplot)
        PFit = Gaussian(kplot, kparams[0], kparams[1], kparams[2])
        if plot == True:
            plt.figure(n)
            plt.plot(kplot,Pplot,'o', label = 'model data', markersize = 3)
            plt.plot(kplot,PFit,'-', label = 'fit')
            axes = plt.gca()
            axes.set_xlim([-25, 25])
            axes.set_ylim([0, 1.1])
            plt.grid(which = 'major', axis = 'both')
            plt.xlabel('k_x / mm^-1')
            plt.ylabel('Normalised spatial frequency distribution')
            plt.legend()
            plt.tight_layout()
        return kparams

def Gaussian(space, offset, height, width): # Defines Gaussian function for fitting; space is a 1D array. 
    return height * np.exp(-((space-offset)/width)**2)

def Gaussfit(space,Array,init_width=30):# Fit Gaussian to magnitude of Amplitude array. Return fit parameters. 
    init_params = [0.1,1,init_width]    # Use initial width parameter smaller than expected width for beam profile. Otherwise, use initial width parameter larger than expected width - avoids bad fit in k-space. 
    est_params, est_err = fit(Gaussian, space, Array, p0 = init_params)
    return est_params # [offset, amplitude, width]

def TT_corr(b,c):   # Direction errors applied at mirrors. Returns, x and k offsets at W2.
    U = Beam(w0,z0)
    U = U.propagate(space_0)
    U = U.lens(FI)
    U = U.propagate(space_1)
    U = U.lens(L1)
    U = U.propagate(space_2)
    U = U.tilt(b)
    U = U.propagate(space_3)
    U = U.propagate(space_4)
    U = U.tilt(c)
    U = U.propagate(space_5)
    U = U.lens(L2)
    U = U.propagate(space_6)
    U = U.lens(L3)
    U = U.propagate(space_7)
    xparams = U.amp_fit()
    Dx = xparams[0] #/ abs(xparams[2]) # normalise offset to width in x-space
    kparams = U.freq_fit()
    Dk = kparams[0] #/ abs(kparams[2]) # normalise offset to width in k-space
    Da = 1000 * np.arctan(np.sqrt(1 / ((kwav / Dk)**2 - 1))) # angle in mrad
    return (Dx, Da)

def WFS_sense(a0,x0,space_7): # Displacement and Direction errors applied at W1. Returns x- and k- offsets at WFS or at W2. 
    U = Beam(w1,z1,x0,a0)
    U = U.propagate(space_4)
    U = U.propagate(space_5)
    U = U.lens(L2)
    U = U.propagate(space_6)
    U = U.lens(L3)
    U = U.propagate(space_7)
    xparams = U.amp_fit()
    Dx = xparams[0] #/ abs(xparams[2]) # normalise offset to width in x-space
    kparams = U.freq_fit()
    Dk = kparams[0] #/ abs(kparams[2]) # normalise offset to width in k-space
    Da = 1000 * np.arctan(np.sqrt(1 / ((kwav / Dk)**2 - 1))) # angle in mrad
    return Dx, Da

def TT_test(): # Direction error applied at mirrors. Return x and k offsets at W2.
    x1, a1 = TT_corr(1.0,0)
    x2, a2 = TT_corr(0,1.0)
    A = np.array([[x1, x2], [a1, a2]])
    return A

def WFS_test(): # Apply +/- displacement/direction errors at SRM and return x-displacement at WFS1 and WFS2.
    x1_displ = WFS_sense(1.0,0.0,var_space[0])[0]
    x2_displ = WFS_sense(1.0,0.0,var_space[1])[0]
    x1_direc = WFS_sense(0.0,1.0,var_space[0])[0]
    x2_direc = WFS_sense(0.0,1.0,var_space[1])[0]
    B = np.array([[x1_displ, x1_direc], [x2_displ, x2_direc]])
    return B

def W1W2_test(): # Apply +/- displacement/direction errors at SRM and return x-displacement at WFS1 and WFS2.
    x_displ, a_displ = WFS_sense(1.0,0.0,space_7)
    x_direc, a_direc = WFS_sense(0.0,1.0,space_7)
    C = np.array([[x_displ, x_direc], [a_displ, a_direc]])
    return C

def servo():
    A = TT_test()
    B = WFS_test()
    C = W1W2_test()
    IA = np.linalg.inv(A)
    IB = np.linalg.inv(B)
    IC = np.linalg.inv(C)
    WFS_err = np.array([[1],[0]])
    print(WFS_err)
    TT = IA @ C @ IB @ -WFS_err
    print(TT)
    WFS_corr = B @ IC @ A @ TT
    print(WFS_corr)
    WFS_res = WFS_err + WFS_corr
    print(WFS_res)

def main():
    TT_test()
    WFS_test()
    W1W2_test()
    servo()
    #plt.show()
    
if __name__ == "__main__":
    main()