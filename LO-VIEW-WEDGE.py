import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as fit
from scipy.fft import fft #as fft
from scipy.fft import ifft #as ifft

# Program description
'''
Uses Fourier method of Sziklas and Seigman (1975) [see also Leavey and Courtial, Young TIM User Guide (2017).] for beam propagation.
Calculation of:
1) the effect of transverse displacement of septum window in LO beam path, with a notional focal length of 1 km (due to imperfect polish), upon the position and direction of the beam at the OMC waist.
'''

# General comment: 
# Erroneous results can occur due to poor fit in k-space. If this is the case, use initial width greater than expected width in Gaussfit. 
# Beam profile requires initial width to be smaller than expected width. 

# Inputs
wav = 1.064e-3                                                      # wavelength in mm
z0 = -3330.5                                                        # input waist location
b0 = 701.1                                                          # input Rayleigh range
w0 = np.sqrt(b0 * wav / np.pi)                                      # input waist size - specified by Rayleigh range
x0 = 0.0                                                            # initial offset in mm
a0 = 0.0                                                            # initial angle in mrad
space_0 = 1356                                                      # BHDL1 - LO-VIEW
space_1 = 2220                                                      # LO-VIEW - OM1
space_2 = 1590                                                      # OM1 - OM2
space_3 = 1220                                                      # OM2 - OMC
alpha = 0.0059                                                      # Deflection imparted by wedged substrate with wedge angle of 0.75˚
R1 = 1000 * 5.709                                                   # OM1
R2 = 1000 * 2.360                                                   # OM2

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
        U0 = U0 * np.exp(-1j * self.kwav * self.x * np.sin(a0))                 # Tilt beam by initial angle, a0
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
        Uout = Uin * np.exp(-1j * self.kwav * self.x * np.sin(a))
        self.U = Uout
        return self

    def lens(self,f,x0): # Lens element of focal length, f, centre offset from beam by x0. 
        Uin = self.U
        Uout = Uin * np.exp(-1j * self.kwav * (self.x - x0)**2 / (2 * f))
        self.U = Uout
        return self

    def wedge(self,alpha,x0): # Wedge imparting deflection, alpha to beam, centre offset from beam by x0. 
        Uin = self.U
        Uout = Uin * np.exp(-1j * self.kwav * (self.x - x0) * np.sin(alpha))
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
            plt.plot(self.x,UFit,'-', label = count)
            axes = plt.gca()
            axes.set_xlim([-1, 1])
            axes.set_ylim([0, 1.1])
            plt.grid(which = 'major', axis = 'both')
            plt.xlabel('x / mm')
            plt.ylabel('Normalised amplitude distribution')
            #plt.legend()
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
            #plt.legend()
            plt.tight_layout()
        return kparams

def Gaussian(space, offset, height, width): # Defines Gaussian function for fitting; space is a 1D array. 
    return height * np.exp(-((space-offset)/width)**2)

def Gaussfit(space,Array,init_width=30):# Fit Gaussian to magnitude of Amplitude array. Return fit parameters. 
    init_params = [0.1,1,init_width]    # Use initial width parameter smaller than expected width for beam profile. Otherwise, use initial width parameter larger than expected width - avoids bad fit in k-space. 
    est_params, est_err = fit(Gaussian, space, Array, p0 = init_params)
    return est_params # [offset, amplitude, width]

def beam_profile(): 
    # Runs series of methods corresponding to propagation of beam through various elements in system. 
    # Calculate beam waist along the way ('True' condition passed to propagate). 
    U = Beam(w0,z0)
    U = U.propagate(space_0,True)
    U = U.wedge(alpha,0)
    U = U.wedge(-alpha,0)
    U = U.propagate(space_1,True)
    U = U.mirror(R1)
    U = U.propagate(space_2,True)
    U = U.mirror(R2)
    U = U.propagate(space_3,True)
    U = U.propagate(3500,True)
    width_plot(U.z,U.w)

def width_plot(distance_list,width_list,n=3): # Plots beam profile for a given waist array. 
    zplot = 0.001 * np.asarray(distance_list)
    wplot = np.asarray(width_list)
    plt.figure(figsize=(9, 7), dpi=120)
    plt.plot(zplot,wplot, linewidth = 3)
    axes = plt.gca()
    axes.set_xlim([0, 10])
    axes.set_ylim(0.0,2.6)
    axes.set_xticks(np.linspace(0,10,11))
    axes.set_yticks(np.linspace(0.0,2.6,14))
    plt.grid(which = 'both', axis = 'both', linestyle = '--')
    axes.set_xlabel('Distance from BHDL1 / m')
    axes.set_ylabel('Beam size / mm')
    axes.vlines(x = 0.001 * space_0, ymin = 0, ymax = 120,\
    linewidth = 2,color = 0.75*np.array([1,0.25,0.25]),linestyles = 'dashed',label = 'Viewport')
    #axes.vlines(x = 0.001 * (space_0 + space_1), ymin = 0, ymax = 120,\
    #linewidth = 2,color = 0.75*np.array([0.25,1,0.25]),linestyles = 'dashed',label = 'OM1') 
    #axes.vlines(x = 0.001 * (space_0 + space_1 + space_2), ymin = 0, ymax = 120,\
    #linewidth = 2,color = 0.75*np.array([1,1,0.25]),linestyles = 'dashed',label = 'OM2') 
    axes.vlines(x = 0.001 * (space_0 + space_1 + space_2 + space_3), ymin = 0, ymax = 120,\
    linewidth = 2,color = 0.75*np.array([0.25,0.25,0.25]),linestyles = 'dashed',label = 'OMC')
    plt.legend()
    #plt.title('')
    plt.tight_layout()

def L1_displ(x0): 
    # Runs series of methods corresponding to propagation of beam through various elements in system. Fixed spacings, defined in global variables. 
    # L1 displaced by x0. Returns, x and k offsets at OMC waist. 
    U = Beam(w0,z0)
    U = U.propagate(space_0,True)
    U = U.wedge(alpha,x0)
    #U = U.wedge(-alpha,0)
    U = U.propagate(space_1,True)
    U = U.mirror(R1)
    U = U.propagate(space_2,True)
    U = U.mirror(R2)
    U = U.propagate(space_3,True)
    xparams = U.amp_fit()
    Dx = xparams[0] #/ abs(xparams[2]) # normalise offset to width in x-space
    kparams = U.freq_fit()
    Dk = kparams[0] #/ abs(kparams[2]) # normalise offset to width in k-space
    Beta = 1000 * np.arctan(Dk / np.sqrt(U.kwav**2 - Dk**2))
    return (Dx, Beta)

def L1_test(): # Displace L1 by equal positive and negative amounts. Return x and k offsets at OMC waist.
    x1 = []
    b1 = []
    displ = np.linspace(-1e3, 1e3, 2)
    for i in range(len(displ)):
        Dx, Beta = L1_displ(displ[i])
        x1.append(Dx)
        b1.append(Beta)
    return (x1,b1)

def L1_dep():  # Calculates orthogonality between mirrors: angle between unit vectors corresponding to effect of mirror tilt in x-k space. 
    Mtest = L1_test()
    xb_plot(Mtest[0],Mtest[1])

def xb_plot(x1,b1,n=4): # Plots displacement in x-k space caused when lens, L1 displaced.
    plt.figure(n, figsize=(6, 5.5), dpi=120)
    plt.plot(x1, b1)                                                
    plt.title('L1 displaced by +/- 1 m')         
    axes = plt.gca()
    plt.xlabel('Change in position at OMC waist  / mm')
    plt.ylabel('Change in direction at OMC waist / mrad')
    textstr = 'Sensitivity:\n+%.12f mm / m\n %.12f mrad / m' % ((x1[1] - x1[0])/2,(b1[1] - b1[0])/2)
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    axes.text(0.5, 0.98, textstr, transform=axes.transAxes, fontsize=10, verticalalignment='top', bbox=props)
    plt.tight_layout()

def main():
    #beam_profile()
    #print(L1_displ(10))
    #L1_test()
    L1_dep()
    plt.show()
    
if __name__ == "__main__":
    main()