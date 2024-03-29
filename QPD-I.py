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
wav = 1.064e-3                                                      # wavelength in mm
# Beam Parameter at SR2
z0 = -20536                                                         # input waist location
b = 2090                                                            # input Rayleigh range
w0 = np.sqrt(b * wav / np.pi)                                       # input waist size - specified by Rayleigh range
x0 = 0.0                                                            # initial offset in mm
a0 = 0.0                                                            # initial angle in mrad
L0_OM0 = 1152
OM0_L1 = 660
space_0 = 15756                                                     # SR2 - SRM
space_1 = 1193 #1120                                                      # SRM - L0
space_2 = 1271 #1152                                                      # L0 - OM0
space_3 = 458 #660                                                       # OM0 - L1
space_4 = 276 #380                                                       # L1 - QPD
z_OM0 = (space_0 + space_1 + space_2) 
z_L1 = (space_0 + space_1 + space_2 + space_3)
z_QPD = (space_0 + space_1 + space_2 + space_3 + space_4)
n1064 = 1.44963
R_SRM = 5694
F_SRM = R_SRM / (n1064 - 1)
F_L0 = -13100                                                         # OFI lens
F_L1 = 525 #750                                                              # L1

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
        W = 1000                                                                 # Width of window in mm
        xres = 0.05                                                             # 1D array representing kz-space. Derived from condition for monochromaticity.
        N = int(W / xres)                                                       # Number of x-bins (keeps resolution the same)  
        self.x = np.linspace(-W/2, W/2, N)                                      # 1D array representing x-space
        self.kneg = np.linspace(-(np.pi * N)/W, -(2 * np.pi)/W, int(N/2))       # 1D array representing kx-space from max -ve value to min -ve value
        self.kpos = np.linspace((2 * np.pi)/W, (np.pi * N)/W, int(N/2))         # 1D array representing kx-space from min +ve value to max +ve value
        self.kx = np.concatenate((self.kpos,self.kneg), axis = 0)               # 1D array representing kx-space. Order of values matches that spatial frequency distribution derived from FFT of amlitude distribution
        self.kwav = 2 * np.pi / wav                                             # Magnitude of wave-vector
        self.kz = np.sqrt(self.kwav**2 - self.kx**2)                            # 1D array representing kz-space. Derived from condition for monochromaticity. 
        self.zres = 20000                                                        # z-resolution in mm: 3000 for most; 50 for beam profile. 
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
        axes.set_xlim([-20, 20])
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
            axes.set_xlim([-20, 20])
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
    U = U.propagate(space_0, True)
    U = U.lens(F_SRM)
    U = U.propagate(space_1, True)
    U = U.lens(F_L0)
    U = U.propagate(space_2, True)
    U = U.propagate(space_3, True)
    U = U.lens(F_L1)
    U = U.propagate(22000 - (space_0 + space_1 + space_2 + space_3),True)
    width_plot(U.z,U.w)

def width_plot(distance_list,width_list,n=3): # Plots beam profile for a given waist array. 
    zplot = np.asarray(distance_list) - z_OM0
    wplot = np.asarray(width_list)
    plt.figure(figsize=(6, 5), dpi=120)
    plt.plot(zplot,wplot, linewidth = 3)
    axes = plt.gca()
    #axes.set_xlim([0, 20])
    #axes.set_ylim(0.0,10)
    #axes.set_xticks(np.linspace(0,20,11))
    #axes.set_yticks(np.linspace(0.0,10,11))
    axes.set_xlim([0,2000])
    axes.set_ylim(0.0,2)
    axes.set_xticks(np.linspace(0,2000,11))
    axes.set_yticks(np.linspace(0.0,2,11))
    axes.vlines(x = space_3, ymin = 0, ymax = 2,\
    linewidth = 2,color = 0.7*np.array([0,1,0]),label = 'Lens')
    axes.vlines(x = space_3 + space_4, ymin = 0, ymax = 2,\
    linewidth = 2,color = 0.7*np.array([1,0,0]),label = 'QPD')
    plt.grid(which = 'both', axis = 'both', linestyle = '--')
    axes.set_xlabel('Distance from OM0 / mm')
    axes.set_ylabel('Beam size / mm')
    #plt.title('Beam path from SR2 to QPD-I')
    plt.legend()
    plt.tight_layout()

def QPD_sense(x0,a0,var_space): 
    # Runs series of methods corresponding to propagation of beam through various elements in system. Fixed spacings, defined in global variables. 
    # Displacement and Direction errors applied at SRM. Returns, x offsets at WFS1 and WFS2. 
    U = Beam(w0,z0,x0,a0)
    U = U.propagate(space_0)
    U = U.lens(F_SRM)
    U = U.propagate(space_1)
    U = U.lens(F_L0)
    U = U.propagate(space_2)
    if var_space <= space_3:
        U = U.propagate(var_space)
    else:
        U = U.propagate(space_3)
        U = U.lens(F_L1)
        U = U.propagate(var_space - space_3)
    xparams = U.amp_fit()
    Dx = xparams[0] #/ abs(xparams[2]) # normalise offset to width in x-space
    return Dx

def sen_dep(): # Calculates x offset as a function of distance from L2 for pure displacement and direction errors
    s3 = np.linspace(0,2000,31)
    xoff = []
    for i in range(len(s3)):
        Dx = QPD_sense(0.0,0.1,int(s3[i]))
        xoff.append(Dx)
    sen_plot(s3,xoff)

def sen_plot(dist,xoff,n=6): # Plots x offset as function of distance from L2 for pure displacement and direction errors
    plt.figure(figsize=(6, 5), dpi=120)
    plt.plot(dist, xoff, color = 'blue')
    axes = plt.gca()
    axes.set_xlim([0, 2000])
    axes.set_ylim([-2, 2])
    axes.set_xticks(np.linspace(0,2000,11))
    axes.vlines(x = space_3, ymin = -2, ymax = 2,\
    linewidth = 2,color = 0.7*np.array([0,1,0]),label = 'Lens')
    axes.vlines(x = space_3 + space_4, ymin = -2, ymax = 2,\
    linewidth = 2,color = 0.7*np.array([1,0,0]),label = 'QPD')
    plt.title('Apply 100 microrad tilt to SR2')     
    plt.xlabel('distance from OM0 / mm')
    plt.ylabel('offset in x / mm')
    plt.grid(which = 'major', axis = 'both')
    plt.legend()
    plt.tight_layout()

def main():
    beam_profile()
    sen_dep()
    plt.show()
    
if __name__ == "__main__":
    main()

'''
def Mirror_corr(b,c): 
    # Runs series of methods corresponding to propagation of beam through various elements in system. Fixed spacings, defined in global variables. 
    # Mirrors M1 and M2 tilted by angles b and c. Returns, x and k offsets at OMC waist. 
    U = Beam(w0,z0)
    U = U.propagate(space_0, True)
    U = U.lens(F_SRM)
    U = U.propagate(space_1, True)
    U = U.lens(F_L0)
    U = U.propagate(space_2, True)
    U = U.lens(F_L1)
    U = U.propagate(space_3)
    xparams = U.amp_fit()
    Dx = xparams[0] / abs(xparams[2]) # normalise offset to width in x-space
    return Dx

def Mirror_test(): # Tilt mirrors in turn by equal positive and negative amounts. Return x and k offsets at OMC waist.
    x1 = []
    angle = np.linspace(-1.0, 1.0, 2)
    for i in range(len(angle)):
        Dx= Mirror_corr(angle[i],0)
        x1.append(Dx)
    return x1

def Mirror_dep():  # Calculates orthogonality between mirrors: angle between unit vectors corresponding to effect of mirror tilt in x-k space. 
    Mtest = Mirror_test()
    #print(Mtest)
'''