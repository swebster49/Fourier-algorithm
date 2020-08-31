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
z0 = -3550                                                          # input waist location
b = 1090                                                            # input Rayleigh range
w0 = np.sqrt(b * wav / np.pi)                                       # input waist size - specified by Rayleigh range
z1 = 0.0                                                            # focus inbetween TTs
b1 = 1875                                                           # Rayleigh Range at focus after OMS
w1 = np.sqrt(b1 * wav / np.pi)                                      # input waist size - specified by Rayleigh range
z2 = 0.0                                                            # focus inbetween TTs
b2 = 1875                                                           # Rayleigh Range at focus after OM1
w2 = np.sqrt(b2 * wav / np.pi)                                      # input waist size - specified by Rayleigh range
z3 = 0.0                                                            # focus inbetween TTs
b3 = 696                                                            # Rayleigh Range at focus after OM2
w3 = np.sqrt(b3 * wav / np.pi)                                      # input waist size - specified by Rayleigh range
x0 = 0.0                                                            # initial offset in mm
a0 = 0.0                                                            # initial angle in mrad
space_0 = 1120                                                      # SRM - OFI
space_1 = 2282                                                      # OFI - OM*S
space_2 = 438                                                       # OM*S - W1
space_3 = 802                                                       # W1 - OM*1
space_4 = 230                                                       # OM*1 = W2
space_5 = 1360                                                      # W2 - OM*2
space_6 = 1228                                                      # OM*2 - W2
space_7 = 677                                                       # W3 - L1
space_8 = 320                                                       # L1 - L2
var_space = (210, 580)                                              # L2 - WFS1, L2 - WFS2)
space_9 = 395                                                       # L2 - waist between WFS
FI = -17100                                                         # OFI lens
R1 = 5700                                                           # ROC of OM*1
R2 = 2300                                                           # ROC of OM*2
L1 = 1000 * (1 / 1.220)                                             # L1 before WFS36
L2 = 1000 * (1 / 1.755)                                              # L2 before WFS36

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
            axes.set_xlim([-2, 2])
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
    #U = Beam(w0,z0)
    #U = U.propagate(space_0, True)
    #U = U.lens(FI)
    #U = U.propagate(space_1, True)
    #U = U.propagate(space_2, True)
    ##U = Beam(w1,z1)
    #U = U.propagate(space_3, True)
    #U = U.mirror(R1)
    #U = U.propagate(space_4, True)
    U = Beam(w2,z2)
    U = U.propagate(space_5, True)
    U = U.mirror(R2)
    U = U.propagate(space_6, True)
    #U = Beam(w3,z3)
    U = U.propagate(space_7, True)
    U = U.lens(L1)
    U = U.propagate(space_8, True)
    U = U.lens(L2)
    U = U.propagate(10000 - (space_0 + space_1 + space_2 + space_3 + space_4 + space_5),True)
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
    axes.set_xlabel('Distance from SRM / m')
    axes.set_ylabel('Beam size / mm')
    plt.title('Layout 5f + WFS36. Beam profile calculated with Fourier algorithm.')
    plt.tight_layout()

def Mirror_corr(b,c): 
    # Runs series of methods corresponding to propagation of beam through various elements in system. Fixed spacings, defined in global variables. 
    # Mirrors M1 and M2 tilted by angles b and c. Returns, x and k offsets at OMC waist. 
    U = Beam(w0,z0)
    U = U.propagate(space_0)
    U = U.lens(FI)
    U = U.propagate(space_1)
    U = U.tilt(b)
    U = U.propagate(space_2)
    U = U.propagate(space_3)
    U = U.mirror(R1)
    U = U.propagate(space_4)
    U = U.propagate(space_5)
    U = U.mirror(R2)
    U = U.tilt(c)
    U = U.propagate(space_6)
    U = U.propagate(space_7)
    U = U.lens(L1)
    U = U.propagate(space_8)
    U = U.lens(L2)
    U = U.propagate(space_9)
    xparams = U.amp_fit()
    Dx = xparams[0] / abs(xparams[2]) # normalise offset to width in x-space
    kparams = U.freq_fit()
    Dk = kparams[0] / abs(kparams[2]) # normalise offset to width in k-space
    return (Dx, Dk)

def Mirror_test(): # Tilt mirrors in turn by equal positive and negative amounts. Return x and k offsets at OMC waist.
    x1 = []
    k1 = []
    x2 = []
    k2 = []
    angle = np.linspace(-1.0, 1.0, 2)
    for i in range(len(angle)):
        Dx, Dk = Mirror_corr(angle[i],0)
        x1.append(Dx)
        k1.append(Dk)
    for i in range(len(angle)):
        Dx, Dk = Mirror_corr(0,angle[i])
        x2.append(Dx)
        k2.append(Dk)
    return (x1,k1,x2,k2)

def Mirror_dep():  # Calculates orthogonality between mirrors: angle between unit vectors corresponding to effect of mirror tilt in x-k space. 
    Mtest = Mirror_test()
    #orthogonality(Mtest[0],Mtest[1],Mtest[2],Mtest[3])
    xk_plot(Mtest[0],Mtest[1],Mtest[2],Mtest[3])

def orthogonality(x1,k1,x2,k2): # Calculates phase-separation between two mirrors which, when tilted, cause displacements of (x*,k*) at the OMC waist .
    v1 = np.array([[x1[-1] - x1[0], k1[-1] - k1[0]]])
    v2 = np.array([[x2[-1] - x2[0]], [k2[-1] - k2[0]]])
    v1_norm = v1 / np.sqrt(v1[0,0]**2 + v1[0,1]**2)
    v2_norm = v2 / np.sqrt(v2[0,0]**2 + v2[1,0]**2)
    dot_product = np.dot(v1_norm,v2_norm)
    phi_ASC = (180 / np.pi) * np.arccos(dot_product)
    #print('%.1f' % (phi_ASC[0,0],))
    return phi_ASC[0,0]

def xk_plot(x1,k1,x2,k2,n=4): # Plots displacement in x-k space caused when mirrors, M1 and M2 are tilted.
    plt.figure(n, figsize=(6, 5.5), dpi=120)
    plt.plot(x1, k1, label = 'M1')                        
    plt.plot(x2, k2, label = 'M2')                         
    plt.title('Mirrors tilted through +/- 1 mrad')         
    plt.legend()
    axes = plt.gca()
    axes.set_xlim([-3, 3])
    axes.set_ylim([-3, 3])
    plt.xlabel('offset in x at waist between WFS / 1/e^2 radius in x-space')
    plt.ylabel('offset in k_x at waist between WFS / 1/e^2 radius in k-space')
    textstr = 'Orthogonality: %.1f˚' % (WFS_orthogonality(x1,k1,x2,k2),)
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    axes.text(0.02, 0.98, textstr, transform=axes.transAxes, fontsize=10,verticalalignment='top', bbox=props)
    plt.tight_layout()

def WFS_sense(x0,a0,space_9): 
    # Runs series of methods corresponding to propagation of beam through various elements in system. Fixed spacings, defined in global variables. 
    # Displacement and Direction errors applied at SRM. Returns, x offsets at WFS1 and WFS2. 
    #U = Beam(w0,z0,x0,a0)
    #U = U.propagate(space_0)
    #U = U.lens(FI)
    #U = U.propagate(space_1)
    #U = U.propagate(space_2)
    #U = Beam(w1,z1,x0,a0)
    #U = U.propagate(space_3)
    #U = U.mirror(R1)
    #U = U.propagate(space_4)
    #U = Beam(w2,z2,x0,a0)
    #U = U.propagate(space_5)
    #U = U.mirror(R2)
    #U = U.propagate(space_6)
    U = Beam(w3,z3,x0,a0)
    U = U.propagate(space_7)
    U = U.lens(L1)
    U = U.propagate(space_8)
    U = U.lens(L2)
    U = U.propagate(space_9)
    xparams = U.amp_fit()
    Dx = xparams[0] #/ abs(xparams[2]) # normalise offset to width in x-space
    return Dx

def WFS_test(var_space): # Apply +/- displacement/direction errors at SRM and return x-displacement at WFS1 and WFS2.
    x1_displ = []
    x2_displ = []
    x1_direc = []
    x2_direc = []
    displ = np.linspace(-1.0, 1.0, 2)
    direc = np.linspace(-1.0, 1.0, 2)
    for i in range(len(displ)):
        Dx1 = WFS_sense(displ[i],0.0,var_space[0])
        Dx2 = WFS_sense(displ[i],0.0,var_space[1])
        x1_displ.append(Dx1)
        x2_displ.append(Dx2)
    for i in range(len(direc)):
        Dx1 = WFS_sense(0.0,direc[i],var_space[0])
        Dx2 = WFS_sense(0.0,direc[i],var_space[1])
        x1_direc.append(Dx1)
        x2_direc.append(Dx2)
    return (x1_displ,x2_displ,x1_direc,x2_direc)

def WFS_dep():  # Calculates x offsets at WFS1 and WFS2 for +/- dislpacement/direction errors at SRM.
        Stest = WFS_test(var_space)
        #WFS_orthogonality(Stest[0],Stest[1],Stest[2],Stest[3])
        WFS_plot(Stest[0],Stest[1],Stest[2],Stest[3])

def WFS_orthogonality(x1_displ,x2_displ,x1_direc,x2_direc): # Calculates phase-separation between WFS which sense displacement and direction errors upstream.
    v1 = np.array([[x1_displ[-1] - x1_displ[0], x2_displ[-1] - x2_displ[0]]])
    v2 = np.array([[x1_direc[-1] - x1_direc[0]], [x2_direc[-1] - x2_direc[0]]])
    v1_norm = v1 / np.sqrt(v1[0,0]**2 + v1[0,1]**2)
    v2_norm = v2 / np.sqrt(v2[0,0]**2 + v2[1,0]**2)
    dot_product = np.dot(v1_norm,v2_norm)
    phi_ASC = (180 / np.pi) * np.arccos(dot_product)
    #phi_ASC = 90 - phi_ASC % 90
    #print('%.1f' % (phi_ASC[0,0],))
    return phi_ASC[0,0]

def WFS_plot(x1_displ,x2_displ,x1_direc,x2_direc,n=4): # Plots displacement at WFS1 vs displacement at WFS2 when displacement and direction errors are applied at the SRM. 
    plt.figure(n, figsize=(6, 5.5), dpi=120)
    plt.plot(x1_displ, x2_displ, color = 'blue', label = 'displacement')                        
    plt.plot(x1_direc, x2_direc, color = 'orange', label = 'direction')                         
    plt.title('Apply Pure Displacement and Direction errors at Waist 3')         
    plt.legend(loc = 'upper right')
    axes = plt.gca()
    axes.set_xlim([-3.5, 3.5])
    axes.set_ylim([-3.5, 3.5])
    plt.xlabel('offset in x at WFS1 / 1/e^2 radius')
    plt.ylabel('offset in x at WFS2 / 1/e^2 radius')
    textstr = 'Orthogonality: %.1f˚' % (WFS_orthogonality(x1_displ,x2_displ,x1_direc,x2_direc),)
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    axes.text(0.02, 0.98, textstr, transform=axes.transAxes, fontsize=10,verticalalignment='top', bbox=props)
    plt.tight_layout()

def sen_dep(): # Calculates x offset as a function of distance from L2 for pure displacement and direction errors
    s9 = np.linspace(0,1000,21)
    Dx_displ = []
    Dx_direc = []
    for i in range(len(s9)):
        Dx = WFS_sense(1.0,0.0,int(s9[i]))
        Dx_displ.append(Dx)
    for i in range(len(s9)):
        Dx = WFS_sense(0.0,1.0,int(s9[i]))
        Dx_direc.append(Dx)
    sen_plot(s9,Dx_displ,Dx_direc)

def sen_plot(dist,Dx_displ,Dx_direc,n=6): # Plots x offset as function of distance from L2 for pure displacement and direction errors
    fig, ax1 = plt.subplots()
    ax1.plot(dist,Dx_displ, color = 'blue', label = 'displacement')
    ax1.plot(dist,Dx_direc, color = 'orange', label = 'direction')
    ax1.set_xlim([0, 1000])
    #ax1.set_ylim([-3, 3])
    ax1.vlines(x = var_space[0], ymin = -3, ymax = +3,\
    linewidth = 2,color = 0.5*np.array([1,1,1]),linestyles = 'dashed',label = 'WFS')
    ax1.vlines(x = var_space[1], ymin = -3, ymax = +3,\
    linewidth = 2,color = 0.5*np.array([1,1,1]),linestyles = 'dashed')#,label = 'WFS2')
    ax1.grid(which = 'major', axis = 'both')
    plt.title('Apply pure displacement and direction errors at W3')     
    ax1.set_xlabel('distance from L2 / mm')
    ax1.set_ylabel('offset in x / mm')
    fig.legend(loc = 'upper right', bbox_to_anchor=(0.95, 0.92))
    fig.tight_layout()

def main():
    #beam_profile()
    #Mirror_dep()
    #WFS_dep()
    sen_dep()
    plt.show()
    
if __name__ == "__main__":
    main()