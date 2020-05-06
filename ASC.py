import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as fit
from scipy.fft import fft #as fft
from scipy.fft import ifft #as ifft

# Program description
'''
Uses Fourier method of Sziklas and Seigman (1975) [see also Leavey and Countial, Young TIM User Guide (2017).] for beam propagation.
Calculation of:
1) waist size along beam path. (beam_profile)
2) effect of mirror tilt on position and direction of beam at waist. (simple_correction, simple_corr_dep)
3) effect of mirror tilt on position and direction of beam at waist as a function of mirror separation. (corr_dep)
4) effect of input error (in position and direction) on transverse position of beam as a function of distance from beam waist. (sen_dep)
'''

# General comment: 
# Erroneous results can occur due to poor fit in k-space. If this is the case, use initial width greater than expected width in Gaussfit. 
# Beam profile requires initial width to be smaller than expected width. 

# Globals
wav = 1.064e-3                                                      # wavelength in mm
kwav = 2 * np.pi / wav                                              # magnitude of wave-vector
W = 200                                                             # width of window in mm
xres = 0.01                                                         # x resolution in mm
N = int(W / xres)                                                   # number of x-bins (keeps resolution the same)
x = np.linspace(-W/2, W/2, N)                                       # 1D array representing x-space
kneg = np.linspace(-(np.pi * N)/W, -(2 * np.pi)/W, int(N/2))        # 1D array representing kx-space from max -ve value to min -ve value
kpos = np.linspace((2 * np.pi)/W, (np.pi * N)/W, int(N/2))          # 1D array representing kx-space from min +ve value to max +ve value
kx = np.concatenate((kpos,kneg), axis = 0)                          # 1D array representing kx-space. Order of values matches that spatial frequency distribution derived from FFT of amlitude distribution
kz = np.sqrt(kwav**2 - kx**2)                                       # 1D array representing kz-space. Derived from condition for monochromaticity. 
zres = 3000 #50                                                     # z resolution in mm: 3000 for most; 50 for beam profile. 
z = [0]                                                             # initialise z array
w = []                                                              # initialise waist array
b = 1090                                                            # input Rayleigh range
a0 = 0                                                              # initial angle in mrad
x0 = 0                                                              # initial offset in mm
w0 = np.sqrt(b * wav / np.pi)                                       # input waist size - specified by Rayleigh range
#w0 = 0.6076                                                        # input waist size - specified directly
z0 = -3550                                                          # input waist location
wt = 0.485                                                          # target waist
bt = int(np.pi * wt**2 / wav)                                       # target Rayleigh range
space_0 = 1120                                                      # SRM - OFI
space_1 = 2320                                                      # OFI - OM*S
space_2 = 2015                                                      # OM*S - OM*1
space_3 = 1770                                                      # OM*1 - OM*2
space_4 = 1210                                                      # OM*2 - OMC*
FI = -9000                                                          # OFI lens
R1 = 5500                                                           # ROC of OM*1
R2 = 2700                                                           # ROC of OM*2

def initialise(w0,z0, x0 = 0, a0 = 0):  # Calculates beam parameter and input array given waist and waist position; plus initial offset and angle. 
    a0 = a0 / 1000                                                  # converts angle from mrad to rad
    q0 = z0 - 1j * np.pi * w0**2 / wav                              # Input beam parameter
    U0 = (1/q0) * np.exp(1j * kwav * (x-x0)**2 / (2 * q0))          # Input array, offset by x0
    U0 = U0 * np.exp(-1j * kwav * x * np.sin(a0))                   # Tilt beam by initial angle, a0
    U = Amplitude(U0)                                               # Input Amplitude object
    w.append(Gaussfit(x,U.Uabs,1)[2])                                 # First element in beam waist array
    return U

def Gaussian(space, offset, height, width): # Defines Gaussian function for fitting; space is a 1D array. 
    return height * np.exp(-((space-offset)/width)**2)

def Gaussfit(space,Array,init_width=30): # Fit Gaussian to magnitude of Amplitude array. Return fit parameters. 
    init_params = [0.1,1,init_width]    # Use initial width parameter smaller than expected width for beam profile. 
                                        # Otherwise, use initial width parameter larger than expected width - avoids bad fit in k-space. 
    est_params, est_err = fit(Gaussian, space, Array, p0 = init_params)
    return est_params # [offset, amplitude, width]

class Amplitude: # Represents the amplitude of a beam in plane z. Attributes: 1D array of complex numbers; 1D array of absolute values

    def __init__(self, complex_array): # Initialises amplitude array.
        self.U = complex_array
        self.Uabs = abs(self.U)

    def step(self, D): # Propagate input Amplitude object over distance, D; return Amplitude object. Fourier algorithm. 
        Pin = fft(self.U)                       # FFT of amplitude distribution, U, gives spatial frequency distribution, Pin at initial position, z.
        Pout = Pin * np.exp(1j * kz * D)        # Multiply Spatial frequency distribution by phase-factor corresponding to propagation through distance, D, to give new spatial frequency distribution, Pout. 
        Uout = ifft(Pout)                       # IFFT of spatial frequency distribution gives amplitude distribution, Uout, at plane z = z + D
        Ureturn = Amplitude(Uout)               # Return new amplitude distribution as amplitude object, Ureturn. 
        return Ureturn

    def propagate(self,distance,profile=False): # Propagate Amplitude object through distance with resolution, zres; return Amplitude object. 
        Uprev = self
        if profile:
            res = 50                    # Set res to 50 if generating plot of beam profile.
        else:
            res = zres                  # Otherwise use global variable, zres
        num = distance // res           # number of steps: divide distance by resolution. 
        rem = distance % res            # remainder of division: final step size. If num = 0, i.e. zres > distance, single step taken, equal to distance. 
        for i in range(num):            # num steps
            Unext = Uprev.step(res)
            zprev = z[-1]
            z.append(zprev + res)       # Build up z-array as go along. 
            if profile:
                w.append(Gaussfit(x,Unext.Uabs,1)[2]) # Build up waist-array as go along if profile set to 'True'.
            Uprev = Unext
        Unext = Uprev.step(rem)         # Final step of size rem. 
        zprev = z[-1]
        z.append(zprev + rem) 
        if profile:
            w.append(Gaussfit(x,Unext.Uabs,1)[2])
        Uprev = Unext
        return Unext

    def tilt(self,angle): # Applies linear phase-gradient, simulating effect of tilting mirror. Input angle in mrad. 
        Uin = self.U
        a = angle / 1000
        Uout = Uin * np.exp(-1j * kwav * x * np.sin(a))
        Ureturn = Amplitude(Uout)
        return Ureturn

    def lens(self,f): # Lens element of focal length, f. 
        Uin = self.U
        Uout = Uin * np.exp(-1j * kwav * x**2 / (2 * f))
        Ureturn = Amplitude(Uout)
        return Ureturn

    def mirror(self,R): # Mirror element of radius of curvature, R. 
        Uin = self.U
        Uout = Uin * np.exp(-1j * kwav * x**2 / R)
        Ureturn = Amplitude(Uout)
        return Ureturn

    def amp_fit(self,plot=False,n=1): # Fit (and Plot) magnitude of Amplitude array in x-space. 
        Uplot = self.Uabs/max(self.Uabs)
        xparams = Gaussfit(x,Uplot)
        UFit = Gaussian(x, xparams[0], xparams[1], xparams[2])
        if plot == True:
            plt.figure(n)
            plt.plot(x,Uplot,'o', label = 'model data', markersize = 3)
            plt.plot(x,UFit,'-', label = 'fit')
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
        kplot = np.concatenate((kneg,kpos), axis = 0)     
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

def beam_profile(): 
    # Runs series of methods corresponding to propagation of beam through various elements in system. 
    # Calculate beam waist along the way ('True' condition passed to propagate). 
    U = initialise(w0,z0)
    U = U.propagate(space_0,True)
    U = U.lens(FI)
    U = U.propagate(space_1,True)
    U = U.propagate(space_2,True)
    U = U.mirror(R1)
    U = U.propagate(space_3,True)
    U = U.mirror(R2)
    U = U.propagate(space_4,True)
    U = U.propagate(10000 - (space_0 + space_1 + space_2 + space_3 + space_4),True)
    waist_plot(w,1)

def waist_plot(waist_array,n=3): # Plots beam profile for a given waist array. 
    plt.figure(figsize=(9, 7), dpi=120)
    zplot = 0.001 * np.asarray(z)
    plt.plot(zplot,waist_array, linewidth = 3)
    axes = plt.gca()
    axes.set_xlim([0, 10])
    axes.set_ylim(0.0,2.6)
    axes.set_xticks(np.linspace(0,10,11))
    axes.set_yticks(np.linspace(0.0,2.6,14))
    plt.grid(which = 'both', axis = 'both', linestyle = '--')
    axes.set_xlabel('Distance along beam path / m')
    axes.set_ylabel('1/e^2 beam radius / mm')
    plt.title('Layout 9c. Beam profile calculated with Fourier algorithm.')
    plt.tight_layout()

def simple_correction(b,c): 
    # Runs series of methods corresponding to propagation of beam through various elements in system. Fixed spacings, defined in global variables. 
    # Mirrors M1 and M2 tilted by angles b and c. Returns, x and k offsets at OMC waist. 
    U = initialise(w0,z0)
    U = U.propagate(space_0)
    U = U.lens(FI)
    U = U.propagate(space_1)
    U = U.tilt(b)
    U = U.propagate(space_2)
    U = U.mirror(R1)
    U = U.propagate(space_3)
    U = U.mirror(R2)
    U = U.tilt(c)
    U = U.propagate(space_4)
    xparams = U.amp_fit()
    Dx = xparams[0] / abs(xparams[2]) # normalise offset to width in x-space
    kparams = U.freq_fit()
    Dk = kparams[0] / abs(kparams[2]) # normalise offset to width in k-space
    return (Dx, Dk)

def simple_mirror_test(): # Tilt mirrors in turn by equal positive and negative amounts. Return x and k offsets at OMC waist.
    x1 = []
    k1 = []
    x2 = []
    k2 = []
    angle = np.linspace(-1.0, 1.0, 2)
    for i in range(len(angle)):
        Dx, Dk = simple_correction(angle[i],0)
        x1.append(Dx)
        k1.append(Dk)
    for i in range(len(angle)):
        Dx, Dk = simple_correction(0,angle[i])
        x2.append(Dx)
        k2.append(Dk)
    return (x1,k1,x2,k2)

def simple_corr_dep():  # Calculates orthogonality between mirrors: angle between unit vectors corresponding to effect of mirror tilt in x-k space. 
        Mtest = simple_mirror_test()
        orthogonality(Mtest[0],Mtest[1],Mtest[2],Mtest[3])
        xk_plot(Mtest[0],Mtest[1],Mtest[2],Mtest[3])

def xk_plot(x1,k1,x2,k2,dist=0,multiple=False,count=0,n=4): # Plots displacement in x-k space caused when mirrors, M1 and M2 are tilted.
    if multiple:                                                    # for multiple traces
            plt.figure(n, figsize=(7.2, 5.5), dpi=120)
            if count == 0:
                plt.plot(x2, k2, 'k', label = 'M2')                      
                plt.plot([], [], ' ', label = 'M1 at position:')   
            plt.plot(x1, k1, label = '%.0f' % (dist,))              
            plt.title('M1 moved relative to M2; M2 fixed. ')        
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            #plt.legend(loc = 'lower left', prop={'size': 8})
    else:                                                           # for single trace
            plt.figure(n, figsize=(6, 5.5), dpi=120)
            plt.plot(x1, k1, label = 'M1')                        
            plt.plot(x2, k2, label = 'M2')                         
            plt.title('Mirrors tilted through +/- 1 mrad')         
            plt.legend()
    axes = plt.gca()
    axes.set_xlim([-3, 3])
    axes.set_ylim([-3, 3])
    plt.xlabel('offset in x at OMC waist / 1/e^2 radius in x-space')
    plt.ylabel('offset in k_x at OMC waist / 1/e^2 radius in k-space')
    plt.tight_layout()

def orthogonality(x1,k1,x2,k2): # Calculates phase-separation between two mirrors which, when tilted, cause displacements of (x*,k*) at the OMC waist .
    v1 = np.array([[x1[-1] - x1[0], k1[-1] - k1[0]]])
    v2 = np.array([[x2[-1] - x2[0]], [k2[-1] - k2[0]]])
    v1_norm = v1 / np.sqrt(v1[0,0]**2 + v1[0,1]**2)
    v2_norm = v2 / np.sqrt(v2[0,0]**2 + v2[1,0]**2)
    dot_product = np.dot(v1_norm,v2_norm)
    phi_ASC = (180 / np.pi) * np.arccos(dot_product)
    #print('%.0f' % (phi_ASC[0,0],))
    return phi_ASC[0,0]

def correction(b,c,sp1,sp2,sp3,order): 
    # Runs series of methods corresponding to propagation of beam through various elements in system. Variable spacings passed to function.
    # Change order of tilt(b) and mirror(R1) depending on spacing between tilt(b) and tilt(c)
    U = initialise(w0,z0)
    U = U.propagate(space_0)
    U = U.lens(FI)
    if order == 1:
        U = U.propagate(sp1)
        U = U.tilt(b)
        U = U.propagate(sp2)
        U = U.mirror(R1)
        U = U.propagate(sp3)
    elif order == 2:
        U = U.propagate(sp1)
        U = U.mirror(R1)
        U = U.propagate(sp2)
        U = U.tilt(b)
        U = U.propagate(sp3)        
    U = U.mirror(R2)
    U = U.tilt(c)
    U = U.propagate(space_4)
    xparams = U.amp_fit()
    Dx = xparams[0] / abs(xparams[2]) # normalise offset to width in x-space
    kparams = U.freq_fit()
    Dk = kparams[0] / abs(kparams[2]) # normalise offset to width in k-space
    return (Dx, Dk)

def mirror_test(sp1,sp2,sp3,order): 
    # Tilt mirrors in turn by equal positive and negative amounts. 
    # Return x and k offsets at OMC waist. Variable spacings passed to function.
    x1 = []
    k1 = []
    x2 = []
    k2 = []
    angle = np.linspace(-1.0, 1.0, 2)
    for i in range(len(angle)):
        Dx, Dk = correction(angle[i],0,sp1,sp2,sp3,order)
        x1.append(Dx)
        k1.append(Dk)
    for i in range(len(angle)):
        Dx, Dk = correction(0,angle[i],sp1,sp2,sp3,order)
        x2.append(Dx)
        k2.append(Dk)
    return (x1,k1,x2,k2)

def corr_dep(): 
    # Calculates orthogonality between mirrors: angle between unit vectors corresponding to effect of mirror tilt in x-k space.
    # Varies separation between mirrors. Calculated in two stages: 1) tilt(b) before mirror(R1), 2) tilt(b) after mirror(R1). 
    s1 = np.linspace(1535,4335,14)
    s2 = 4335 - s1
    s3 = 1770
    phi_ASC = []
    OMAS_pos = []
    count = 0
    for i in range(len(s1)):
        Mtest = mirror_test(int(s1[i]),int(s2[i]),s3,1)
        phi_ASC.append(orthogonality(Mtest[0],Mtest[1],Mtest[2],Mtest[3]))
        OMAS_pos.append(-s2[i]-s3)
        xk_plot(Mtest[0],Mtest[1],Mtest[2],Mtest[3],OMAS_pos[count],True,count)
        count += 1
    s1 = 4335
    s2 = np.linspace(0,1770,10)
    s3 = 1770 - s2
    for i in range(len(s2)):
        Mtest = mirror_test(s1,int(s2[i]),int(s3[i]),2)
        phi_ASC.append(orthogonality(Mtest[0],Mtest[1],Mtest[2],Mtest[3]))
        OMAS_pos.append(-s3[i])
        xk_plot(Mtest[0],Mtest[1],Mtest[2],Mtest[3],OMAS_pos[count],True,count)
        count += 1
    phase_plot(OMAS_pos,phi_ASC)

def phase_plot(dist,angle,n=5): # Plots orthogonality as a function of mirror separation. 
    plt.figure(n)
    plt.plot(dist,angle, label = 'Result of Fourier algorithm')
    axes = plt.gca()
    axes.set_xlim([-4500,150])
    axes.set_ylim([0, 120])
    plt.yticks(np.linspace(0.0,120,9))
    axes.vlines(x = -2015-1770, ymin = 0, ymax = 120,\
    linewidth = 2,color = 0.75*np.array([1,0.25,0.25]),linestyles = 'dashed',label = 'OMB3 - MM calculator design')    
    axes.vlines(x = -1770, ymin = 0, ymax = 120,\
    linewidth = 2,color = 0.75*np.array([0.25,1,0.25]),linestyles = 'dashed',label = 'OMB1')
    axes.vlines(x = 0, ymin = 0, ymax = 120,\
    linewidth = 2,color = 0.5*np.array([1,1,1]),linestyles = 'dashed',label = 'OMB2')
    plt.grid(which = 'major', axis = 'both')
    plt.xlabel('mirror separation / mm')
    plt.ylabel('orthogonality of actuation / ˚')
    #plt.title('Vary position of OMB3 relative to OMB2')
    plt.tight_layout()
    plt.legend(loc = 'upper right')

def sense(var_space_4): # Introduce input error: offset, x0, or angle, a0. Variable spacing passed to function. Returns x and k offsets. 
    U = initialise(w0,z0,x0,a0)
    U = U.propagate(space_0)
    U = U.lens(FI)
    U = U.propagate(space_1)
    #U = U.tilt(0)
    U = U.propagate(space_2)
    U = U.mirror(R1)
    U = U.propagate(space_3)
    U = U.mirror(R2)
    #U = U.tilt(0)
    U = U.propagate(var_space_4)
    xparams = U.amp_fit()
    Dx = xparams[0] / abs(xparams[2]) # normalise offset to width in x-space
    kparams = U.freq_fit()
    Dk = kparams[0] / abs(kparams[2]) # normalise offset to width in k-space
    return (Dx, Dk)

def sen_dep(): # Calculates x and k offsets as a function of distance from OMA2, mirror(R2). 
    s4 = np.linspace(0,2000,11)
    senx = []
    senk = []
    for i in range(len(s4)):
        (Dx, Dk) = sense(int(s4[i]))
        senx.append(Dx)
        senk.append(Dk)
    sen_plot(s4,senx,senk)
    
def sen_plot(dist,xdispl,kdispl,n=6): # Plots x- and k- offset as function of distance from OMA2, mirror(R2).
    fig, ax1 = plt.subplots()
    ax1.plot(dist,xdispl, color = 'blue', label = 'x')
    ax2 = ax1.twinx()
    ax2.plot(dist,kdispl, color = 'orange', label = 'k_x')
    ax1.set_xlim([0, 2000])
    ax1.set_ylim([-3, 3])
    ax2.set_ylim([-3, 3])
    ax2.vlines(x = 1210, ymin = -3, ymax = +3,\
    linewidth = 2,color = 0.5*np.array([1,1,1]),linestyles = 'dashed',label = 'OMC waist')
    ax1.grid(which = 'major', axis = 'both')
    ax1.set_xlabel('distance from OMA2 / mm')
    ax1.set_ylabel('centre of amplitude distribution / 1/e^2 radius in x-space')
    ax2.set_ylabel('centre of spatial frequency distribution / 1/e^2 radius in kx-space')
    fig.legend(loc = 'lower left', bbox_to_anchor=(0.69, 0.72))
    fig.tight_layout()

def main():
    #beam_profile()
    #print(simple_correction(0,0))
    #simple_corr_dep()
    #corr_dep()
    sen_dep()
    plt.show()
    
if __name__ == "__main__":
    main()

# Redundant or stored code
# ------------------------
# Offset in x and k at +/- Rayleigh range from beam waist. 
'''
def sense45(x0,a0): 
    # Runs series of methods corresponding to propagation of beam through various elements in system. 
    # Input offset, x0, or angle applied. Measure change in x and k at the two points +/- one Rayleigh range from target waist.
    U = initialise(w0,z0,x0,a0)
    U = U.propagate(space_0)
    U = U.lens(FI)
    U = U.propagate(space_1)
    U = U.propagate(space_2)
    U = U.mirror(R1)
    U = U.propagate(space_3)
    U = U.mirror(R2)
    U = U.propagate(space_4 - bt)
    xparams1 = U.amp_fit(True)
    Dx1 = xparams1[0] / abs(xparams1[2]) # normalise offset to width in x-space
    kparams1 = U.freq_fit(True)
    Dk1 = kparams1[0] / abs(kparams1[2]) # normalise offset to width in k-space
    U = U.propagate(2 * bt)
    xparams2 = U.amp_fit(True,3)
    Dx2 = xparams2[0] / abs(xparams2[2]) # normalise offset to width in x-space
    kparams2 = U.freq_fit(True,4)
    Dk2 = kparams2[0] / abs(kparams2[2]) # normalise offset to width in k-space
    return (Dx1, Dk1, Dx2, Dk2)

def sense45_test(): 
    x1 = []
    k1 = []
    x2 = []
    k2 = []
    offset = np.linspace(-1.0, 1.0, 2)
    for i in range(len(offset)):
        Dx1, Dk1, Dx2, Dk2 = sense45(offset[i],0)
        x1.append(Dx1)
        k1.append(Dk1)
        x2.append(Dx2)
        k2.append(Dk2)
    angle = np.linspace(-0.1, 0.1, 2)
    for i in range(len(angle)):
        Dx1, Dk1, Dx2, Dk2 = sense45(0,angle[i])
        x1.append(Dx1)
        k1.append(Dk1)
        x2.append(Dx2)
        k2.append(Dk2)
    return (x1,k1,x2,k2)

def sense45_dep():
    S45test = sense45_test()
    #xk_plot(S45test[0],S45test[1],S45test[2],S45test[3])
    #orthogonality(S45test[0],S45test[1],S45test[2],S45test[3])
'''
# beam profile method for Amplitude class
'''
    def beam_profile(self,distance): # Same as propagate with addition: calculates width of beam along beam path.
        Uprev = self
        num = distance // zres
        rem = distance % zres
        for i in range(num):
            Unext = Uprev.step(zres)
            zprev = z[-1]
            z.append(zprev + zres)
            w.append(Gaussfit(x,Unext.Uabs)[2]) # build up waist-array as go along.
            Uprev = Unext
        Unext = Uprev.step(rem)
        zprev = z[-1]
        z.append(zprev + rem)
        w.append(Gaussfit(x,Unext.Uabs)[2])
        Uprev = Unext
        return Unext
'''
# z-array definition
'''
#L = 1000                                                            # final z-co-ordinate
#zstepnum = int(L * zstepsize)                                       # number of steps in z
#z = np.linspace(0, L, (zstepnum + 1))                               # 1D array representing z-space
'''
# Aperture
'''
 U0 = np.concatenate((np.zeros(int(0.475*R)),np.ones(int(0.01 * R)),np.zeros(int(0.03 * R)),np.ones(int(0.01 * R)),np.zeros(int(0.475 * R))), axis = 0)
'''
# Amplitude class definition
'''
        #self.Ureal = np.real(self.U)
        #self.Uimag = np.imag(self.U)
        #self.Uphas = np.arctan(self.Uimag / self.Ureal)
'''
# Gaussfit method for Amplitude class
'''
def Gaussfit(self): # Fit Gaussian to magnitude of Amplitude array. Return width. 
    Ufit = self.Uabs
    init_params = [0,1,1]
    est_params, est_err = fit(Gaussian, x, Ufit, p0 = init_params)
    return est_params[2]
'''
# simple - test case
# b = 1000 mm
'''
    w0 = 0.5642
    z0 = 0
    U = input_array(w0,z0)
    U = U.propagate(1000,True)
    U.amp_fit()
    waist_plot(w,2)
    plt.show()
'''
# beam profile - test case
# b = 1000 mm
'''
    w0 = 0.3915
    z0 = 1486
    U = initialise(w0,z0)
    U = U.lens(1000)
    U = U.propagate(1000,True)
    U = U.tilt(0)
    U = U.propagate(2000,True)
    U = U.tilt(0)
    U = U.propagate(1000,True)
    U = U.lens(1260)
    U = U.propagate(1000,True)
    U = U.propagate(2000,True)
    U = U.propagate(1000,TRue)
    waist_plot(w,3)
    plt.show()
'''
# deflection - test case
# b = 1000 mm
'''
    w0 = 0.3915
    z0 = 1486
    U = initialise(w0,z0)
    U = U.lens(1000)
    U = U.propagate(1000)
    U = U.tilt(0)
    U = U.propagate(2000)
    U = U.tilt(-0.1)
    U = U.propagate(1000)
    U = U.lens(1260)
    U = U.propagate(1000)
    P1 = Gaussfit(U.Uabs)[0]
    U.amp_fit()
    U = U.propagate(2000)
    U.amp_fit()
    P2 = Gaussfit(U.Uabs)[0]
    U = U.propagate(1000)
    plt.show()
    print(1000*P1)
    print(1000*P2)
'''
