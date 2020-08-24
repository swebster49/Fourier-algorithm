import numpy as np
import matplotlib.pyplot as plt
import beam as beam
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
z0 = -3000                                                          # input waist location
b = 1000                                                            # input Rayleigh range
w0 = np.sqrt(b * wav / np.pi)                                       # input waist size - specified by Rayleigh range
x0 = 0.0                                                            # initial offset in mm
a0 = 0.0                                                            # initial angle in mrad
space_0 = 2000                                                      # Start - M1
space_1 = 1000                                                      # M1 - W1
space_2 = 1000                                                      # W1 - M2
space_3 = 2000                                                      # M2 - L1
space_4 = 2000                                                      # L1 - S1
space_5 = 1000                                                      # S1 - W2
space_6 = 1000                                                      # W2 - S2
space_7 = 2000                                                      # S2 - End
var_space = (space_4, space_4 + space_5 + space_6)
F_L1 = 1670                                                         # L1

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
        W = 600                                                                 # Width of window in mm
        xres = 0.003                                                             # 1D array representing kz-space. Derived from condition for monochromaticity.
        N = int(W / xres)                                                       # Number of x-bins (keeps resolution the same)  
        self.x = np.linspace(-W/2, W/2, N)                                      # 1D array representing x-space
        self.kneg = np.linspace(-(np.pi * N)/W, -(2 * np.pi)/W, int(N/2))       # 1D array representing kx-space from max -ve value to min -ve value
        self.kpos = np.linspace((2 * np.pi)/W, (np.pi * N)/W, int(N/2))         # 1D array representing kx-space from min +ve value to max +ve value
        self.kx = np.concatenate((self.kpos,self.kneg), axis = 0)               # 1D array representing kx-space. Order of values matches that spatial frequency distribution derived from FFT of amlitude distribution
        self.kwav = 2 * np.pi / wav                                             # Magnitude of wave-vector
        self.kz = np.sqrt(self.kwav**2 - self.kx**2)                            # 1D array representing kz-space. Derived from condition for monochromaticity. 
        self.zres = 5000                                                        # z-resolution in mm: 3000 for most; 50 for beam profile. 
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
    U = U.propagate(space_1, True)
    U = U.propagate(space_2, True)
    U = U.propagate(space_3, True)
    U = U.lens(F_L1)
    U = U.propagate(space_4, True)
    U = U.propagate(space_5, True)
    U = U.propagate(space_6, True)
    U = U.propagate(space_7, True)
    width_plot(U.z,U.w)

def width_plot(distance_list,width_list,n=3): # Plots beam profile for a given waist array. 
    zplot = 0.001 * np.asarray(distance_list)
    wplot = np.asarray(width_list)
    plt.figure(figsize=(9, 7), dpi=120)
    plt.plot(zplot,wplot, linewidth = 3)
    axes = plt.gca()
    axes.set_xlim([0, 12])
    axes.set_ylim(0.0,2)
    axes.set_xticks(np.linspace(0,12,13))
    axes.set_yticks(np.linspace(0.0,2,11))
    plt.grid(which = 'both', axis = 'both', linestyle = '--')
    axes.set_xlabel('Distance from Start / m')
    axes.set_ylabel('Beam size / mm')
    #plt.title('Beam path from SR2 to QPD-I')
    plt.tight_layout()


def mirror(b,c): 
    # Runs series of methods corresponding to propagation of beam through various elements in system. Fixed spacings, defined in global variables. 
    # Mirrors M1 and M2 tilted by angles b and c. Returns, x and k offsets at OMC waist. 
    U = Beam(w0,z0)
    U = U.propagate(space_0)
    U = U.tilt(b)
    U = U.propagate(space_1)
    U = U.propagate(space_2)
    U = U.tilt(c)
    U = U.propagate(space_3)
    U = U.lens(F_L1)
    U = U.propagate(space_4)
    U = U.propagate(space_5)
    #U = U.propagate(space_6)
    #U = U.propagate(space_7)
    xparams = U.amp_fit()
    Dx = xparams[0] / abs(xparams[2]) # normalise offset to width in x-space
    kparams = U.freq_fit()
    Dk = kparams[0] / abs(kparams[2]) # normalise offset to width in k-space
    return (Dx, Dk)

def mirror_test(): # Tilt mirrors in turn by equal positive and negative amounts. Return x and k offsets at OMC waist.
    x1 = []
    k1 = []
    x2 = []
    k2 = []
    angle = np.linspace(-1.0, 1.0, 2)
    for i in range(len(angle)):
        Dx, Dk = mirror(angle[i],0)
        x1.append(Dx)
        k1.append(Dk)
    for i in range(len(angle)):
        Dx, Dk = mirror(0,angle[i])
        x2.append(Dx)
        k2.append(Dk)
    return (x1,k1,x2,k2)

def mirror_dep():  # Calculates orthogonality between mirrors: angle between unit vectors corresponding to effect of mirror tilt in x-k space. 
    Mtest = mirror_test()
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
    plt.title('Tilt mirrors, M1 & M2. ')         
    plt.legend()
    axes = plt.gca()
    axes.set_xlim([-3, 3])
    axes.set_ylim([-3, 3])
    plt.xlabel('offset in x at W2 / 1/e^2 radius in x-space')
    plt.ylabel('offset in k_x at W2 / 1/e^2 radius in k-space')
    textstr = 'Orthogonality: %.0f˚' % (orthogonality(x1,k1,x2,k2),)
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    axes.text(0.02, 0.98, textstr, transform=axes.transAxes, fontsize=10,verticalalignment='top', bbox=props)
    plt.tight_layout()

def sensor_test(var_space): # Apply +/- displacement/direction errors at SRM and return x-displacement at WFS1 and WFS2.
    x1_displ = []
    x2_displ = []
    x1_direc = []
    x2_direc = []
    displ = np.linspace(-1.0, 1.0, 2)
    direc = np.linspace(-1.0, 1.0, 2)
    for i in range(len(displ)):
        Dx1 = sensor(displ[i],0.0,var_space[0])
        Dx2 = sensor(displ[i],0.0,var_space[1])
        x1_displ.append(Dx1)
        x2_displ.append(Dx2)
    for i in range(len(direc)):
        Dx1 = sensor(0.0,direc[i],var_space[0])
        Dx2 = sensor(0.0,direc[i],var_space[1])
        x1_direc.append(Dx1)
        x2_direc.append(Dx2)
    return (x1_displ,x2_displ,x1_direc,x2_direc)

def sensor_dep():  # Calculates x offsets at WFS1 and WFS2 for +/- dislpacement/direction errors at SRM.
        Stest = sensor_test(var_space)
        #WFS_orthogonality(Stest[0],Stest[1],Stest[2],Stest[3])
        sensor_plot(Stest[0],Stest[1],Stest[2],Stest[3])

def sensor_orthogonality(x1_displ,x2_displ,x1_direc,x2_direc): # Calculates phase-separation between WFS which sense displacement and direction errors upstream.
    v1 = np.array([[x1_displ[-1] - x1_displ[0], x2_displ[-1] - x2_displ[0]]])
    v2 = np.array([[x1_direc[-1] - x1_direc[0]], [x2_direc[-1] - x2_direc[0]]])
    v1_norm = v1 / np.sqrt(v1[0,0]**2 + v1[0,1]**2)
    v2_norm = v2 / np.sqrt(v2[0,0]**2 + v2[1,0]**2)
    dot_product = np.dot(v1_norm,v2_norm)
    phi_ASC = (180 / np.pi) * np.arccos(dot_product)
    #print('%.1f' % (phi_ASC[0,0],))
    return phi_ASC[0,0]

def sensor_plot(x1_displ,x2_displ,x1_direc,x2_direc,n=5): # Plots displacement at WFS1 vs displacement at WFS2 when displacement and direction errors are applied at the SRM. 
    plt.figure(n, figsize=(6, 5.5), dpi=120)
    plt.plot(x1_displ, x2_displ, label = 'displacement')                        
    plt.plot(x1_direc, x2_direc, label = 'direction')                         
    axes = plt.gca()
    axes.set_xlim([-3, 3])
    axes.set_ylim([-3, 3])
    plt.title('Apply pure displacement and direction errors at W1')         
    plt.xlabel('offset in x at S1 / 1/e^2 radius')
    plt.ylabel('offset in x at S2 / 1/e^2 radius')
    plt.legend(loc = 'upper right')
    textstr = 'Orthogonality: %.0f˚' % (sensor_orthogonality(x1_displ,x2_displ,x1_direc,x2_direc),)
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    axes.text(0.02, 0.98, textstr, transform=axes.transAxes, fontsize=10,verticalalignment='top', bbox=props)
    plt.tight_layout()
    
def sensor(x0,a0,var_space):#,count): 
    # Runs series of methods corresponding to propagation of beam through various elements in system. Fixed spacings, defined in global variables. 
    # Displacement and Direction errors applied at SRM. Returns, x offsets at WFS1 and WFS2. 
    U = Beam(w0,0.0,x0,a0)
    if var_space <= 3000:
        space_A = var_space
        U = U.propagate(space_A)
    elif var_space > 3000:
        space_A = 3000
        space_B = var_space - 3000
        U = U.propagate(space_A)
        U = U.lens(F_L1)
        U = U.propagate(space_B)
    xparams = U.amp_fit()
    Dx = xparams[0] / abs(xparams[2]) # normalise offset to width in x-space
    return Dx

def sensor_position(show = True): # Calculates x offset as a function of distance from L1 for pure displacement and direction errors
    s4 = np.concatenate((np.linspace(0,3000,31),np.linspace(3100,9000,60)), axis = 0)
    Dx_displ = []
    Dx_direc = []
    count = 0
    for i in range(len(s4)):
        Dx = sensor(1.0,0.0,int(s4[i]))#,count)
        Dx_displ.append(Dx)
        #if s4[i] == 3000:
        #    count += 1
    for i in range(len(s4)):
        Dx = sensor(0.0,1.0,int(s4[i]))#,count)
        Dx_direc.append(Dx)
        #if s4[i] == 3000:
        #    count += 1
    if show:
        sensor_position_plot(s4,Dx_displ,Dx_direc)
    return (Dx_displ,Dx_direc)

def sensor_position_plot(dist,Dx_displ,Dx_direc,n=6): # Plots x offset as function of distance from L1 for pure displacement and direction errors
    d_plot = dist / 1000
    plt.figure(n, figsize=(6, 5.5), dpi=120)
    plt.plot(d_plot,Dx_displ, label = 'displacement')
    plt.plot(d_plot,Dx_direc, label = 'direction')
    plt.xlim([0, 9])
    plt.ylim([-3, 3])
    plt.grid(which = 'major', axis = 'both')
    plt.title('Apply pure displacement and direction errors at W1')     
    plt.xlabel('distance from W1 / mm')
    plt.ylabel('offset in x / 1/e^2 radius')
    plt.legend(loc = 'upper right')
    plt.tight_layout()

def beam_propagate():
    s0 = (space_2 + space_3) / 1000
    s1 = (space_4 + space_5 + space_6 + space_7) / 1000
    z1 = 0.0
    b1 = b / 1000
    S_L1 = 1 / (F_L1 / 1000)
    q0 = beam.beamParameter(z1+1j*b1)    
    q1 = beam.propagate(q0,s0)         
    q2 = beam.lens(q1,S_L1,0.0)       
    q3 = beam.propagate(q2,s1)
    return (q0, q1, q2, q3)

def beam_distance():
    s0 = (space_2 + space_3) / 1000
    s1 = (space_4 + space_5 + space_6 + space_7) / 1000
    z0 = np.linspace(0,s0,31)                                        
    z1 = np.linspace(s0 + 0.1, s0 + s1, 60)
    z = np.concatenate((z0,z1),axis= 0)
    return (z0, z1, z)

def beam_Gouy(q_params,z_params):
    q0 = q_params[0]
    q2 = q_params[2]
    z0 = z_params[0]
    z1 = z_params[1]
    s0 = (space_2 + space_3) / 1000
    s1 = (space_4 + space_5 + space_6 + space_7) / 1000
    q0_gouy = beam.gouyPhase(q0,z0)\
    - beam.gouyPhase(q0,0.0)
    q2_gouy = beam.gouyPhase(beam.propagate(q2,-(s0 + 0.1)),z1)\
    - beam.gouyPhase(q0,0.0)\
    - (beam.gouyPhase(beam.propagate(q2,-(s0+0.1)),s0) - beam.gouyPhase(beam.propagate(q0,-0.0),s0))
    Gouy_Phase = np.concatenate((q0_gouy,q2_gouy),axis = 0)
    return Gouy_Phase

def Gouy(show = True): # Calculates Gouy phase as a function of distance
    q_params = beam_propagate()
    z_params = beam_distance()
    z = z_params[2]
    Gouy_Phase = beam_Gouy(q_params,z_params)
    if show:
        plot_Gouy(z, Gouy_Phase)
    return Gouy_Phase

def plot_Gouy(z, Gouy_Phase, n=7):
    plt.figure(n, figsize=(6, 5.5), dpi=120)
    plt.plot(z, Gouy_Phase*180/np.pi)
    plt.grid(which = 'both', linestyle='--')
    plt.xlim(0,9)
    plt.xticks(np.linspace(0,9,10))
    plt.ylim(0,270)
    plt.yticks(np.linspace(0,270,10))
    plt.grid(which = 'both', linestyle='--')
    plt.xlabel('Distance from W1 / m')
    plt.ylabel('Gouy Phase (degree)')
    plt.tight_layout()  # otherwise the right y-label is slightly clipped

def sense_vs_Gouy():
    Sensors = sensor_position(False)
    Gouy_Phase = Gouy(False)
    sensor_vs_Gouy_plot(Gouy_Phase,Sensors[0],Sensors[1])

def sensor_vs_Gouy_plot(Gouy_Phase,Dx_displ,Dx_direc,n=8): # Plots x offset as function of distance from L1 for pure displacement and direction errors
    plt.figure(n, figsize=(6, 5.5), dpi=120)
    plt.plot(Gouy_Phase*180/np.pi, Dx_displ, label = 'displacement')
    plt.plot(Gouy_Phase*180/np.pi, Dx_direc, label = 'direction')
    plt.xlim([0, 270])
    plt.xticks(np.linspace(0,270,10))
    plt.ylim([-3, 3])
    plt.grid(which = 'major', axis = 'both')
    plt.grid(which = 'minor', axis = 'both')
    plt.title('Apply pure displacement and direction errors at W1')     
    plt.xlabel('Gouy phase separation from W1 / mm')
    plt.ylabel('offset in x / 1/e^2 radius')
    plt.legend(loc = 'upper right')
    plt.tight_layout()

def main():
    #beam_profile()
    #mirror_dep()
    #sensor_dep()
    #sensor_position()
    #Gouy()
    sense_vs_Gouy()
    plt.show()
    
if __name__ == "__main__":
    main()

'''
elif var_space == 3000:
    if count == 0:
        space_A = var_space
        U = U.propagate(space_A)
    elif count == 1:
        space_A = 3000
        #space_B = var_space - 3000
        U = U.propagate(space_A)
        #U = U.lens(F_L1)
        #U = U.propagate(space_B)  
'''