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
z0 = 0                                                              # input waist location
b = 1000                                                            # input Rayleigh range
w0 = np.sqrt(b * wav / np.pi)                                       # input waist size - specified by Rayleigh range
x0 = 0.0                                                            # initial offset in mm
a0 = 0.0                                                            # initial angle in mrad
ST_M1 = 2000                                                      # Start - M1
M1_W1 = 1000                                                      # M1 - W1
W1_M2 = 1000                                                      # W1 - M2
M2_L1 = 2000                                                      # M2 - L1
L1_S1 = 2000                                                      # L1 - S1
S1_W2 = 1000                                                      # S1 - W2
W2_S2 = 1000                                                      # W2 - S2
S2_EN = 2000                                                      # S2 - End
space_A = W1_M2 + M2_L1
space_B = L1_S1 + S1_W2 + W2_S2 + S2_EN
space_C = L1_S1 + S1_W2
space_D = ST_M1 + M1_W1 + W1_M2 + M2_L1
F_L1 = 1670

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
        self.idx = int(N/2)                                                     # First point of second-half of array
        self.x = np.linspace(-W/2, W/2, N)                                      # 1D array representing x-space
        self.kneg = np.linspace(-(np.pi * N)/W, -(2 * np.pi)/W, self.idx)       # 1D array representing kx-space from max -ve value to min -ve value
        self.kpos = np.linspace((2 * np.pi)/W, (np.pi * N)/W, self.idx)         # 1D array representing kx-space from min +ve value to max +ve value
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
        self.g = [0]                                                            # Initialise Gouy-phase list
        self.z = [0]                                                            # Initialise z-position list.

    def step(self, D): # Propagate input Beam object over distance, D; return Beam object. Fourier algorithm. 
        Pin = fft(self.U)                       # FFT of amplitude distribution, U, gives spatial frequency distribution, Pin at initial position, z.
        Pout = Pin * np.exp(1j * self.kz * D)   # Multiply Spatial frequency distribution by phase-factor corresponding to propagation through distance, D, to give new spatial frequency distribution, Pout. 
        Uout = ifft(Pout)                       # IFFT of spatial frequency distribution gives amplitude distribution, Uout, at plane z = z + D
        self.U = Uout
        return self

    def profile(self,step_size):
        self.z.append(self.z[-1] + step_size)   # Accumulate z-array.
        wnext = Gaussfit(self.x,abs(self.U),1)[2]
        self.w.append(wnext)
        p1 = ((np.angle(self.U[self.idx]) + np.angle(self.U[self.idx-1]))/2)    # Mean of phase of the two central points of the distribution
        pl = (self.kwav * step_size) % (2 * np.pi)                              # Phase accumulated by plane wave modulo 2pi
        g1 = 2 * (pl - p1) % (2 * np.pi)    # !! Fudge-factor of 2 - gives correct answer. Gouy-phase is phase of Gaussian beam relative to that of a plane wave propagated by the same distance. 
        self.g.append(self.g[-1] + g1)      # Accumulate Gouy-phase array.
        self.U = self.U * np.exp(-1j * p1)
        return self

    def propagate(self,distance,res=10000,show=False): # Propagate Beam object through distance with resolution, zres; return Beam object. 
        num = distance // res           # number of steps: divide distance by resolution. 
        rem = distance % res            # remainder of division: final step size. If num = 0, i.e. zres > distance, single step taken, equal to distance. 
        p0 = ((np.angle(self.U[self.idx]) + np.angle(self.U[self.idx-1]))/2) #% (2 * np.pi) # average phase at centre of distribution
        self.U = self.U * np.exp(-1j * p0)   # re-set phase to zero
        for i in range(num):            # num steps
            self = self.step(res)
            if show:
                self = self.profile(res)
        if rem != 0:
            self = self.step(rem)         # Final step of size rem. 
            if show:
                self = self.profile(rem)
        return self

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

def beam_profile(space_0, space_1, offset,res,beam=False,gouy=False): 
    U = Beam(w0,offset)
    U = U.propagate(space_0 - offset, res, True)
    U = U.lens(F_L1)
    U = U.propagate(space_1, res, True)
    if beam:
        width_plot(U.z,U.w)
    if gouy:
        gouy_plot(U.z,U.g)
    return (U.z, U.g)

def width_plot(distance_list,width_list,n=3): # Plots beam profile for a given waist array. 
    zplot = 0.001 * np.asarray(distance_list)
    wplot = np.asarray(width_list)
    plt.figure(n,figsize=(6, 5.5), dpi=120)
    plt.plot(zplot,wplot, linewidth = 2)
    axes = plt.gca()
    axes.set_xlim([0, 12])
    axes.set_ylim(0.0,2.6)
    axes.set_xticks(np.linspace(0,12,13))
    axes.set_yticks(np.linspace(0.0,2.6,14))
    plt.grid(which = 'both', axis = 'both', linestyle = '--')
    axes.set_xlabel('distance along beam / m')
    axes.set_ylabel('beam radius / mm')
    #plt.title('')
    plt.tight_layout()

def gouy_plot(distance_list,gouy_list,n=4):
    zplot = 0.001 * np.asarray(distance_list)
    gplot = (180 / np.pi) * np.asarray(gouy_list)
    plt.figure(n,figsize=(6, 5.5), dpi=120)
    plt.plot(zplot,gplot, linewidth = 2)
    axes = plt.gca()
    axes.set_xlim([0, 12])
    axes.set_ylim(0.0,300)
    axes.set_xticks(np.linspace(0,12,13))
    axes.set_yticks(np.linspace(0,300,11))
    plt.grid(which = 'both', axis = 'both', linestyle = '--')
    axes.set_xlabel('distance along beam / m')
    axes.set_ylabel('gouy-phase / ˚')
    #plt.title('')
    plt.tight_layout()

###########################################################################################################################################
# Start of Code which combines calculation of beam offsets using Fourier algorithm with calculation of Gouy phasee using ABCD Matrix method
###########################################################################################################################################

def sensor(x0,a0,var_space,s0,count=0): # Propagates beam from W1 to sensor. Variable distance between W1 and sensor. Displacement and direction errors applied at W1. Handles cases with sensor before and after lens. 
    U = Beam(w0,0.0,x0,a0)
    if var_space < s0:
        space_A = var_space
        U = U.propagate(space_A)
    elif int(var_space) == s0:
        if count == 0:
            space_A = var_space
            U = U.propagate(space_A)
        elif count == 1:
            space_A = s0
            space_B = var_space - s0
            U = U.propagate(space_A)
            U = U.lens(F_L1)
            U = U.propagate(space_B)
    elif var_space > s0:
        space_A = s0
        space_B = var_space - s0
        U = U.propagate(space_A)
        U = U.lens(F_L1)
        U = U.propagate(space_B)
    xparams = U.amp_fit()
    Dx = xparams[0] #/ abs(xparams[2])
    return Dx

def sensor_offset_distance(act_off,res,show = False): # Calculates x-offset as function of distance from W1 to sensor for displacement and direction errors applied at W1
    s0 = space_A - act_off
    s1 = space_B
    n0 = int(s0/res + 1)
    n1 = int(s1/res)
    s = np.concatenate((np.linspace(0,s0,n0),np.linspace(s0+res,s0+s1,n1)), axis = 0)
    Dx_direc = []
    Dx_displ = []
    phi_direc = []
    phi_displ = []
    count = 0
    for i in range(len(s)):
        direc = sensor(0.0,1.0,int(s[i]),int(s0),count)
        Dx_direc.append(direc)
        displ = sensor(1.0,0.0,int(s[i]),int(s0),count)
        Dx_displ.append(displ)
        phi_direc.append(np.arctan(direc / displ))
        phi_displ.append(np.arctan(displ / direc))
        if int(s[i]) == s0:
            count += 1    
    if show:
        sensor_offset_distance_plot(s,Dx_direc,Dx_displ,act_off)
        sensor_phase_distance_plot(s,phi_direc,phi_displ,sen_off)
    return (Dx_direc,Dx_displ,phi_direc,phi_displ)

def sensor_offset_distance_plot(dist,Dx_direc,Dx_displ,act_off,n=5): # Plots x offset as function of distance from W1 to sensor for displacement and direction errors applied at W1
    d_plot = dist / 1000
    a_plot = act_off / 1000
    plt.figure(n, figsize=(6, 5.5), dpi=120)
    plt.plot(d_plot,Dx_direc, label = 'direction error')
    plt.plot(d_plot,Dx_displ, label = 'displacement error')
    plt.xlim([0, 12])
    #plt.ylim([-6, 6])
    plt.grid(which = 'major', axis = 'both')
    plt.title('Actuator fixed, offset %.0f m from waist; move sensor.' % (a_plot,))     
    plt.xlabel('separation between actuator and sensor / m')
    plt.ylabel('x-offset at sensor/ mm')
    plt.legend(loc = 'upper right')
    plt.tight_layout()

def sensor_phase_distance_plot(dist,phi_direc,phi_displ,act_off,n=13): # Plots x- and k- offsets as a function of distance from mirror at which direction correction is applied
    d_plot = dist / 1000
    direc_plot = np.asarray(phi_direc)*180/np.pi
    displ_plot = np.asarray(phi_displ)*180/np.pi
    a_plot = act_off / 1000
    plt.figure(n, figsize=(6, 5.5), dpi=120)
    plt.plot(d_plot,direc_plot, label = 'direction error')
    plt.plot(d_plot,displ_plot, label = 'displacement error')
    plt.xlim([0, 12])
    plt.ylim([-90, 90])
    plt.yticks(np.linspace(-90,90,7))
    plt.grid(which = 'major', axis = 'both')
    plt.title('Actuator fixed, offset %.0f m from waist; move sensor.' % (a_plot,))     
    plt.xlabel('separation between actuator and sensor / m')
    plt.ylabel('angle relative to axis / ˚')
    plt.legend()
    plt.tight_layout()

def sensor_offset_Gouy(space_A,space_B,act_off,res,show_offset,show_Gouy,show_offset_Gouy,show_phase_Gouy):
    Offsets = sensor_offset_distance(act_off,res,show_offset)
    Gouy_Phase = beam_profile(space_A, space_B, act_off,res,False,show_Gouy)
    if show_offset_Gouy:
        sensor_offset_Gouy_plot(Gouy_Phase[1],Offsets[0],Offsets[1],act_off)
    if show_phase_Gouy:
        sensor_phase_Gouy_plot(Gouy_Phase[1],Offsets[2],Offsets[3],act_off)

def sensor_offset_Gouy_plot(Gouy_Phase,Dx_direc,Dx_displ,act_off,n=7): # Plots x offset at sensor as a function of phase-separation from W1 with displacement and direction errors applied at W1
    a_plot = act_off / 1000
    g_plot = np.asarray(Gouy_Phase)*180/np.pi
    direc_plot = np.asarray(Dx_direc)
    displ_plot = np.asarray(Dx_displ)
    plt.figure(n, figsize=(6, 5.5), dpi=120)
    plt.plot(g_plot, direc_plot, label = 'direction error')
    plt.plot(g_plot, displ_plot, label = 'displacement error')
    plt.xlim([0, 300])
    plt.xticks(np.linspace(0,300,11))
    #plt.ylim([-6, 6])
    plt.grid(which = 'major', axis = 'both')
    plt.title('Actuator fixed, offset %.0f m from waist; move sensor.' % (a_plot,))     
    plt.xlabel('phase separation between actuator and sensor / ˚')
    plt.ylabel('x-offset at sensor / mm')
    plt.legend(loc = 'upper right')
    plt.tight_layout()

def sensor_phase_Gouy_plot(Gouy_Phase,phi_direc,phi_displ,act_off,n=14): # Plots x- and k-offsets at W2 as function of distance mirror at which a direction correction is applied
    a_plot = act_off / 1000
    g_plot = np.asarray(Gouy_Phase)*180/np.pi
    direc_plot = np.asarray(phi_direc)*180/np.pi
    displ_plot = np.asarray(phi_displ)*180/np.pi
    plt.figure(n, figsize=(6, 5), dpi=120)
    plt.plot(g_plot, direc_plot, label = 'direction error')
    plt.plot(g_plot, displ_plot, label = 'displacement error')
    plt.xlim([0, 300])
    plt.xticks(np.linspace(0,300,11))
    plt.ylim([-90, 90])
    plt.yticks(np.linspace(-90,90,7))
    plt.grid(which = 'major', axis = 'both')
    plt.title('Actuator fixed, offset %.0f m from waist; move sensor.' % (a_plot,))     
    plt.xlabel('phase separation between actuator and waist / ˚')
    plt.ylabel('angle relative to axis / ˚')
    plt.legend()
    plt.tight_layout()

def actuator(x0,a0,var_space,s0,count=0): # Propagates beam from W1 to sensor. Variable distance between W1 and sensor. Displacement and direction errors applied at W1. Handles cases with sensor before and after lens. 
    sen_off = s0 - space_C
    if var_space < s0:
        space_B = var_space
        U = Beam(w0, sen_off - var_space, x0,a0)
        U = U.propagate(space_B)
    elif int(var_space) == s0:
        if count == 0:
            space_B = var_space
            U = Beam(w0, sen_off - var_space, x0,a0)
            U = U.propagate(space_B)
        elif count == 1:
            space_B = s0
            space_A = var_space - s0
            U = Beam(w0, space_D + sen_off - var_space, x0,a0)
            U = U.propagate(space_A)
            U = U.lens(F_L1)
            U = U.propagate(space_B)
    elif var_space > s0:
        space_B = s0
        space_A = var_space - s0
        U = Beam(w0, space_D + sen_off - var_space,x0,a0)
        U = U.propagate(space_A)
        U = U.lens(F_L1)
        U = U.propagate(space_B)
    xparams = U.amp_fit()
    kparams = U.freq_fit()
    Dx = xparams[0] / abs(xparams[2])
    Dk = kparams[0] / abs(kparams[2])
    return Dx, Dk

def actuator_offset_distance(sen_off,res,show = False): # Calculates x-offset as function of distance from W1 to sensor for displacement and direction errors applied at W1
    s0 = space_C + sen_off # Add offset because treating beam as if it is propagating backwards - from W2 to the Start. 
    s1 = space_D
    n0 = int(s0/res + 1)
    n1 = int(s1/res)
    s = np.concatenate((np.linspace(0,s0,n0),np.linspace(s0+res,s0+s1,n1)), axis = 0)
    Dx_direc = []
    Dk_direc = []
    phi_x = []
    phi_k = []
    count = 0
    for i in range(len(s)):
        Dx, Dk = actuator(0.0,1.0,int(s[i]),int(s0),count)
        Dx_direc.append(Dx)
        Dk_direc.append(Dk)
        if int(s[i]) == s0:
            count += 1
        phi_x.append(np.arctan(Dk / Dx))
        phi_k.append(np.arctan(Dx / Dk))
    if show:
        actuator_offset_distance_plot(s,Dx_direc,Dk_direc,sen_off)
        actuator_phase_distance_plot(s,phi_x,phi_k,sen_off)
    return (Dx_direc,Dk_direc,phi_x,phi_k)

def actuator_offset_distance_plot(dist,Dx_direc,Dk_direc,sen_off,n=8): # Plots x- and k- offsets as a function of distance from mirror at which direction correction is applied
    d_plot = dist / 1000
    s_plot = sen_off / 1000
    plt.figure(n, figsize=(6, 5.5), dpi=120)
    plt.plot(d_plot,Dx_direc, label = 'x-offset')
    plt.plot(d_plot,Dk_direc, label = 'k-offset')
    plt.xlim([0, 12])
    plt.grid(which = 'major', axis = 'both')
    plt.title('Move actuator; sensor fixed, offset %.0f m from waist.' % (s_plot,))     
    plt.xlabel('separation between actuator and sensor / m')
    plt.ylabel('offset at sensor/ mm')
    plt.legend(loc = 'upper right')
    plt.tight_layout()

def actuator_phase_distance_plot(dist,phi_x,phi_k,sen_off,n=11): # Plots x- and k- offsets as a function of distance from mirror at which direction correction is applied
    d_plot = dist / 1000
    px_plot = np.asarray(phi_x)*180/np.pi
    pk_plot = np.asarray(phi_k)*180/np.pi
    s_plot = sen_off / 1000
    plt.figure(n, figsize=(6, 5.5), dpi=120)
    plt.plot(d_plot,px_plot, label = 'x')
    plt.plot(d_plot,pk_plot, label = 'k')
    plt.xlim([0, 12])
    plt.ylim([-90, 90])
    plt.yticks(np.linspace(-90,90,7))
    plt.grid(which = 'major', axis = 'both')
    plt.title('Move actuator; sensor fixed, offset %.0f m from waist.' % (s_plot,))     
    plt.xlabel('separation between actuator and sensor / m')
    plt.ylabel('angle relative to axis / ˚')
    plt.legend()
    plt.tight_layout()

def actuator_offset_Gouy(space_C,space_D,sen_off,res,show_offset,show_Gouy,show_offset_Gouy,show_phase_Gouy,show_xk): 
    Offsets = actuator_offset_distance(sen_off,res,show_offset)
    Gouy_Phase = beam_profile(space_C,space_D,-sen_off,res,False,show_Gouy) # Negative of sen_off because propagate beam backwards in this case. 
    if show_offset_Gouy:
        actuator_offset_Gouy_plot(Gouy_Phase[1],Offsets[0],Offsets[1],sen_off)
    if show_phase_Gouy:
        actuator_phase_Gouy_plot(Gouy_Phase[1],Offsets[2],Offsets[3],sen_off)
    if show_xk:
        xk_space_plot(Gouy_Phase[1],Offsets[0],Offsets[1],sen_off)

def actuator_offset_Gouy_plot(Gouy_Phase,Dx_direc,Dk_direc,sen_off,n=10): # Plots x- and k-offsets at W2 as function of distance mirror at which a direction correction is applied
    s_plot = sen_off / 1000
    g_plot = np.asarray(Gouy_Phase)*180/np.pi
    x_plot = np.asarray(Dx_direc)
    k_plot = np.asarray(Dk_direc)
    plt.figure(n, figsize=(6, 5.5), dpi=120)
    plt.plot(g_plot, x_plot, label = 'x-offset')
    plt.plot(g_plot, k_plot, label = 'k-offset')
    plt.xlim([0, 300])
    plt.xticks(np.linspace(0,300,11))
    #plt.ylim([-6, 6])
    plt.grid(which = 'major', axis = 'both')
    plt.title('Move actuator; sensor fixed, offset %.0f from waist.' % (s_plot,))     
    plt.xlabel('phase separation between actuator and waist / ˚')
    plt.ylabel('offset at sensor/ mm')
    plt.legend(loc = 'upper right')
    plt.tight_layout()

def actuator_phase_Gouy_plot(Gouy_Phase,phi_x,phi_k,sen_off,n=12): # Plots x- and k-offsets at W2 as function of distance mirror at which a direction correction is applied
    s_plot = sen_off / 1000
    g_plot = np.asarray(Gouy_Phase)*180/np.pi
    px_plot = np.asarray(phi_x)*180/np.pi
    offset = 360
    new_plot = [px_plot[0] + offset]
    for i in range(len(px_plot)-1):
        if px_plot[i+1] - px_plot[i] > 90:
            offset += -180
        new_plot.append(px_plot[i+1] + offset)
    #pk_plot = np.asarray(phi_k)*180/np.pi
    plt.figure(n, figsize=(6, 5), dpi=120)
    plt.plot(g_plot, new_plot, label = 'x')
    #plt.plot(g_plot, pk_plot, label = 'k')
    plt.xlim([0, 300])
    plt.xticks(np.linspace(0,300,11))
    plt.ylim([0, 300])
    plt.yticks(np.linspace(0,300,11))
    plt.grid(which = 'major', axis = 'both')
    plt.title('Move actuator; sensor fixed, offset %.0f from waist.' % (s_plot,))     
    plt.xlabel('phase separation between actuator and waist / ˚')
    plt.ylabel('angle relative to axis / ˚')
    #plt.legend()
    plt.tight_layout()

def xk_space_plot(Gouy_Phase,Dx_direc,Dk_direc,sen_off,n=14):
    s_plot = sen_off / 1000
    g_calc = np.asarray(Gouy_Phase)*180/np.pi
    phase = 0
    incr = 5
    plt.figure(n, figsize=(6, 5.5), dpi=120)
    plt.plot([0, Dx_direc[0]],[0, Dk_direc[0]])
    for i in range(len(Gouy_Phase)-1):
        if g_calc[i+1] >= phase + incr: # interpolate to plot with regular increments in Gouy phase
            phase = phase + incr
            f = ((phase) - g_calc[i]) / (g_calc[i+1] - g_calc[i])
            x_plot = Dx_direc[i] + f * (Dx_direc[i+1] - Dx_direc[i])
            k_plot = Dk_direc[i] + f * (Dk_direc[i+1] - Dk_direc[i])
            plt.plot([0, x_plot],[0, k_plot])
            #print(phase, '\t', g_calc[i], '\t', g_calc[i+1], '\t', f, '\t', x_plot, '\t', k_plot)
    plt.title('Move actuator; sensor fixed, offset %.0f m from waist.' % (s_plot,))     
    plt.xlabel('x-offset at sensor/ 1/e^2 radius')
    plt.ylabel('k-offset at sensor/ 1/e^2 radius')
    axes = plt.gca()
    textstr = '%.0f˚ increments in Gouy phase' % (incr,)
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    axes.text(0.02, 0.98, textstr, transform=axes.transAxes, fontsize=10,verticalalignment='top', bbox=props)
    plt.tight_layout()

def main():
    res = 100
    act_off = 0
    sen_off = 0
    #beam_profile(space_A,space_B,act_off,res,False,True)                        # Booleans: Beam-radius vs distance; Gouy-Phase vs distance
    #sensor_offset_distance(act_off,res,True)                                   # Boolean: Offset/Phase vs distance
    #sensor_offset_Gouy(space_A,space_B,act_off,res,False,False,True,True)      # Booleans: Offset/Phase vs distance; Gouy vs distance; Offset/Phase vs Gouy; Phase vs Gouy
    #actuator_offset_distance(sen_off,res,True)                                 # Boolean: Offset/Phase vs distance
    actuator_offset_Gouy(space_C,space_D,sen_off,res,False,False,True,True,True)   # Booleans: Offset/Phase vs distance; Gouy vs distance; Offset vs Gouy; Phase vs Gouy; xk-plot
    plt.show()
    
if __name__ == "__main__":
    main()