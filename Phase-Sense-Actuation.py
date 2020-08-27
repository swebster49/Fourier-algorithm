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
        W = 200                                                                 # Width of window in mm
        xres = 0.01                                                             # 1D array representing kz-space. Derived from condition for monochromaticity.
        N = int(W / xres)                                                       # Number of x-bins (keeps resolution the same)  
        self.idx = int(N/2)
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

    def propagate(self,distance,res=10000,profile=False): # Propagate Beam object through distance with resolution, zres; return Beam object. 
        Uprev = self
        if profile:
            w = self.w                 # unpack width list
            g = self.g                 # unpack gouy-phase list
            z = self.z                 # unpack z-position list
        num = distance // res           # number of steps: divide distance by resolution. 
        rem = distance % res            # remainder of division: final step size. If num = 0, i.e. zres > distance, single step taken, equal to distance. 
        idx = self.idx
        kwav = self.kwav
        p0 = ((np.angle(Uprev.U[idx]) + np.angle(Uprev.U[idx-1]))/2) #% (2 * np.pi) # average phase at centre of distribution
        Uprev.U = Uprev.U * np.exp(-1j * p0)   # re-set phase to zero
        pl = (kwav * res) % (2 * np.pi)
        for i in range(num):            # num steps
            Unext = Uprev.step(res)
            if profile:
                zprev = z[-1]
                z.append(zprev + res)   # Build up z-array as go along. 
                wnext = Gaussfit(Unext.x,abs(Unext.U),1)[2]
                w.append(wnext)
                p1 = ((np.angle(Unext.U[idx]) + np.angle(Unext.U[idx-1]))/2) #% (2 * np.pi)
                g1 = 2 * (pl - p1) % (2 * np.pi) # Fudge-factor of 2 - gives correct answer. 
                g.append(g[-1] + g1)
                Unext.U = Unext.U * np.exp(-1j * p1)
            Uprev = Unext
        pl = (kwav * rem) % (2 * np.pi)
        Unext = Uprev.step(rem)         # Final step of size rem. 
        if profile:
            zprev = z[-1]
            z.append(zprev + rem) 
            wnext = Gaussfit(Unext.x,abs(Unext.U),1)[2]
            w.append(wnext)
            p1 = (np.angle(Unext.U[idx]) + np.angle(Unext.U[idx-1]))/2
            g1 = 2 * (pl - p1) % (2 * np.pi) # Fudge-factor of 2 - gives correct answer. 
            if g1 > 1.9 * np.pi:
                g1 = g1 - 2 * np.pi
            g.append(g[-1] + g1)
            Unext.w = w
            Unext.g = g
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
    U = Beam(w0,z0)
    U = U.propagate(space_0 + space_1 + space_2 + space_3, 50, True)
    U = U.lens(F_L1)
    U = U.propagate(space_4 + space_5 + space_6 + space_7, 50, True)
    width_plot(U.z,U.w)
    gouy_plot(U.z,U.g)

def width_plot(distance_list,width_list,n=3): # Plots beam profile for a given waist array. 
    zplot = 0.001 * np.asarray(distance_list)
    wplot = np.asarray(width_list)
    plt.figure(n,figsize=(4, 3.6), dpi=120)
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
    plt.figure(n,figsize=(4, 3.6), dpi=120)
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

def sensor_offset_distance(act_off,show = False): # Calculates x-offset as function of distance from W1 to sensor for displacement and direction errors applied at W1
    s0 = 3000 - act_off
    s1 = 6000
    n0 = int(s0/100 + 1)
    n1 = int(s1/100 + 1)
    s = np.concatenate((np.linspace(0,s0,n0),np.linspace(s0,s0+s1,n1)), axis = 0)
    Dx_direc = []
    Dx_displ = []
    count = 0
    for i in range(len(s)):
        Dx = sensor(0.0,1.0,int(s[i]),int(s0),count)
        Dx_direc.append(Dx)
        if int(s[i]) == s0:
            count += 1
    #count = 0
    #for i in range(len(s)):
    #    Dx = sensor(1.0,0.0,int(s[i]),int(s0),count)
    #    Dx_displ.append(Dx)
    #    if int(s[i]) == s0:
    #        count += 1
    if show:
        sensor_offset_distance_plot(s,Dx_direc,Dx_displ,act_off)
    return (Dx_direc,Dx_displ)

def sensor_offset_distance_plot(dist,Dx_direc,Dx_displ,act_off,n=5): # Plots x offset as function of distance from W1 to sensor for displacement and direction errors applied at W1
    d_plot = dist / 1000
    a_plot = act_off / 1000
    plt.figure(n, figsize=(4,3.6), dpi=120)
    plt.plot(d_plot,Dx_direc, label = 'direction error')
    #plt.plot(d_plot,Dx_displ, label = 'displacement error')
    plt.xlim([0, 12])
    #plt.ylim([-6, 6])
    plt.grid(which = 'major', axis = 'both')
    plt.title('Actuator fixed, offset %.0f m from waist; move sensor.' % (a_plot,))     
    plt.xlabel('separation between actuator and sensor / m')
    plt.ylabel('x-offset at sensor/ mm')
    #plt.legend(loc = 'upper right')
    plt.tight_layout()

def sensor_propagate(act_off): # Propagates beam from W1 to End
    s0 = (space_2 + space_3 - act_off) / 1000
    s1 = (space_4 + space_5 + space_6 + space_7) / 1000
    z1 = act_off / 1000
    b1 = b / 1000
    S_L1 = 1 / (F_L1 / 1000)
    q0 = beam.beamParameter(z1+1j*b1)    
    q1 = beam.propagate(q0,s0)         
    q2 = beam.lens(q1,S_L1,0.0)       
    q3 = beam.propagate(q2,s1)
    return (q0, q1, q2, q3)

def sensor_distance(act_off): # Develops z-axis array: from W2 to End
    s0 = (space_2 + space_3 - act_off) / 1000
    s1 = (space_4 + space_5 + space_6 + space_7) / 1000
    n0 = int(s0*10 + 1)
    n1 = int(s1*10 + 1)
    z0 = np.linspace(0,s0,n0)                                        
    z1 = np.linspace(s0,s0+s1,n1)
    z = np.concatenate((z0,z1),axis= 0)
    return (z0, z1, z)

def sensor_Gouy(q_params,z_params,act_off): # Develops Gouy-phase array
    q0 = q_params[0]
    q2 = q_params[2]
    z0 = z_params[0]
    z1 = z_params[1]
    s0 = (space_2 + space_3 - act_off) / 1000
    s1 = (space_4 + space_5 + space_6 + space_7) / 1000
    q0_gouy = beam.gouyPhase(q0,z0)\
    - beam.gouyPhase(q0,0.0)
    q2_gouy = beam.gouyPhase(beam.propagate(q2,-(s0)),z1)\
    - beam.gouyPhase(q0,0.0)\
    - (beam.gouyPhase(beam.propagate(q2,-(s0)),s0) - beam.gouyPhase(beam.propagate(q0,-0.0),s0))
    Gouy_Phase = np.concatenate((q0_gouy,q2_gouy),axis = 0)
    return Gouy_Phase

def sensor_Gouy_distance(act_off,show = False): # Optionally plots Gouy phase as a function of distance
    q_params = sensor_propagate(act_off)
    z_params = sensor_distance(act_off)
    z = z_params[2]
    Gouy_Phase = sensor_Gouy(q_params,z_params,act_off)
    if show:
        sensor_Gouy_distance_plot(z, Gouy_Phase, act_off)
    return Gouy_Phase

def sensor_Gouy_distance_plot(z, Gouy_Phase, act_off, n=6):  # Plots Gouy phase as a function of distance
    a_plot = act_off / 1000
    plt.figure(n, figsize=(4, 3.6), dpi=120)
    plt.plot(z, Gouy_Phase*180/np.pi)
    plt.grid(which = 'both', linestyle='--')
    plt.xlim(0,12)
    plt.xticks(np.linspace(0,12,13))
    plt.ylim(0,300)
    plt.yticks(np.linspace(0,300,11))
    plt.grid(which = 'both', linestyle='--')
    plt.title('Actuator fixed, offset %.0f m from waist; move sensor.' % (a_plot,))
    plt.xlabel('separation between actuator and sensor / m')
    plt.ylabel('phase separation / ˚')
    plt.tight_layout()  # otherwise the right y-label is slightly clipped

def sensor_offset_Gouy(act_off,show_offset,show_Gouy):
    Offsets = sensor_offset_distance(act_off,show_offset)
    Gouy_Phase = sensor_Gouy_distance(act_off,show_Gouy)
    sensor_offset_Gouy_plot(Gouy_Phase,Offsets[0],Offsets[1],act_off)

def sensor_offset_Gouy_plot(Gouy_Phase,Dx_direc,Dx_displ,act_off,n=7): # Plots x offset at sensor as a function of phase-separation from W1 with displacement and direction errors applied at W1
    a_plot = act_off / 1000
    plt.figure(n, figsize=(4,3.6), dpi=120)
    plt.plot(Gouy_Phase*180/np.pi, Dx_direc, label = 'direction error')
    #plt.plot(Gouy_Phase*180/np.pi, Dx_displ, label = 'displacement error')
    plt.xlim([0, 300])
    plt.xticks(np.linspace(0,300,11))
    #plt.ylim([-6, 6])
    plt.grid(which = 'major', axis = 'both')
    plt.title('Actuator fixed, offset %.0f m from waist; move sensor.' % (a_plot,))     
    plt.xlabel('phase separation between actuator and sensor / ˚')
    plt.ylabel('x-offset at sensor / mm')
    #plt.legend(loc = 'upper right')
    plt.tight_layout()

def actuator(x0,a0,var_space,s0,count=0): # Propagates beam from W1 to sensor. Variable distance between W1 and sensor. Displacement and direction errors applied at W1. Handles cases with sensor before and after lens. 
    sen_off = s0 - 3000
    if var_space < s0:
        space_B = var_space
        U = Beam(w0,sen_off-var_space,x0,a0)
        U = U.propagate(space_B)
    elif int(var_space) == s0:
        if count == 0:
            space_B = var_space
            U = Beam(w0,sen_off-var_space,x0,a0)
            U = U.propagate(space_B)
        elif count == 1:
            space_B = s0
            space_A = var_space - s0
            U = Beam(w0,6000+sen_off-var_space,x0,a0)
            U = U.propagate(space_A)
            U = U.lens(F_L1)
            U = U.propagate(space_B)
    elif var_space > s0:
        space_B = s0
        space_A = var_space - s0
        U = Beam(w0,6000+sen_off-var_space,x0,a0)
        U = U.propagate(space_A)
        U = U.lens(F_L1)
        U = U.propagate(space_B)
    xparams = U.amp_fit()
    kparams = U.freq_fit()
    Dx = xparams[0] #/ abs(xparams[2])
    Dk = kparams[0] #/ abs(kparams[2])
    return Dx, Dk

def actuator_offset_distance(sen_off,show = False): # Calculates x-offset as function of distance from W1 to sensor for displacement and direction errors applied at W1
    s0 = 3000 + sen_off
    s1 = 6000
    n0 = int(s0/100 + 1)
    n1 = int(s1/100 + 1)
    s = np.concatenate((np.linspace(0,s0,n0),np.linspace(s0,s0+s1,n1)), axis = 0)
    Dx_direc = []
    Dk_direc = []
    count = 0
    for i in range(len(s)):
        Dx, Dk = actuator(0.0,1.0,int(s[i]),int(s0),count)
        Dx_direc.append(Dx)
        Dk_direc.append(Dk)
        if int(s[i]) == s0:
            count += 1
    if show:
        actuator_offset_distance_plot(s,Dx_direc,Dk_direc,sen_off)
    return (Dx_direc,Dk_direc)

def actuator_offset_distance_plot(dist,Dx_direc,Dk_direc,sen_off,n=8): # Plots x- and k- offsets as a function of distance from mirror at which direction correction is applied
    d_plot = dist / 1000
    s_plot = sen_off / 1000
    plt.figure(n, figsize=(4, 3.6), dpi=120)
    plt.plot(d_plot,Dx_direc, label = 'x-offset')
    #plt.plot(d_plot,Dk_direc, label = 'k-offset')
    plt.xlim([0, 12])
    #plt.ylim([-6, 6])
    plt.grid(which = 'major', axis = 'both')
    plt.title('Move actuator; sensor fixed, offset %.0f m from waist.' % (s_plot,))     
    plt.xlabel('separation between actuator and sensor / m')
    plt.ylabel('offset at sensor/ mm')
    #plt.legend(loc = 'upper right')
    plt.tight_layout()

def actuator_propagate(sen_off): # Propagates beam (backwards) from W2 to Start
    s1 = (space_0 + space_1 + space_2 + space_3) / 1000
    s0 = (space_4 + space_5 + sen_off) / 1000
    z1 = - sen_off / 1000
    b1 = b / 1000
    S_L1 = 1 / (F_L1 / 1000)
    q0 = beam.beamParameter(z1+1j*b1)    
    q1 = beam.propagate(q0,s0)         
    q2 = beam.lens(q1,S_L1,0.0)       
    q3 = beam.propagate(q2,s1)
    return (q0, q1, q2, q3)

def actuator_distance(sen_off): # Develops z-axis array: from W2 to Start
    s1 = (space_0 + space_1 + space_2 + space_3) / 1000
    s0 = (space_4 + space_5 + sen_off) / 1000
    n0 = int(s0*10 + 1)
    n1 = int(s1*10 + 1)
    z0 = np.linspace(0,s0,n0)                                        
    z1 = np.linspace(s0,s0+s1,n1)
    z = np.concatenate((z0,z1),axis= 0)
    return (z0, z1, z)

def actuator_Gouy(q_params,z_params,sen_off): # Develops Gouy-phase array
    q0 = q_params[0]
    q2 = q_params[2]
    z0 = z_params[0]
    z1 = z_params[1]
    s1 = (space_0 + space_1 + space_2 + space_3) / 1000
    s0 = (space_4 + space_5 + sen_off) / 1000
    q0_gouy = beam.gouyPhase(q0,z0)\
    - beam.gouyPhase(q0,0.0)
    q2_gouy = beam.gouyPhase(beam.propagate(q2,-(s0)),z1)\
    - beam.gouyPhase(q0,0.0)\
    - (beam.gouyPhase(beam.propagate(q2,-(s0)),s0) - beam.gouyPhase(beam.propagate(q0,-0.0),s0))
    Gouy_Phase = np.concatenate((q0_gouy,q2_gouy),axis = 0)
    return Gouy_Phase

def actuator_Gouy_distance(sen_off,show = False): # Optionally plots Gouy phase as a function of distance
    q_params = actuator_propagate(sen_off)
    z_params = actuator_distance(sen_off)
    z = z_params[2]
    Gouy_Phase = actuator_Gouy(q_params,z_params,sen_off)
    if show:
        actuator_Gouy_distance_plot(z, Gouy_Phase, sen_off)
    return Gouy_Phase

def actuator_Gouy_distance_plot(z, Gouy_Phase, sen_off, n=9):  # Plots Gouy phase as a function of distance
    s_plot = sen_off / 1000    
    plt.figure(n, figsize=(4, 3.6), dpi=120)
    plt.plot(z, Gouy_Phase*180/np.pi)
    plt.grid(which = 'both', linestyle='--')
    plt.xlim(0,12)
    plt.xticks(np.linspace(0,12,13))
    plt.ylim(0,300)
    plt.yticks(np.linspace(0,300,11))
    plt.grid(which = 'both', linestyle='--')
    plt.title('Move actuator; sensor fixed, offset %.0f m from waist.' % (s_plot,))
    plt.xlabel('separation between actuator and sensor / m')
    plt.ylabel('phase separation / ˚')
    plt.tight_layout()  # otherwise the right y-label is slightly clipped

def actuator_offset_Gouy(sen_off,show_offset,show_Gouy): # 
    Offsets = actuator_offset_distance(sen_off,show_offset)
    Gouy_Phase = actuator_Gouy_distance(sen_off,show_Gouy)
    actuator_offset_Gouy_plot(Gouy_Phase,Offsets[0],Offsets[1],sen_off)

def actuator_offset_Gouy_plot(Gouy_Phase,Dx_direc,Dk_direc,sen_off,n=10): # Plots x- and k-offsets at W2 as function of distance mirror at which a direction correction is applied
    s_plot = sen_off / 1000
    plt.figure(n, figsize=(4, 3.6), dpi=120)
    plt.plot(Gouy_Phase*180/np.pi, Dx_direc, label = 'x-offset')
    #plt.plot(Gouy_Phase*180/np.pi, Dk_direc, label = 'k-offset')
    plt.xlim([0, 300])
    plt.xticks(np.linspace(0,300,11))
    #plt.ylim([-6, 6])
    plt.grid(which = 'major', axis = 'both')
    plt.title('Move actuator; sensor fixed, offset %.0f from waist.' % (s_plot,))     
    plt.xlabel('phase separation between actuator and waist / ˚')
    plt.ylabel('offset at sensor/ mm')
    #plt.legend(loc = 'upper right')
    plt.tight_layout()

def main():
    beam_profile()
    #sensor_offset_distance(-2000,True)
    #sensor_Gouy_distance(-2000,True)
    #sensor_offset_Gouy(-2000,True,True)
    #actuator_offset_distance(2000,True)
    #actuator_Gouy_distance(2000,True)
    #actuator_offset_Gouy(2000,True,True)
    plt.show()
    
if __name__ == "__main__":
    main()

'''

def actuator(var_space,count=0): # Propagates beam from 3 m before W1 to W2. Tilt applied at mirror. Variable distance between mirror and W2. Handles cases with mirror before and after lens. 
    a = 0.3
    U = Beam(w0,z0)
    if var_space > 3000:
        space_A = 9000 - var_space
        space_B = var_space - 3000
        space_C = 3000
        U = U.propagate(space_A)
        U = U.tilt(a)
        U = U.propagate(space_B)
        U = U.lens(F_L1)
        U = U.propagate(space_C)
    elif var_space == 3000:
        if count == 0:
            space_A = 9000 - var_space
            space_B = var_space - 3000
            space_C = 3000
            U = U.propagate(space_A)
            U = U.tilt(a)
            U = U.propagate(space_B)
            U = U.lens(F_L1)
            U = U.propagate(space_C)
        elif count == 1:
            space_A = 6000
            space_B = 3000 - var_space
            space_C = var_space
            U = U.propagate(space_A)
            U = U.lens(F_L1)
            U = U.propagate(space_B) 
            U = U.tilt(a)
            U = U.propagate(space_C)
    elif var_space < 3000:
        space_A = 6000
        space_B = 3000 - var_space
        space_C = var_space
        U = U.propagate(space_A)
        U = U.lens(F_L1)
        U = U.propagate(space_B) 
        U = U.tilt(a)
        U = U.propagate(space_C)
    xparams = U.amp_fit()
    kparams = U.freq_fit()
    Dx = xparams[0] / abs(xparams[2]) 
    Dk = kparams[0] / abs(kparams[2])
    return (Dx, Dk)

def actuator_offset_distance(show = False): # Calculates x-and k-offsets as a function of distance from mirror at which direction correction is applied
    s = np.concatenate((np.linspace(9000,3000,61),np.linspace(3000,0,31)), axis = 0)
    Dx_direc = []
    Dk_direc = []
    count = 0
    for i in range(len(s)):
        Dx, Dk = actuator(int(s[i]),count)
        Dx_direc.append(Dx)
        Dk_direc.append(Dk)
        if int(s[i]) == 3000:
            count += 1
    if show:
        actuator_offset_distance_plot(s,Dx_direc,Dk_direc)
    return (Dx_direc,Dk_direc)

def actuator_offset_distance_plot(dist,Dx_direc,Dk_direc,n=4): # Plots x- and k- offsets as a function of distance from mirror at which direction correction is applied
    d_plot = dist / 1000
    plt.figure(n, figsize=(6, 5.5), dpi=120)
    plt.plot(d_plot,Dx_direc, label = 'x-offset')
    plt.plot(d_plot,Dk_direc, label = 'k-offset')
    plt.xlim([0, 9])
    plt.ylim([-3, 3])
    plt.grid(which = 'major', axis = 'both')
    plt.title('Direction correction applied at actuator, measure offsets at waist')     
    plt.xlabel('separation between actuator and waist / m')
    plt.ylabel('offset / 1/e^2 radius')
    plt.legend(loc = 'upper right')
    plt.tight_layout()
'''