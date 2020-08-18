import numpy as np

class beamParameter:
    def __init__(self,*args):
        '''
        initialise the beam parameter with either a complex number (m + 1j*m),
        or a waist size (m) and waist position (m). Currently only works for
        1064nm light
        '''
        # want to include defocus and width setting way of defining beam parameter

        #one args means that the user is defining the beam parameter with a
        #complex number
        if(len(args) == 1):
            if isinstance(args[0],complex):
                self.q = args[0]
            else:
                raise TypeError('Input must be complex')
        #three args means the user is defining waist size and waist position or waist size and defocus
        #if the third arguement is true, then you define the waist size and positions
        #of false, define the defocus and the beam size at the position.
        elif(len(args) == 3):
            if((isinstance(args[0],float) or  isinstance(args[1],float)) and isinstance(args[2],bool)):
                if args[2]:
                    waist_size = args[0]
                    waist_pos = args[1]
                    self.q = 1j*np.pi*waist_size**2/1064e-9 - waist_pos
                else:
                    self.q = 1.0/(args[0] - 1j*1064e-9/(np.pi*args[1]**2))
            else:
                raise TypeError('Inputs must be beamParameter(float,float,bool) or beamParameter(complex)')
        # 3 args means you define

        else:
            raise Exception('incorrect amount of inputs')

    def get_q(self):
        '''
        returns the beam parameter in the form of a row vector
        '''
        return(np.array([self.q,1]))

    def __str__(self):
        return str(self.q)

class astigmaticBeam:
    #need to make this safe!
    def __init__(self,bpt,bps):
        self.qt = bpt
        self.qs = bps

    def get_qt(self):
        return self.qt
    def get_qs(self):
        return self.qs


def propagate(beam,z):
    '''
    takes a beam parameter (type beamParameter) and propagates it z metres
    (type float). Returns a beamParameter
    '''

    #check if inputs are ok
    if not isinstance(z,float):
        raise TypeError('distance should be  and a float')
    if( isinstance(beam,beamParameter)):
        #if the input beam is none astigmatic, a beamParameter object is used
        #for the calculation, and the function returns a beamParameter
        q1 = beam.get_q() # get beam beamParameter row vector
        #matrix multiply to get a column vector
        q2 = np.dot(q1,np.array([[1,0],[z,1]]))
        #make a new beamParameter from the top and bottom of the column vector
        return beamParameter(q2[0]/q2[1])
    elif(isinstance(beam,astigmaticBeam)):
        #if the input beam is an astigmatic one, a astigmaticBeam object is used
        #for the calculation, and the function returns a astigmatic beam.
        #get the beam parameter row vector and matrix multiply to get the transformed
        #beam
        q1t = beam.qt.get_q()
        q1s = beam.qs.get_q()

        q2t = np.dot(q1t,np.array([[1,0],[z,1]]))
        q2s = np.dot(q1s,np.array([[1,0],[z,1]]))
        return(astigmaticBeam(beamParameter(q2t[0]/q2t[1]),beamParameter(q2s[0]/q2s[1])))
    else:
        raise TypeError('inputs should be a beamParameter object or a astigmaticBeam, and a float')

def lens(beam,power,angle):                                                                    # Added lens function. Basically the same as mirror.
    '''
    takes a beam parameter (type beamParameter) and calculates the new one after               # Defocus = 1/R?
    transmission through a thin lens with power diopters (type float). Takes an angle in       # Note: Power = 1/f = 2/R
    deg (type float). (type float). Returns 3 beamParameters, the none astigmatic
    one, the tangential, and the saggital.
    '''
    if not(isinstance(power,float) and isinstance(angle,float)):
        raise('inputs should be a beam parameter object and a float')
    if( isinstance(beam,beamParameter)):
        #if the input beam is none astigmatic, a beamParameter object is used
        #for the calculation, and the function returns a beamParameter
        q1 = beam.get_q() # get beam beamParameter row vector
        #matrix multiply to get a column vector
        q2 = np.dot(q1,np.array([[1,-power],[0,1]]))                                           # Power = 2/R is the correct quantity here.
        #make a new beamParameter from the top and bottom of the column vector
        return(beamParameter(q2[0]/q2[1]))
    elif( isinstance(beam,astigmaticBeam)):
        #if the input beam is an astigmatic one, a astigmaticBeam object is used
        #for the calculates, and the function returns a astigmatic beam.
        #get the beam parameter row vector and matrix multiply to get the transformed
        #beam
        q2t = np.dot(beam.qt.get_q(),np.array([[1,-power/np.cos(angle/180*np.pi)],[0,1]]))
        q2s = np.dot(beam.qs.get_q(),np.array([[1,-power*np.cos(angle/180*np.pi)],[0,1]]))
        #contruct a astigmatic beam from the beam Parameters obtained above
        return(astigmaticBeam(beamParameter(q2t[0]/q2t[1]),beamParameter(q2s[0]/q2s[1])))
    else:
        raise TypeError('inputs should be a beam parameter object, float and float')

def thicklens(beam,S1,S2,T,n):                                                                    
    '''
    takes a beam parameter (type beamParameter) and calculates the new one after        
    transmission through a thick lens surfaces having defoci, S1 and S2, a thickness, T,
    and made from a material with refractive index, n.
    Returns 3 beamParameters, the none astigmatic
    one, the tangential, and the saggital.
    '''
    if not(isinstance(S1,float) and isinstance(S2,float) and isinstance(T,float) and isinstance(n,float)):
        raise('inputs should be a beam parameter object and floats')
    q1 = beam.get_q() # get beam beamParameter row vector
    #matrix multiply to get a column vector
    q2 = np.dot(q1,np.array([[(1 + S1 * T * (n-1) / n), ((n-1) * (S1 - S2) - S1 * S2 * T * (n-1)**2 / n)],[(T/n), (1 - S2 * T * (n-1) / n)]]))                                           # Power = 2/R is the correct quantity here.
    #make a new beamParameter from the top and bottom of the column vector
    return(beamParameter(q2[0]/q2[1]))

def mirror(beam,power,angle):
    '''
    takes a beam parameter (type beamParameter) and calculates the new one after
    reflection from a curved mirror with defocus in 1/m (type float). Takes an angle in
    deg (type float). Returns 3 beamParameters, the none astigmatic
    one, the tangential, and the saggital.
    '''
    if not(isinstance(power,float) and isinstance(angle,float)):
        raise('inputs should be a beam parameter object and a float')
    if( isinstance(beam,beamParameter)):
        #if the input beam is none astigmatic, a beamParameter object is used
        #for the calculation, and the function returns a beamParameter
        q1 = beam.get_q() # get beam beamParameter row vector
        #matrix multiply to get a column vector
        q2 = np.dot(q1,np.array([[1,-power],[0,1]]))
        #make a new beamParameter from the top and bottom of the column vector
        return(beamParameter(q2[0]/q2[1]))
    elif( isinstance(beam,astigmaticBeam)):
        #if the input beam is an astigmatic one, a astigmaticBeam object is used
        #for the calculates, and the function returns a astigmatic beam.
        #get the beam parameter row vector and matrix multiply to get the transformed
        #beam
        q2t = np.dot(beam.qt.get_q(),np.array([[1,-power/np.cos(angle/180*np.pi)],[0,1]]))
        q2s = np.dot(beam.qs.get_q(),np.array([[1,-power*np.cos(angle/180*np.pi)],[0,1]]))
        #contruct a astigmatic beam from the beam Parameters obtained above
        return(astigmaticBeam(beamParameter(q2t[0]/q2t[1]),beamParameter(q2s[0]/q2s[1])))
    else:
        raise TypeError('inputs should be a beam parameter object, float and float')

# need to add safety features

def beamSize(bp,z = None):
    '''
    takes a beam beamParameter (type beamParameter) and returns the beam size
    of the beam (m).
    '''

    #check if inputs are beam parameter or a list of beam parameters
    if not (isinstance(bp,beamParameter) or isinstance(bp,list)):
         raise TypeError('input must be type beamParameter or a list of beamParameter')

    if(isinstance(bp,list)):
        #this is for many beam parameters entered as np.array
        beamSizes = np.empty(len(bp))
        for idx,bpi in enumerate(bp):
            beamSizes[idx] = beamSize(bp[idx],z)
        return beamSizes
    if z is None:
        return np.sqrt(1064e-9/(-np.pi*np.imag(1/bp.q)))
    else:
        beamSize_z = np.zeros(len(z))
        for idx, zi in enumerate(z):
            bp_prop = propagate(bp,zi)
            beamSize_zi = beamSize(bp_prop)
            beamSize_z[idx] = beamSize_zi
        return beamSize_z

#hmmm
def beamDefocus(bp):
    '''
    takes a beam beamParameter (type beamParameter) and returns the ROC of the
    beam (m).
    '''
    #check if inputs are beam parameter or a list of beam parameters
    if not (isinstance(bp,beamParameter) or isinstance(bp,list)):
         raise TypeError('input must be type beamParameter or a list of beamParameter')
    if(isinstance(bp,list)):
        #this is for many beam parameters entered as np.array
        beamDefocuses = np.empty(len(bp))
        for idx,bpi in enumerate(bp):
            beamDefocuses[idx] = beamDefocus(bp[idx])
        return beamDefocuses
    return np.real(1/bp.q)

def waistSize(bp):
    '''
    takes a beam beamParameter (type beamParameter) and returns the waist size
    of the beam (m)
    '''
    if not isinstance(bp,beamParameter):
         raise TypeError('input must be type beamParameter')
         #kind of weird, propagate the beam back to the origin and calculate
         #beam size of this new beam at the origin
    q0 =  propagate(bp,waistLocation(bp))
    return beamSize(q0)

def waistLocation(bp):
    '''
    takes a beam beamParameter (type beamParameter) and returns the waist location
    of the beam (m)
    '''
    if not isinstance(bp,beamParameter):
         raise TypeError('input must be type beamParameter')
    return -np.real(bp.q)

def zr(bp):
    '''
    takes a beam beamParameter (type beamParameter) and returns the Rayliegh
    range (m)
    '''
    if not isinstance(bp,beamParameter):
         raise TypeError('input must be type beamParameter')
    return np.imag(bp.q)

def modematching(bp1,bp2):
    if(isinstance(bp1,astigmaticBeam) and isinstance(bp2,astigmaticBeam)):
        #this calculation always returns a real number, this is to stpo a warning
        return np.real(np.sqrt((bp1.get_qt().q - np.conj(bp1.get_qt().q))*\
            (bp1.get_qs().q - np.conj(bp1.get_qs().q))*\
            (bp2.get_qt().q - np.conj(bp2.get_qt().q))*\
            (bp2.get_qs().q - np.conj(bp2.get_qs().q)))/\
            (np.absolute(bp2.get_qs().q - np.conj(bp1.get_qs().q))*\
            np.absolute(bp2.get_qt().q - np.conj(bp1.get_qt().q))))
    elif(isinstance(bp1,beamParameter) and isinstance(bp2,beamParameter)):
        return modematching(astigmaticBeam(bp1,bp1),astigmaticBeam(bp2,bp2))
    elif(isinstance(bp1,astigmaticBeam) and isinstance(bp2,beamParameter)):
        return modematching(bp1,astigmaticBeam(bp2,bp2))
    elif(isinstance(bp1,beamParameter) and isinstance(bp2,astigmaticBeam)):
        return modematching(astigmaticBeam(bp1,bp1),bp2)
    else:
        raise TypeError('Use beamParameter and or astigmaticBeam')

def wsContour(W,S,qt):
    '''
    returns the WS phase space to help visualise mode matching. takes beam widths
    (W) in meters, defocus (S) in dioptre and a beam parameter. I think this
    might even work for astigmatic beams as well since it uses the modematching
    function.
    '''
    # haven't impleneted  creating beam parameters from size and defocus, since this
    # is how you define 1/beam parameter (which is sort of like the same thing...)
    OL = np.ones((len(S),len(W)))
    for i, Si in enumerate(S):
        for j, Wj in enumerate(W):
            q_test = beamParameter(Si,Wj,False)
            OL[i,j] = modematching(q_test,qt)
    return OL

def gouyPhase(bp,z = None):
    '''
    returns the gouy phase for a beam parameter bp. if a z is provided, it calculates
    it at z (works for z being a np.arrays or float)
    '''
    if not isinstance(bp,beamParameter):
         raise TypeError('input must be type beamParameter')
    if z is None:
        return np.arctan(np.real(bp.q)/np.imag(bp.q))
    elif isinstance(z,float):                               # Included this case as it wasn't accepting a float.
        bp_prop = propagate(bp,z)
        return gouyPhase(bp_prop)
    else:
        gouphase_z = np.zeros(len(z))
        for idx, zi in enumerate(z):
            bp_prop = propagate(bp,zi)
            gouphase_zi = gouyPhase(bp_prop)
            gouphase_z[idx] = gouphase_zi
        return gouphase_z

def closest(input_array, value):
    ''' Takes input array and value
        and returns element of array
        closest to that value.
        Needed for mapping function. '''
    for i in range(len(input_array)):
        if input_array[i] > value:
            return (input_array[i-1], input_array[i])
    raise Exception('Value outside range of elements of array. ')

def mapping(array1,array2):
    ''' Takes two equal length arrays as inputs.
        Creates one-to-one mapping between their elements.
        Returns dictionary with keys as elements of one array
        and values as elements of the other array. '''
    if len(array1) != len(array2):
         raise Exception('Input arrays must be of equal length. ')
    d = {}
    for i in range(len(array2)):
        d[array1[i]] = array2[i]
    return d

def interpolate(array1, array2, value):
    ''' Takes two equal length arrays and value corresponding to the quantity represented by the first array.
        Interpolates to get value of the second array at the value given. 
        Return this value. '''
    if len(array1) != len(array2):
        raise Exception('Input arrays must be of equal length. ')
    if not(isinstance(array1,np.ndarray) and isinstance(array2,np.ndarray) and isinstance(value,float)):
        raise('inputs should be two nd arrays and a float')
    x = array1
    y = array2
    x_val = value
    dictionary = mapping(x,y)
    x_cl_val = closest(x, x_val)
    grad = (dictionary[x_cl_val[1]] - dictionary[x_cl_val[0]])/(x_cl_val[1] - x_cl_val[0])
    y_val = dictionary[x_cl_val[0]] + grad * (x_val - x_cl_val[0])
    return y_val

def gouyASC(array1, array2, tuple1):
    ''' Takes two equal length arrays and tuple as inputs.
        Arrays are the z co-ordinate and the Gouy phase along the beam path
        Tuple contains the z positions of the beam-steering mirrors which are furthest apart.
        Interpolates to get Gouy phase at the two mirror positions
        Returns these values. '''
    if len(array1) != len(array2):
        raise Exception('Input arrays must be of equal length. ')
    if not(isinstance(array1,np.ndarray) and isinstance(array2,np.ndarray) and isinstance(tuple1,tuple)):
        raise('inputs should be two nd arrays and a 2-valued tuple')
    z = array1
    gp = array2
    z_min = tuple1[0]
    z_max = tuple1[1]
    gouy_dict = mapping(z,gp)
    z_cl_min = closest(z, z_min)
    z_cl_max = closest(z, z_max)
    grad_min = (gouy_dict[z_cl_min[1]] - gouy_dict[z_cl_min[0]])/(z_cl_min[1] - z_cl_min[0])
    grad_max = (gouy_dict[z_cl_max[1]] - gouy_dict[z_cl_max[0]])/(z_cl_max[1] - z_cl_max[0])
    gouy_min = gouy_dict[z_cl_min[0]] + grad_min * (z_min - z_cl_min[0])
    gouy_max = gouy_dict[z_cl_max[0]] + grad_max * (z_max - z_cl_max[0])
    return (gouy_min, gouy_max)
