#!/usr/bin/env python3
# coding=UTF-8

import sys, time
from math import *
import scipy.constants
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model


def ReorderBeamMatrix(A_in):
    """
    convert between the classical ordering of pixels on a screen
    and the ordering used in a beam (optimized for FFT transport)
    where the center is mapped to (0,0) and the negative positions
    are wrapped around to the highest indizes
    """
    (nx, ny) = A_in.shape
    # names correspond to the quadrant in the target field
    # idexing: (left -> right, top -> down)
    ll = A_in[nx//2:nx, 0:ny//2]
    lr = A_in[0:nx//2, 0:ny//2]
    ul = A_in[nx//2:nx, ny//2:ny]
    ur = A_in[0:nx//2, ny//2:ny]
    l = np.concatenate((ll,lr), axis=0)
    u = np.concatenate((ul,ur), axis=0)
    return np.concatenate((u,l), axis=1)

def ImportTeufelSingleFrequency(filename, freq):
    """
    Create SingleComponentBeam() objects importing fields on a rectangular screen
    from a TEUFEL calculation. A list of two beams [beam_S, beam_P] is returned
    containing the two polarization directions.
    """
    hdf = h5py.File(filename, "r")
    # Get the groups
    pos = hdf['ObservationPosition']
    Nx = pos.attrs.get('Nx')
    Ny = pos.attrs.get('Ny')
    print("Nx=%d Ny=%d" % (Nx,Ny))
    field = hdf['ElMagField']
    t0 = field.attrs.get('t0')
    dt = field.attrs.get('dt')
    nots = field.attrs.get('NOTS')
    print("t0=%g dt=%g NOTS=%d" % (t0, dt, nots))
    pos = np.array(pos)
    A = np.array(field)
    hdf.close()
    # delta_x and delta_y are vectors
    delta_x = (pos[Nx-1,Ny//2] - pos[0,Ny//2]) / (Nx-1)
    delta_y = (pos[Nx//2,Ny-1] - pos[Nx//2,0]) / (Ny-1)
    # the pixel spacing
    dx = sqrt(np.sum(np.square(delta_x)))
    dy = sqrt(np.sum(np.square(delta_y)))
    # unit vectors
    e_x = delta_x / dx
    e_y = delta_y / dy
    # index in the fourier spectrum
    f_index = int(round(freq*dt*nots))
    
    beam_S = SingleComponentBeam(freq, Nx, Ny, dx, dy)
    M = np.zeros((Nx,Ny),dtype=np.cdouble)
    for ix in range(Nx):
        for iy in range(Ny):
            trace = A[ix][iy]
            data = trace.transpose()
            Ex = data[0]
            Ey = data[1]
            Ez = data[2]
            EVec = np.array([Ex, Ey, Ez]).transpose()
            E = np.dot(EVec,e_x)
            spect = np.fft.fft(E)
            M[ix,iy] = spect[f_index]
    beam_S.A = ReorderBeamMatrix(M)

    beam_P = SingleComponentBeam(freq, Nx, Ny, dx, dy)
    M = np.zeros((Nx,Ny),dtype=np.cdouble)
    for ix in range(Nx):
        for iy in range(Ny):
            trace = A[ix][iy]
            data = trace.transpose()
            Ex = data[0]
            Ey = data[1]
            Ez = data[2]
            EVec = np.array([Ex, Ey, Ez]).transpose()
            E = np.dot(EVec,e_y)
            spect = np.fft.fft(E)
            M[ix,iy] = spect[f_index]
    beam_P.A = ReorderBeamMatrix(M)

    return [beam_S, beam_P]

def HuygensFresnel(source, target, dist):
    """
    Direct computation of the Huygens-Fresnel integral serves as a reference
    for all propagation methods. The routine expects a source beam and
    a separate target beam which defines the geometry of the output.
    The intensity/phase distribution of the target field is computed and overwritten.
    The performance of the algorithm is not suited for larger fields.
    """
    # check for correct type of the input
    λ = scipy.constants.c/source.freq
    k = 2*pi/λ
    # compute integral
    for tx in range(target.nx):
        for ty in range(target.ny):
            target.A[tx,ty] = 0.0
            for sx in range(source.nx):
                for sy in range(source.ny):
                    x_squared = pow(target.x(tx)-source.x(sx),2)
                    y_squared = pow(target.y(ty)-source.y(sy),2)
                    target.A[tx,ty] += source.A[sx,sy] * -1.0j/λ/dist * \
                        np.exp(1.0j*k/(2.0*dist)*(x_squared+y_squared))
    target.freq = source.freq
    target.A *= source.dx*source.dy
               
class SingleComponentBeam():
    """
    This class describes an optical beam sampled at on observation plane
    perpendicular to the direction of propagation. Only one frequency
    component is described, for polychromatic beams several objects of
    this class need to be combined.

    The optical beam is given by an array of complex amplitudes on a rectangular grid.
    The amplitude refers to the elektric field strength. The power density is the
    square of the absolute value of the amplitude.
    The dimensions of that grid are given when creating the beam object.
    Both values should be integer powers of 2 in order to allow fast Fourier transforms.

    The amplitudes are stored in a 2-dim complex-valued NumPy array.
    To ease the Fourier transforms the array are stored in an unconventional order.
    The index=0 elements always refer to the center of the beam (propagation axis).
    The indices 1...N/2-1 refer to positive displacements off-axis.
    The indices N/2...N-1 contain negative displacement values with N-1 being
    the pixel closest to the axis (-1*dx).
    Essentially, the high-intensity central part of the beam is stored in the
    corners of the matrix.
    """
    
    def __init__(self, f, nx, ny, dx, dy):
        """
        Create an empty SingleFrequencyBeam object with radiation frequency f.
        nx/ny are the horizontal/vertical field sizes (should be powers of 2 for FFT).
        dx/dy are the corresponding pixel sizes.
        """
        self.freq = f
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.A = np.zeros((nx,ny),dtype=np.cdouble)
    
    @classmethod
    def PlaneWave(cls, f, nx, ny, dx, dy):
        """
        Create a plane wave with frequency f.
        nx/ny are the horizontal/vertical field sizes (should be powers of 2 for FFT).
        dx/dy are the corresponding pixel sizes.
        """
        beam = cls(f, nx, ny, dx, dy)
        beam.A = np.ones((nx,ny),dtype=np.cdouble)
        beam.Normalize()
        return beam
    
    @classmethod
    def GaussianBeamWaist(cls, f, nx, ny, dx, dy, w0x, w0y):
        """
        Create a gaussian beam with width parameters w0x/w0y
        in horizontal/vertical direction.
        """
        beam = cls(f, nx, ny, dx, dy)
        λ = scipy.constants.c/f
        print("λ = %f mm" % (1e3*λ))
        k = 2*pi/λ
        print("k = %f m⁻¹" % k)
        sx2 = pow(w0x,2)
        sy2 = pow(w0y,2)
        for ix in range(nx):
            for iy in range(ny):
                a = exp(-pow(beam.x(ix),2)/sx2 - pow(beam.y(iy),2)/sy2)
                beam.A[ix,iy] = a
        beam.Normalize()
        return beam

    @classmethod
    def GaussianBeam(cls, f, nx, ny, dx, dy, zR, z):
        """
        Create a gaussian beam with a Rayleigh range zR
        at a distance z from the waist. z cannot be exactly zero.
        """
        beam = cls(f, nx, ny, dx, dy)
        λ = scipy.constants.c/f
        print("λ = %f mm" % (1e3*λ))
        k = 2*pi/λ
        print("k = %f m⁻¹" % k)
        w0 = sqrt(zR*λ/pi)
        print("w0 = %f mm" % (1e3*w0))
        w = w0 * sqrt(1+pow(z/zR,2))
        print("w = %f mm" % (1e3*w))
        R = z*(1.0 + pow(zR,2)/pow(z,2))
        print("R = %f m" % R)
        zeta = atan(z/zR)
        for ix in range(nx):
            for iy in range(ny):
                r2 = pow(beam.x(ix),2) + pow(beam.y(iy),2)
                a = np.exp( -r2/pow(w,2) - 1.0j*(k*z-zeta) - 1.0j*k*r2/(2.0*R) )
                beam.A[ix,iy] = a
        beam.Normalize()
        return beam

    def NearFieldProp(self, dist):
        """
        Transport this beam over a certain distance using the near-field angular propagation method.
        The created beam has the same geometrical properties as the source beam.

        Diffraction propagation is computed based on the propagation of plane waves.
        A Fourier transform is used to decompose the complex amplitude distribution
        into a set of plane waves with different propagation direction (transverse
        wave number). That's why this method is also called angular spectrum propagation.
        The phase factors for arrival of these plane waves are easily computed
        from the direktion of propagation. After applying these phase factors
        the set is retransformed into aplitude distribution after the propagation step.

        Return: propagated SingleComponentBeam object
        """
        k = 2*pi*self.freq/scipy.constants.c
        nx = self.nx
        ny = self.ny
        dkx = 2.0*pi/self.dx/nx
        dky = 2.0*pi/self.dy/ny
        # create the new beam
        beam = SingleComponentBeam(self.freq, nx, ny, self.dx, self.dy)
        # Fourier transform into momentum space
        FF = np.fft.fft2(self.A)
        # apply the phase factors according to the propagation length
        FFD = np.zeros_like(FF)
        for ix in range(nx):
            for iy in range(ny):
                FFD[ix,iy] = FF[ix,iy] * ( np.exp(-1.0j*k*dist) *
                    np.exp(1.0j*(pow(beam.xi(ix),2)+pow(beam.eta(iy),2))*dist/(2.0*k)) )
        # back-transform into position space
        beam.A = np.fft.ifft2(FFD)
        return beam
    
    def FarFieldProp(self, dist):
        """
        Transport this beam over a certain distance using the far-field propagation method.

        Diffraction propagation is computed based on the propagation of plane waves.
        A Fourier transform is used to compute the convolution of the input distribution
        with a diffraction kernel.

        Return: propagated SingleComponentBeam object
        """
        k = 2*pi*self.freq/scipy.constants.c
        nx = self.nx
        ny = self.ny
        # create the new beam
        dx2 = fabs(2.0*pi*dist/(k*nx*self.dx))
        dy2 = fabs(2.0*pi*dist/(k*ny*self.dy))
        if (dx2<self.dx) or (dy2<self.dy):
            print("Warning: not in far field, reducing field size")
            print("dx : %.1g => %.1g" % (self.dx,dx2))
            print("dy : %.1g => %.1g" % (self.dy,dy2))
        beam = SingleComponentBeam(self.freq, nx, ny, dx2, dy2)
        # apply the internal phase factor according to the propagation length
        FA = np.zeros_like(self.A)
        for ix in range(nx):
            for iy in range(ny):
                FA[ix,iy] = self.A[ix,iy] * \
                    np.exp(-1.0j*k/2.0/dist*(pow(self.x(ix),2)+pow(self.y(iy),2)) )
        # Fourier transform
        if dist>0:
            FFA = np.fft.ifft2(FA) * nx*ny
        else:
            FFA = np.fft.fft2(FA)
        # apply the external phase factor according to the propagation length
        for ix in range(nx):
            for iy in range(ny):
                beam.A[ix,iy] = FFA[ix,iy] * \
                    -2.0*pi*k/1.0j/dist * \
                    np.exp(-1.0j*k*dist) * \
                    np.exp(-1.0j*k/2.0/dist*(pow(beam.x(ix),2)+pow(beam.y(iy),2)) )
        beam.A *= self.dx*self.dy / (4*pi*pi)
        return beam

    def shape(self):
        """
        Report the shape of the beam matrix.
        """
        return self.A.shape
        
    def x(self, ix):
        """
        Horizontal position of the pixel center with given index ix.
        Indeces up to nx/2-1 map to positive x positions in increasing order starting at zero.
        Indeces starting from nx/2 map to negative x positions ending with -dx.
        """
        if ix > self.nx//2 :
            x = self.dx*(ix-self.nx)
        else :
            x = self.dx*ix
        return x
    
    def y(self, iy):
        """
        Verticalal position of the pixel center with given index iy.
        Indeces up to ny/2-1 map to positive y positions in increasing order starting at zero.
        Indeces starting from ny/2 map to negative y positions ending with -dy.
        """
        if iy > self.ny//2 :
            y = self.dy*(iy-self.ny)
        else :
            y = self.dy*iy
        return y

    def xi(self, ix):
        """
        Angular coordinate of the Fourier-space pixel with given index ix.
        """
        dkx = 2.0*pi/self.dx/self.nx
        if ix > self.nx//2 :
            xi = dkx*(ix-self.nx)
        else :
            xi = dkx*ix
        return xi
    
    def eta(self, iy):
        """
        Angular coordinate of the Fourier-space pixel with given index iy.
        """
        dky = 2.0*pi/self.dy/self.ny
        if iy > self.ny//2 :
            eta = dky*(iy-self.ny)
        else :
            eta = dky*iy
        return eta
    
    def pad(self, Nx_target, Ny_target):
        """
        Pad the field with zero-intensity pixels to reach the given shape of the matrix.
        """
        if Nx_target > self.nx:
            # cut into two horizontal parts
            M1 = self.A[0:self.nx//2, :]
            M2 = self.A[self.nx//2:self.nx, :]
            # and insert zeros up to the intended shape
            M0 = np.zeros((Nx_target-self.nx,self.ny))
            M = np.concatenate((M1,M0,M2),axis=0)
            self.nx = Nx_target
        else:
            M = self.A
        if Ny_target > self.ny:
            # cut into two vertical parts
            M1 = M[:, 0:self.ny//2]
            M2 = M[:, self.ny//2:self.ny]
            # and insert zeros up to the intended shape
            M0 = np.zeros((self.nx,Ny_target-self.ny))
            self.A = np.concatenate((M1,M0,M2),axis=1)
            self.ny = Ny_target
        else:
            self.A = M
        
    def crop(self, Nx_target, Ny_target):
        """
        Cut the outer parts of the field to reach the given shape of the matrix.
        """
        if Nx_target < self.nx:
            # cut into two horizontal parts
            M1 = self.A[0:(Nx_target+1)//2, :]
            M2 = self.A[self.nx-Nx_target//2:self.nx, :]
            # re-assemble
            M = np.concatenate((M1,M2),axis=0)
            self.nx = Nx_target
        else:
            M = self.A
        if Ny_target < self.ny:
            # cut into two horizontal parts
            M1 = M[:, 0:(Ny_target+1)//2]
            M2 = M[:, self.ny-Ny_target//2:self.ny]
            # re-assemble
            self.A = np.concatenate((M1,M2),axis=1)
            self.ny = Ny_target
        else:
            self.A = M

    def CircularAperture(self, R, x0=0.0, y0=0.0, Nsampl=4):
        """
        The radiation field is zeroed outside a circle with radius R about a point (x0,y0).
        For all pixels which are not fully inside or outside the circle
        oversampling is used to determine a fractional ratio of attenuation.
        The fractional attenuation is correct for the power, not the amplitude.
        """
        for ix in range(self.nx):
            for iy in range(self.ny):
                # check the 4 corner points of the pixel
                t1 = pow(self.x(ix-0.5)-x0,2)+pow(self.y(iy-0.5)-y0,2)<pow(R,2)
                t2 = pow(self.x(ix-0.5)-x0,2)+pow(self.y(iy+0.5)-y0,2)<pow(R,2)
                t3 = pow(self.x(ix+0.5)-x0,2)+pow(self.y(iy-0.5)-y0,2)<pow(R,2)
                t4 = pow(self.x(ix+0.5)-x0,2)+pow(self.y(iy+0.5)-y0,2)<pow(R,2)
                # if all are in the circle -> unchanged
                if all([t1,t2,t3,t4]):
                    pass
                # if all are out off the circle -> 0.0
                elif not any([t1,t2,t3,t4]):
                    self.A[ix][iy] = 0.0
                # over-sample on a grid of Nsampl*Nsampl points and average
                else:
                    m = 0.0
                    positions = np.arange(-(Nsampl-1)/(2*Nsampl), Nsampl/(2*Nsampl), 1/Nsampl)
                    for dx in positions:
                        for dy in positions:
                            if pow(self.x(ix+dx)-x0,2)+pow(self.y(iy+dy)-y0,2)<pow(R,2):
                                m += 1.0/(Nsampl*Nsampl)
                    self.A[ix][iy] *= sqrt(m)
        
    def EllipticAperture(self, Rhor, Rvert, Nsampl=4):
        """
        The radiation field is zeroed outside an ellipse with half axis length Rhor, Rvert.
        For all pixels which are not fully inside or outside the ellipse
        oversampling is used to determine a fractional ratio of attenuation.
        The fractional attenuation is correct for the power, not the amplitude.
        """
        rh2 = Rhor*Rhor
        rv2 = Rvert*Rvert
        for ix in range(self.nx):
            for iy in range(self.ny):
                # check the 4 corner points of the pixel
                t1 = pow(self.x(ix-0.5),2)/rh2 + pow(self.y(iy-0.5),2)/rv2 < 1.0
                t2 = pow(self.x(ix-0.5),2)/rh2 + pow(self.y(iy+0.5),2)/rv2 < 1.0
                t3 = pow(self.x(ix+0.5),2)/rh2 + pow(self.y(iy-0.5),2)/rv2 < 1.0
                t4 = pow(self.x(ix+0.5),2)/rh2 + pow(self.y(iy+0.5),2)/rv2 < 1.0
                # if all are in the circle -> unchanged
                if all([t1,t2,t3,t4]):
                    pass
                # if all are out off the circle -> 0.0
                elif not any([t1,t2,t3,t4]):
                    self.A[ix][iy] = 0.0
                # over-sample on a grid of Nsampl*Nsampl points and average
                else:
                    m = 0.0
                    positions = np.arange(-(Nsampl-1)/(2*Nsampl), Nsampl/(2*Nsampl), 1/Nsampl)
                    for dx in positions:
                        for dy in positions:
                            if pow(self.x(ix+dx),2)/rh2 + pow(self.y(iy+dy),2)/rv2 < 1.0:
                                m += 1.0/(Nsampl*Nsampl)
                    self.A[ix][iy] *= sqrt(m)

    def RectangularAperture(self, hor, vert):
        """
        The radiation field is zeroed outside an aperture of hor/vert width.
        At the edges fractional attenuation is used if pixels are not fully
        inside or outside the aperture.
        The fractional attenuation is correct for the power, not the amplitude.
        """
        hh = hor/2.0
        hv = vert/2.0
        hdx = self.dx/2.0
        hdy = self.dy/2.0
        for ix in range(self.nx):
            for iy in range(self.ny):
                x = self.x(ix)
                y = self.y(iy)
                if x+hdx > hh:
                    if x-hdx > hh:
                        self.A[ix][iy] = 0.0
                    else:
                        self.A[ix][iy] *= sqrt((hh-x+hdx)/self.dx)
                if x-hdx < -hh:
                    if x+hdx < -hh:
                        self.A[ix][iy] = 0.0
                    else:
                        self.A[ix][iy] *= sqrt((hh+x+hdx)/self.dx)
                if y+hdy > hv:
                    if y-hdy > hv:
                        self.A[ix][iy] = 0.0
                    else:
                        self.A[ix][iy] *= sqrt((hv-y+hdy)/self.dy)
                if y-hdy < -hv:
                    if y+hdy < -hv:
                        self.A[ix][iy] = 0.0
                    else:
                        self.A[ix][iy] *= sqrt((hv+y+hdy)/self.dy)

    def CircularObscurance(self, R, x0=0.0, y0=0.0, Nsampl=4):
        """
        The radiation field is zeroed inside a circle with radius R about a point (x0,y0).
        For all pixels which are not fully inside or outside the circle
        oversampling is used to determine a fractional ratio of attenuation.
        The fractional attenuation is correct for the power, not the amplitude.
        """
        for ix in range(self.nx):
            for iy in range(self.ny):
                # check the 4 corner points of the pixel
                t1 = pow(self.x(ix-0.5)-x0,2)+pow(self.y(iy-0.5)-y0,2)<pow(R,2)
                t2 = pow(self.x(ix-0.5)-x0,2)+pow(self.y(iy+0.5)-y0,2)<pow(R,2)
                t3 = pow(self.x(ix+0.5)-x0,2)+pow(self.y(iy-0.5)-y0,2)<pow(R,2)
                t4 = pow(self.x(ix+0.5)-x0,2)+pow(self.y(iy+0.5)-y0,2)<pow(R,2)
                # if all are in the circle -> 0.0
                if all([t1,t2,t3,t4]):
                    self.A[ix][iy] = 0.0
                # if all are out off the circle -> unchanged
                elif not any([t1,t2,t3,t4]):
                    pass
                # over-sample on a grid of Nsampl*Nsampl points and average
                else:
                    m = 0.0
                    positions = np.arange(-(Nsampl-1)/(2*Nsampl), Nsampl/(2*Nsampl), 1/Nsampl)
                    for dx in positions:
                        for dy in positions:
                            if pow(self.x(ix+dx)-x0,2)+pow(self.y(iy+dy)-y0,2)>pow(R,2):
                                m += 1.0/(Nsampl*Nsampl)
                    self.A[ix][iy] *= sqrt(m)
        
    def MirrorH(self):
        """
        Mirror the field horizonzally. The phase jump is ignored.
        """
        for ix in range(self.nx//2):
            for iy in range(self.ny):
                tmp = self.A[self.nx-ix-1][iy]
                self.A[self.nx-ix-1][iy] = self.A[ix][iy]
                self.A[ix][iy] = tmp
        
    def MirrorV(self):
        """
        Mirror the field horizonzally. The phase jump is ignored.
        """
        for ix in range(self.nx):
            for iy in range(self.ny//2):
                tmp = self.A[ix][self.ny-iy-1]
                self.A[ix][self.ny-iy-1] = self.A[ix][iy]
                self.A[ix][iy] = tmp
        
    def ThinLens(self, f):
        """
        Apply the quadratic phase factors describing a thin lens with focal length f.
        """
        k = 2*pi*self.freq/scipy.constants.c
        for ix in range(self.nx):
            for iy in range(self.ny):
                # radius from the axis
                r2 = pow(self.x(ix),2) + pow(self.y(iy),2)
                # path length deviation
                dl = r2 / (2.0*f)
                self.A[ix,iy] *= np.exp(1.0j*k*dl)        
        
    def TotalPower(self):
        """
        Sum up all the intensity in the beam.
        The power density is the square of the absolute amplitude.
        """
        return np.sum(np.square(np.abs(self.A)))*self.dx*self.dy

    def Normalize(self):
        """
        Normalize total power to 1.0.
        """
        self.A /= np.sqrt(self.TotalPower())

    def Projection(self, axis='x'):
        """
        Create an intensity profile along the given axis for the given polarization direction.
        The absolute value of the intensity is summed along the other axis.
        return position, intensity
        """
        A = np.abs(ReorderBeamMatrix(self.A))
        if axis == 'x':
            pos = np.arange(0.0,self.nx*self.dx,self.dx) - (self.nx//2 * self.dx)
            val = np.sum(A, axis=1)
        elif axis == 'y':
            pos = np.arange(0.0,self.ny*self.dy,self.dy) - (self.ny//2 * self.dy)
            val = np.sum(A, axis=0)
        else:
            pos = np.arange(0.0,self.nx*self.dx,self.dx) - (self.nx//2 * self.dx)
            val = np.zeros(A.shape[0])
        return pos, val

    def FitSizeW(self, order=4, threshold=-3):
        """
        Fit a gaussian profile to the beam and return its size.
        @return wx(horizontal pol.), wy(horizontal pol.), wx(vertical pol.), wy(vertical pol.)
        
        Fitting the logarithmic intensity can be performed to higher oder (default=4)
        to reduce the influence of non-gaussian fractions of the beam to the evaluated second-oder coefficient.
        All intensity below a certain threshold (default=exp(-3)=0.05) relative to the peak amplitude
        will be ignored in te fit.
        """
        # to avoid zeros we add a small number (e^-20.7)
        ampl = np.log(np.abs(ReorderBeamMatrix(self.A)+1e-9).reshape(self.nx*self.ny))
        # normalize the intensity
        ampl = ampl - np.max(ampl)
        # compute the weights
        weights = np.ones_like(ampl)
        weights[ampl<threshold]=0.0
        # fit a polynomial
        ix1 = np.arange(self.nx)-(self.nx//2)
        iy1 = np.arange(self.ny)-(self.ny//2)
        x, y = np.meshgrid(ix1, iy1)
        X = np.array([x,y]).transpose().reshape(self.nx*self.ny,2)
        poly = PolynomialFeatures(degree=order)
        X_ = poly.fit_transform(X)
        clf = linear_model.LinearRegression()
        clf.fit(X_, ampl, sample_weight=weights)
        wx = 1.0/sqrt(-clf.coef_[3]) * self.dx if clf.coef_[3]<=0 else 0.0
        wy = 1.0/sqrt(-clf.coef_[5]) * self.dy if clf.coef_[5]<=0 else 0.0
        return wx, wy
    
    def RMS_SizeW(self):
        """
        Compute the horizontal and vertical beam size w.
        The size is defined as 2 times the rms-radius of the power distribution.
        """
        x = np.array([self.x(ix) for ix in np.arange(self.nx)])
        p = np.sum(np.square(np.abs(self.A)), axis=1)
        total = np.sum(p)
        sp = np.dot(x,p)
        xav = sp/total
        sqs = np.dot(np.square((x-xav)),p)
        xrms = np.sqrt(sqs/total)
        y = np.array([self.y(iy) for iy in np.arange(self.ny)])
        p = np.sum(np.square(np.abs(self.A)), axis=0)
        total = np.sum(p)
        sp = np.dot(y,p)
        yav = sp/total
        sqs = np.dot(np.square((y-yav)),p)
        yrms = np.sqrt(sqs/total)
        return (2.0*xrms, 2.0*yrms)
        

    def plot(self):
        """
        Create amplitude and phase plots
        """
        fig1 = plt.figure(1,figsize=(11,5))
        # determine the axis ranges
        xticks = np.arange(-self.nx//2, self.nx//2) * self.dx
        yticks = np.arange(-self.ny//2, self.ny//2) * self.dy
        amp = np.abs(self.A)
        phase = np.angle(self.A)
        maxA = np.max(amp)

        ax1 = fig1.add_subplot(121)
        M = ReorderBeamMatrix(amp)
        # first index is supposed to be x - thats why we have to transpose the matrix
        # vertical axis plots from top down - we have to flip it
        im = plt.imshow(np.flipud(M.T), interpolation='nearest',
            aspect=1.0, cmap='CMRmap', vmin=0.0, vmax=maxA,
            extent=[xticks[0], xticks[-1], yticks[0], yticks[-1]])
        cb = plt.colorbar()
        plt.xlabel("x / m")
        plt.ylabel("y / m")

        ax2 = fig1.add_subplot(122)
        M = ReorderBeamMatrix(phase)
        # first index is supposed to be x - thats why we have to transpose the matrix
        # vertical axis plots from top down - we have to flip it
        im = plt.imshow(np.flipud(M.T), interpolation='nearest',
            aspect=1.0, cmap='seismic', vmin=-pi, vmax=pi,
            extent=[xticks[0], xticks[-1], yticks[0], yticks[-1]])
        cb = plt.colorbar()
        plt.xlabel("x / m")
        plt.ylabel("y / m")

        return plt
