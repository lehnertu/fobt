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

class SingleFrequencyBeam():
    
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
        self.A_xi = np.zeros((nx,ny),dtype=np.cdouble)
        self.A_eta = np.zeros((nx,ny),dtype=np.cdouble)
    
    @classmethod
    def GaussianBeamWaist(cls, f, nx, ny, dx, dy, sigx, sigy):
        """
        Create a gaussian beam with width parameters sigx/sigy
        in horizontal/vertical direction.
        """
        beam = cls(f, nx, ny, dx, dy)
        sx2 = pow(sigx,2)
        sy2 = pow(sigy,2)
        for ix in range(nx):
            for iy in range(ny):
                a = exp(-pow(beam.xi(ix),2)/sx2 - pow(beam.eta(iy),2)/sy2)
                beam.A_xi[ix,iy] = a
                beam.A_eta[ix,iy] = a
        return beam

    @classmethod
    def GaussianBeam(cls, f, nx, ny, dx, dy, zR, z):
        """
        Create a gaussian beam with a Rayleigh range zR
        at a distance z from the waist (both polarization directions are equal).
        z cannot be exactly zero.
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
                r2 = pow(beam.xi(ix),2) + pow(beam.eta(iy),2)
                a = np.exp( -r2/pow(w,2) - 1.0j*(k*z-zeta) - 1.0j*k*r2/(2.0*R) )
                beam.A_xi[ix,iy] = a
                beam.A_eta[ix,iy] = a
        return beam

    @classmethod
    def NearFieldProp(cls, source, dist):
        """
        Transport a source beam over a certain distance
        using a near-field propagation method.
        The created beam has the same geometrical properties as the source beam.
        """
        λ = scipy.constants.c/source.freq
        k = 2*pi/λ
        nx = source.nx
        ny = source.ny
        dkx = 2.0*pi/source.dx/nx
        dky = 2.0*pi/source.dy/ny
        # create the new beam
        beam = cls(source.freq, nx, ny, source.dx, source.dy)
        # Fourier transform into momentum space
        FF = np.fft.fft2(source.A_xi)
        # apply the phase factors according to the propagation length
        FFD = np.zeros_like(FF)
        for ix in range(nx):
            for iy in range(ny):
                if ix<nx//2:
                    if iy<ny//2:
                        FFD[ix,iy] = FF[ix,iy] * ( np.exp(-1.0j*k*dist) *
                            np.exp(1.0j*(pow(ix*dkx,2)+pow(iy*dky,2))*dist/(2.0*k)) )
                    else:
                        FFD[ix,iy] = FF[ix,iy] * ( np.exp(-1.0j*k*dist) *
                            np.exp(1.0j*(pow(ix*dkx,2)+pow((iy-ny)*dky,2))*dist/(2.0*k)) )
                else:
                    if iy<ny//2:
                        FFD[ix,iy] = FF[ix,iy] * ( np.exp(-1.0j*k*dist) *
                            np.exp(1.0j*(pow((ix-nx)*dkx,2)+pow(iy*dky,2))*dist/(2.0*k)) )
                    else:
                        FFD[ix,iy] = FF[ix,iy] * ( np.exp(-1.0j*k*dist) *
                            np.exp(1.0j*(pow((ix-nx)*dkx,2)+pow((iy-ny)*dky,2))*dist/(2.0*k)) )
        # back-transform into position space
        FFB = np.fft.ifft2(FFD)
        # set the output field
        beam.A_xi = nx*nx*FFB
        # Fourier transform into momentum space
        FF = np.fft.fft2(source.A_eta)
        # apply the phase factors according to the propagation length
        FFD = np.zeros_like(FF)
        for ix in range(nx):
            for iy in range(ny):
                if ix<nx//2:
                    if iy<ny//2:
                        FFD[ix,iy] = FF[ix,iy] * ( np.exp(-1.0j*k*dist) *
                            np.exp(1.0j*(pow(ix*dkx,2)+pow(iy*dky,2))*dist/(2.0*k)) )
                    else:
                        FFD[ix,iy] = FF[ix,iy] * ( np.exp(-1.0j*k*dist) *
                            np.exp(1.0j*(pow(ix*dkx,2)+pow((iy-ny)*dky,2))*dist/(2.0*k)) )
                else:
                    if iy<ny//2:
                        FFD[ix,iy] = FF[ix,iy] * ( np.exp(-1.0j*k*dist) *
                            np.exp(1.0j*(pow((ix-nx)*dkx,2)+pow(iy*dky,2))*dist/(2.0*k)) )
                    else:
                        FFD[ix,iy] = FF[ix,iy] * ( np.exp(-1.0j*k*dist) *
                            np.exp(1.0j*(pow((ix-nx)*dkx,2)+pow((iy-ny)*dky,2))*dist/(2.0*k)) )
        # back-transform into position space
        FFB = np.fft.ifft2(FFD)
        # set the output field
        beam.A_eta = nx*nx*FFB
        return beam
    
    def xi(self, ix):
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
    
    def eta(self, iy):
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

    def Projection(self, polarization='xi', axis='x'):
        """
        Create an intensity profile along the given axis for the given polarization direction.
        The absolute value of the intensity is summed along the other axis.
        return position, intensity
        """
        if polarization=='xi': A = np.abs(ReorderBeamMatrix(self.A_xi))
        elif polarization=='eta': A = np.abs(ReorderBeamMatrix(self.A_eta))
        else: A = np.zeros_like(self.A_xi)
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
        ampl = np.log(np.abs(ReorderBeamMatrix(self.A_xi)+1e-9).reshape(self.nx*self.ny))
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
        wx_xi = 1.0/sqrt(-clf.coef_[3]) * self.dx if clf.coef_[3]<=0 else 0.0
        wy_xi = 1.0/sqrt(-clf.coef_[5]) * self.dy if clf.coef_[5]<=0 else 0.0
        ampl = np.log(np.abs(ReorderBeamMatrix(self.A_eta)+1e-9).reshape(self.nx*self.ny))
        ampl = ampl - np.max(ampl)
        weights = np.ones_like(ampl)
        weights[ampl<threshold]=0.0
        ix1 = np.arange(self.nx)-(self.nx//2)
        iy1 = np.arange(self.ny)-(self.ny//2)
        x, y = np.meshgrid(ix1, iy1)
        X = np.array([x,y]).transpose().reshape(self.nx*self.ny,2)
        poly = PolynomialFeatures(degree=order)
        X_ = poly.fit_transform(X)
        clf = linear_model.LinearRegression()
        clf.fit(X_, ampl, sample_weight=weights)
        wx_eta = 1.0/sqrt(-clf.coef_[3]) * self.dx if clf.coef_[3]<=0 else 0.0
        wy_eta = 1.0/sqrt(-clf.coef_[5]) * self.dy if clf.coef_[5]<=0 else 0.0
        return wx_xi, wy_xi, wx_eta, wy_eta

    def plot(self):
        """
        Create amplitude and phase plots for both polarization directions
        """
        fig1 = plt.figure(1,figsize=(11,9))
        # determine the axis ranges
        xticks = np.arange(-self.nx//2, self.nx//2) * self.dx
        yticks = np.arange(-self.ny//2, self.ny//2) * self.dy
        amp_xi = np.abs(self.A_xi)
        phase_xi = np.angle(self.A_xi)
        amp_eta = np.abs(self.A_eta)
        phase_eta = np.angle(self.A_eta)
        maxX = np.max(amp_xi)
        maxY = np.max(amp_eta)
        maxV = np.max([maxX,maxY])

        ax1 = fig1.add_subplot(221)
        M = ReorderBeamMatrix(amp_xi)
        # first index is supposed to be x - thats why we have to transpose the matrix
        # vertical axis plots from top down - we have to flip it
        im = plt.imshow(np.flipud(M.T), interpolation='nearest',
            aspect=1.0, cmap='CMRmap', vmin=0.0, vmax=maxV,
            extent=[xticks[0], xticks[-1], yticks[0], yticks[-1]])
        cb = plt.colorbar()
        plt.xlabel("x / m")
        plt.ylabel("y / m")

        ax2 = fig1.add_subplot(222)
        M = ReorderBeamMatrix(phase_xi)
        # first index is supposed to be x - thats why we have to transpose the matrix
        # vertical axis plots from top down - we have to flip it
        im = plt.imshow(np.flipud(M.T), interpolation='nearest',
            aspect=1.0, cmap='seismic', vmin=-pi, vmax=pi,
            extent=[xticks[0], xticks[-1], yticks[0], yticks[-1]])
        cb = plt.colorbar()
        plt.xlabel("x / m")
        plt.ylabel("y / m")

        ax3 = fig1.add_subplot(223)
        M = ReorderBeamMatrix(amp_eta)
        # first index is supposed to be x - thats why we have to transpose the matrix
        # vertical axis plots from top down - we have to flip it
        im = plt.imshow(np.flipud(M.T), interpolation='nearest',
            aspect=1.0, cmap='CMRmap', vmin=0.0, vmax=maxV,
            extent=[xticks[0], xticks[-1], yticks[0], yticks[-1]])
        cb = plt.colorbar()
        plt.xlabel("x / m")
        plt.ylabel("y / m")

        ax4 = fig1.add_subplot(224)
        M = ReorderBeamMatrix(phase_eta)
        # first index is supposed to be x - thats why we have to transpose the matrix
        # vertical axis plots from top down - we have to flip it
        im = plt.imshow(np.flipud(M.T), interpolation='nearest',
            aspect=1.0, cmap='seismic', vmin=-pi, vmax=pi,
            extent=[xticks[0], xticks[-1], yticks[0], yticks[-1]])
        cb = plt.colorbar()
        plt.xlabel("x / m")
        plt.ylabel("y / m")

        return plt
