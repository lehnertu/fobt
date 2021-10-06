#!/usr/bin/env python3
# coding=UTF-8

import sys, time
from math import *
import scipy.constants
import numpy as np
import h5py
import matplotlib.pyplot as plt

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
        Create a gaussian beam with width Rayleigh range zR
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
    
    """
    NearFieldDiffractionPropagation[beam_, dist_] := 
      Module[{\[Lambda], k, nn, dx, dk, FF, ul, ur, ll, lr, nx, ny, FFD},

       \[Lambda] = beam[[4]];
       k = 2 \[Pi]/\[Lambda];
       nn = beam[[2]];
       dx = beam[[3]];
       dk = 2 \[Pi]/dx/nn;

       FF = Sqrt[nn]/(2 \[Pi])*Fourier[beam[[1]]]*dx;

       ll = Table[
         Exp[I k dist]*Exp[-I ((nx - 1)^2 + (ny - 1)^2)*dk^2*dist/2/k]*
          FF[[nx, ny]],
         {nx, 1, nn/2}, {ny, 1, nn/2}];
       ul = Table[
         Exp[I k dist]*
          Exp[-I ((nx - nn - 1)^2 + (ny - 1)^2)*dk^2*dist/2/k]*
          FF[[nx, ny]],
         {nx, nn/2 + 1, nn}, {ny, 1, nn/2}];
       lr = Table[
         Exp[I k dist]*
          Exp[-I ((nx - 1)^2 + (ny - nn - 1)^2)*dk^2*dist/2/k]*
          FF[[nx, ny]],
         {nx, 1, nn/2}, {ny, nn/2 + 1, nn}];
       ur = Table[
         Exp[I k dist]*
          Exp[-I ((nx - nn - 1)^2 + (ny - nn - 1)^2)*dk^2*dist/2/k]*
          FF[[nx, ny]],
         {nx, nn/2 + 1, nn}, {ny, nn/2 + 1, nn}];

       FFD = ArrayFlatten[{{ll, lr}, {ul, ur}}];

       {Sqrt[nn]*InverseFourier[FFD]*dk, nn, dx, \[Lambda]}

       ];
    """
    
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

    def plot(self):
        
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
