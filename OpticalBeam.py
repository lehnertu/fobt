#!/usr/bin/env python3
# coding=UTF-8

import sys, time
from math import *
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
    def GaussianBeam(cls, f, nx, ny, dx, dy, sigx, sigy):
        """
        Create a gaussian beam with width parameters sigx/sigy
        in horizontal/vertical direction.
        """
        beam = cls(f, nx, ny, dx, dy)
        sx2 = pow(sigx,2)
        sy2 = pow(sigy,2)
        for ix in range(nx):
            for iy in range(ny):
                beam.A_xi[ix,iy] = exp(-pow(beam.xi(ix),2)/sx2 - pow(beam.eta(iy),2)/sy2)
                beam.A_eta[ix,iy] = 0.0
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
            aspect=1.0, cmap='seismic',
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
            aspect=1.0, cmap='seismic',
            extent=[xticks[0], xticks[-1], yticks[0], yticks[-1]])
        cb = plt.colorbar()
        plt.xlabel("x / m")
        plt.ylabel("y / m")

        return plt
