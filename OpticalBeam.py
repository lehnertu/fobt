#!/usr/bin/env python3
# coding=UTF-8

import sys, time
from math import *
import numpy as np
import h5py
import matplotlib.pyplot as plt

# magnetic field constant in N/A²
mu0 = 4*np.pi*1e-7

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
        self.A = np.zeros((nx,ny),dtype=np.cdouble)
    
    @classmethod
    def GaussianBeam(cls, f, nx, ny, dx, dy, sigx, sigy):
        """
        Create a gaussian beam with width parameters sigx/sigy
        in horizontal/vertical direction.
        """
        beam = cls(f, nx, ny, dx, dy)
        sx2 = pow(sigx,2)
        sy2 = pow(sigy,2)
        # beam.A = np.fromfunction(lambda ix,iy: exp(-pow(beam.xi(ix),2)/sx2 - pow(beam.eta(iy),2)/sy2),
        #     (nx,ny), dtype=np.cdouble)
        for ix in range(nx):
            for iy in range(ny):
                beam.A[ix,iy] = exp(-pow(beam.xi(ix),2)/sx2 - pow(beam.eta(iy),2)/sy2)
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

    def plot_Intensity(self):
        amp = np.abs(self.A)
        # re-order the intensity matrix for display
        ll = amp[self.nx//2:self.nx, 0:self.ny//2]
        lr = amp[0:self.nx//2, 0:self.ny//2]
        ul = amp[self.nx//2:self.nx, self.ny//2:self.ny]
        ur = amp[0:self.nx//2, self.ny//2:self.ny]
        l = np.concatenate((ll,lr), axis=0)
        u = np.concatenate((ul,ur), axis=0)
        M = np.concatenate((u,l), axis=1)
        # create plot
        # fig = plt.figure(figsize=(6,6),dpi=150)
        fig = plt.figure()
        # determine the axis ranges
        xticks = np.arange(-self.nx//2, self.nx//2) * self.dx
        yticks = np.arange(-self.ny//2, self.ny//2) * self.dy
        # first index is supposed to be x - thats why we have to transpose the matrix
        # vertical axis plots from top down - we have to flip it
        im = plt.imshow(np.flipud(M.T), interpolation='nearest',
            aspect=1.0,
            extent=[xticks[0], xticks[-1], yticks[0], yticks[-1]])
        plt.xlabel("x / m")
        plt.ylabel("y / m")
        return plt
