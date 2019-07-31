#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:43:16 2019

@author: severi
"""
import numpy as np
import scipy.optimize
import scipy
import time
from random import random as rand

def LJ_energy(r,s):
    """Calculates the potential energy of the Lennard Jones-bonds. r is a N x 3 matrix containing the 
    positions of the amino acids and s is a N-vector of 1:s and -1:s, depending on the 
    species of the amino acid. """
    energy = 0
    dists = scipy.spatial.distance.cdist(r,r,metric="euclidean")
    for i in range(len(s)):
        for j in range(i+2,len(s)):
            if s[i] != s[j]:#AB
                energy += 4*(dists[i,j]**(-12)+0.5*dists[i,j]**(-6))
            elif s[i] == 1:#AA
                energy += 4*(dists[i,j]**(-12)-dists[i,j]**(-6))
            elif s[i] == -1:#BB
                energy += 4*(dists[i,j]**(-12)-0.5*dists[i,j]**(-6))
    return energy

def bend_energy(v):
    """Calculates the potential energy related to the bends in the amino acid chain. v is the (N-1,3) -matrix
    that contains the relative distances between amino acids in the chain"""
    dots = np.sum(v[0:-1,:]*v[1:,:],axis=1)
    energy = sum(0.25*(1-dots))
    return energy

def rot_matrix_around(e,t):
    """Returns the rotation matrix that performs a rotation by t radians around the Euler axis specified
    by the unit vector e."""
    return np.eye(3)*np.cos(t) + (1-np.cos(t))*np.outer(e,e) + np.array([[0,-e[2],e[1]],[e[2],0,-e[0]],[-e[1],e[0],0]])*np.sin(t)

def r_from(v):
    """Returns the absolute positions of the amino acids in the chain based on the relative distances v 
    ((N-1 x 3)-matrix)."""
    r = np.zeros((len(v)+1,3))
    for i in range(1,len(r)):
        r[i,:] = r[i-1,:] + v[i-1,:]
    return r

def init_v(N):
    """Initializes the relative distance vectors of the N amino acids to unit vectors pointing approximately,
    but not directly, towards the z-direction. v is a (N-1 x 3) -matrix. """
    v = np.zeros((N-1,3))
    for i in range(N-1):
        new = np.array([(rand()-0.5)/10,(rand()-0.5)/10,1])
        v[i,:] = new/np.linalg.norm(new)
    return v

def v_from(theta,phi):
    """Calculates the relative distance vectors from the vectors containing the two angles at each bend. 
    theta is a (N-2) -vector while phi is a (N-3) -vector. v is created consistently using a convention where
    the theta bends are done around the vector denoted by the cross product of the previous two vectors and
    the phi bends around the previous vector. """
    v = np.zeros((len(theta)+1,3))
    v[0,:] = np.array([0,0,1])
    v[1,:] = rot_matrix_around(np.array([1,0,0]),theta[0]).dot(v[0,:])
    for (i,(t,p)) in enumerate(zip(theta[1:],phi)):
        thetadir = np.cross(v[i,:],v[i+1,:])#Let's hope this isn't zero
        thetadir = thetadir/(np.linalg.norm(thetadir))
        v[i+2,:] = rot_matrix_around(thetadir,t).dot(v[i+1,:])
        v[i+2,:] = rot_matrix_around(v[i+1,:],p).dot(v[i+2,:])
    return v

def energy(par,seq):
    """Calculates the energy related to the amino acid sequence seq with the rotations theta and phi 
    ((N-2) and (N-3) -vectors)."""
    N = len(seq)
    v = v_from(par[:N-2],par[N-2:])
    r = r_from(v)
    return LJ_energy(r,seq) + bend_energy(v)

def metropolis(seq,T,times,delta):
    """Runs the basic metropolis algorithm for the sequence denoted by seq at the temperature T.
    The algorithm is run "times" times for all angle parameters. delta is the maximum amount that the 
    angles are deviated at each step. """
    N = len(seq)
    pars = np.zeros(2*N-5)
    pars[:N-2] = np.array([rand()/10 for i in range(N-2)])
    pars[N-2:] = np.array([(rand()-0.5)/10 for i in range(N-3)])
    v = v_from(pars[:N-2],pars[N-2:])
    r = r_from(v)
    E = LJ_energy(r,seq) + bend_energy(v)
    for i in range(times):
        for j in range(2*N-5):
            #This part explained in the annealing function
            oldpar = pars[j]
            if j < N-2:#Theta-bend
                #pars[j] = (pars[j] + 2*delta*rand()-delta)%np.pi#For the correct periodic boundary conditions
                pars[j] = (pars[j] + np.random.normal(scale=delta))%np.pi
                thetadir = np.cross(v[j,:],v[j+1,:])#Let's again hope this isn't zero
                thetadir = thetadir/(np.linalg.norm(thetadir))
                vnew = v.copy()
                vnew[j+1:,:] = v[j+1:,:].dot(rot_matrix_around(thetadir,pars[j]-oldpar).transpose())
            else:#Phi-bend
                #pars[j] = ((pars[j] + (rand()-0.5)/10) + np.pi) % (2*np.pi) - np.pi
                pars[j] = (pars[j] + np.random.normal(scale=delta) + np.pi) % (2*np.pi) - np.pi
                vnew = v.copy()
                vnew[j-(N-2)+2:,:] = v[j-(N-2)+2:,:].dot(rot_matrix_around(v[j-(N-2)+1,:],pars[j]-oldpar).transpose())
            rnew = r_from(vnew)
            Enew = LJ_energy(rnew,seq) + bend_energy(vnew)
            if rand() < np.exp(-(Enew-E)/T):
                E = Enew
                v = vnew.copy()
            else:
                pars[j] = oldpar
    return (r_from(v),v,pars,E)

def tempRampExp(T0,maxi,i):
    rampSpeed = 3
    return (T0/(1-np.exp(-rampSpeed)))*np.exp(-rampSpeed/maxi*i)+T0-T0/(1-np.exp(-rampSpeed))

def tempRampLin(T0,maxi,i):
    return T0 - T0*i/maxi

def annealing(seq,times,startT,endT,delta,reportProgress=False,tempRamp=tempRampLin):
    """Tries to find the minimum enery configuration for the amino acid sequence seq by using the 
    metropolis algorithm for simulated annealing."""
    (r,v,pars,E) = metropolis(seq,startT,200,delta)#Thermalization
    N = len(seq)
    accepted = 0
    for i in range(times):
        T = tempRamp(startT,times,i)#startT + (i/times)*(endT-startT)
        for j in range(2*N-5):
            #First we try changing the parameter to a new value, and then revert if it wasn't accepted
            oldpar = pars[j]
            #The v_from -function is not used here, since only one bend is changed at a time. It's faster to rotate all following v:s the same amount. 
            if j < N-2:#Theta-bend
                #pars[j] = (pars[j] + 2*delta*rand()-delta)%np.pi#For the correct periodic boundary conditions
                pars[j] = (pars[j] + np.random.normal(scale=delta))%np.pi#For the correct periodic boundary conditions
                thetadir = np.cross(v[j,:],v[j+1,:])#Let's again hope this isn't zero
                thetadir = thetadir/(np.linalg.norm(thetadir))
                vnew = v.copy()
                vnew[j+1:,:] = v[j+1:,:].dot(rot_matrix_around(thetadir,pars[j]-oldpar).transpose())
            else:#Phi-bend
                #pars[j] = ((pars[j] +  2*delta*rand()-delta) + np.pi) % (2*np.pi) - np.pi
                pars[j] = (pars[j] + np.random.normal(scale=delta) + np.pi) % (2*np.pi) - np.pi#For the correct periodic boundary conditions
                vnew = v.copy()
                vnew[j-(N-2)+2:,:] = v[j-(N-2)+2:,:].dot(rot_matrix_around(v[j-(N-2)+1,:],pars[j]-oldpar).transpose())
            rnew = r_from(vnew)
            Enew = LJ_energy(rnew,seq) + bend_energy(vnew)
            if rand() < np.exp(-(Enew-E)/T):
                E = Enew
                v = vnew.copy()
                accepted += 1
            else:
                pars[j] = oldpar
        if i%(int(times/10)) == 0 and i != 0 and reportProgress:
            print("%f percent calculated. Temperature: %f. Acceptance: %f" % (100*i/times,T,accepted/((times/10)*(2*N-5))))
            accepted = 0
    return (r_from(v),v,pars,E)

def wrap_basinhopper(seq,niter,T,guessScale=2.5,thermalizationDelta = np.pi/20):
    """Wrapper for scipy.optimize.basinhopper. Gives the bounds and starting parameters automatically."""
    N = len(seq)
    _,_,pars,_ = metropolis(seq,T,200,delta=thermalizationDelta)
    bounds = [(0+1e-12,np.pi-1e-12)]*(N-2) + [(-np.pi,np.pi)]*(N-3)#Avoiding division by zero errors with the bounds for theta
    return scipy.optimize.basinhopping(func=energy,x0=pars,niter=niter,T=T,minimizer_kwargs={"args":seq,"bounds":bounds,"method":'L-BFGS-B'})