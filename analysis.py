#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:02:54 2019

@author: severi
"""

from simulator import *
import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import copy
import time
import math

"""This file contains all the actual simulation runs and plotting commands. """

"""NOTE: Running the optimization sequences can take some time, so if you want to do it faster, change the numOfTries-setting to something smaller for
each sequence length."""

def gen_sequences(N):
    """Generates all non-symmetric sequences of length N having amino acids of type A and B. The idea for the code comes from article
    "Toy model for protein folding", 1993. """
    seqs = []
    #First assume that N is even. Enumerate all possible sequences that start from the center and continue to one of the directions
    halfseqs = list(itertools.product([-1,1],repeat=N//2))
    #Add the centrosymmetric possibilities to seqs
    for i in range(2**(N//2)):
        seqs += [list(reversed(halfseqs[i])) + list(halfseqs[i])]
    #Non-centrosymmetric sequences
    for (a,b) in itertools.combinations(halfseqs,2):
        seqs += [list(reversed(a)) + list(b)]
    #If N not even, twice as many sequences, with 1 or -1 at the center
    if N%2 == 1:
        L = len(seqs)
        seqs = seqs + copy.deepcopy(seqs)
        for i in range(L):
            seqs[i].insert(N//2,-1)
            seqs[i+L].insert(N//2,1)
    return seqs

def round_to_n(x,n=2):
    """In case we want to round to significant digits. We don't, but leaving it here anyways."""
    return round(x, -int(math.floor(math.log10(abs(x)))) + n-1)

def propOfSuccess(optEnergy,energies, accuracy=0.005):
    """Calculates the proportion of successful optimizations which reached the optimal energy optEnergy
    within deviation of amount "accuracy". energies is a vector that holds the energies from the different
    optimization runs."""
    numOfTries = len(energies)
    return sum([e > optEnergy-accuracy and e < optEnergy+accuracy for e in energies])/numOfTries

acDict = {-1:'B', 1:'A'}

#%%
"""Sequence length 3"""
"""In all of these, the objective is to run the optimizations a number of times and record the best solutions
to all sequences. The energies of all the solutions are also recorded."""
report_annealing = False

seqs3 = gen_sequences(3)

numOfTries = 100
temp1,temp2 = annealing(seqs3[0],10,0.1,np.pi/20,False), wrap_basinhopper(seqs3[0],10,0.5)
topAnnealingRes3, topScipyRes3 = [temp1 for i in range(len(seqs3))], [temp2 for i in range(len(seqs3))]
annealingEnergies3, scipyEnergies3 = np.zeros((len(seqs3),numOfTries)), np.zeros((len(seqs3),numOfTries))

for i in range(numOfTries):
    for j,seq in enumerate(seqs3):
        t1 = time.time()
        annealingRes = annealing(seq,4000,0.1,0,np.pi/20,report_annealing)
        t2 = time.time()
        scipyRes = wrap_basinhopper(seq,900,1,guessScale=2.5)
        t3 = time.time()
        
        annealingEnergies3[j,i], scipyEnergies3[j,i] = annealingRes[3], scipyRes.fun
        if annealingRes[3] < topAnnealingRes3[j][3]:
            topAnnealingRes3[j] = annealingRes
        if scipyRes.fun < topScipyRes3[j].fun:
            topScipyRes3[j] = scipyRes
        
        if round(annealingRes[3],2) != round(scipyRes.fun,2):
            print("Annealing: %f, Scipy: %f Times (annealing,scipy): %f, %f" % (round(annealingRes[3],2),round(scipyRes.fun,2),t2-t1,t3-t2))
        else:
            print("Both: %f. Times (annealing,scipy): %f, %f" % (round(annealingRes[3],2),t2-t1,t3-t2))
    print("Run %d finished" % (i+1))
#%%
plt.close("all")
fig = plt.figure()
for (i,res) in enumerate(topAnnealingRes3):
    r = res[0]
    ax = fig.add_subplot(2,3,i+1, projection='3d')
    for j in range(len(r)-1):
        ax.plot([r[j,0], r[j+1,0]], [r[j,1],r[j+1,1]],zs=[r[j,2],r[j+1,2]])
    ax.set_xlim([-len(r)/4,len(r)/4])
    ax.set_ylim([-len(r)/4,len(r)/4])
    ax.set_zlim([0,len(r)/2])
    ax.set_title("".join([acDict[q] for q in seqs3[i]]))
    ax.view_init(elev=34.)
fig.tight_layout()

#Proportions of optimizations that reached approximately the best one found
optEnergies3 = [min(min(annealingEnergies3[i,:]),min(scipyEnergies3[i,:])) for i in range(len(seqs3))]
propAnnealing3 = [propOfSuccess(optEnergies3[i],annealingEnergies3[i,:]) for i in range(len(seqs3))]
propScipy3 = [propOfSuccess(optEnergies3[i],scipyEnergies3[i,:]) for i in range(len(seqs3))]
#%%
"""Sequence length 4"""
report_annealing = False

seqs4 = gen_sequences(4)

numOfTries = 50
temp1,temp2 = annealing(seqs4[0],10,0.1,np.pi/20,False), wrap_basinhopper(seqs4[0],10,0.5)
topAnnealingRes4, topScipyRes4 = [temp1 for i in range(len(seqs4))], [temp2 for i in range(len(seqs4))]
annealingEnergies4, scipyEnergies4 = np.zeros((len(seqs4),numOfTries)), np.zeros((len(seqs4),numOfTries))

for i in range(numOfTries):
    for j,seq in enumerate(seqs4):
        t1 = time.time()
        annealingRes = annealing(seq,4000,0.1,0,np.pi/20,report_annealing)
        t2 = time.time()
        scipyRes = wrap_basinhopper(seq,500,1,guessScale=2.5)
        t3 = time.time()
        
        annealingEnergies4[j,i], scipyEnergies4[j,i] = annealingRes[3], scipyRes.fun
        if annealingRes[3] < topAnnealingRes4[j][3]:
            topAnnealingRes4[j] = annealingRes
        if scipyRes.fun < topScipyRes4[j].fun:
            topScipyRes4[j] = scipyRes
        
        if round(annealingRes[3],2) != round(scipyRes.fun,2):
            print("Annealing: %f, Scipy: %f Times (annealing,scipy): %f, %f" % (round(annealingRes[3],2),round(scipyRes.fun,2),t2-t1,t3-t2))
        else:
            print("Both: %f. Times (annealing,scipy): %f, %f" % (round(annealingRes[3],2),t2-t1,t3-t2))
    print("Run %d finished" % (i+1))

#%%
plt.close("all")
fig = plt.figure()
for (i,res) in enumerate(topAnnealingRes4):
    r = res[0]
    ax = fig.add_subplot(3,4,i+1, projection='3d')
    for j in range(len(r)-1):
        ax.plot([r[j,0], r[j+1,0]], [r[j,1],r[j+1,1]],zs=[r[j,2],r[j+1,2]])
    ax.set_xlim([-len(r)/4,len(r)/4])
    ax.set_ylim([-len(r)/4,len(r)/4])
    ax.set_zlim([0,len(r)/2])
    ax.set_title("".join([acDict[q] for q in seqs4[i]]))
    ax.view_init(elev=34.)
fig.tight_layout()

optEnergies4 = [min(min(annealingEnergies4[i,:]),min(scipyEnergies4[i,:])) for i in range(len(seqs4))]
propAnnealing4 = [propOfSuccess(optEnergies4[i],annealingEnergies4[i,:]) for i in range(len(seqs4))]
propScipy4 = [propOfSuccess(optEnergies4[i],scipyEnergies4[i,:]) for i in range(len(seqs4))]
#%%
"""Sequence length 5"""
report_annealing = False

seqs5 = gen_sequences(5)

numOfTries = 20
temp1,temp2 = annealing(seqs5[0],10,0.1,np.pi/20,False), wrap_basinhopper(seqs5[0],10,0.5)
topAnnealingRes5, topScipyRes5 = [temp1 for i in range(len(seqs5))], [temp2 for i in range(len(seqs5))]
annealingEnergies5, scipyEnergies5 = np.zeros((len(seqs5),numOfTries)), np.zeros((len(seqs5),numOfTries))

for i in range(numOfTries):
    for j,seq in enumerate(seqs5):
        t1 = time.time()
        annealingRes = annealing(seq,4000,0.1,0,np.pi/20,report_annealing)
        t2 = time.time()
        scipyRes = wrap_basinhopper(seq,400,1,guessScale=2.5)
        t3 = time.time()
        
        annealingEnergies5[j,i], scipyEnergies5[j,i] = annealingRes[3], scipyRes.fun
        if annealingRes[3] < topAnnealingRes5[j][3]:
            topAnnealingRes5[j] = annealingRes
        if scipyRes.fun < topScipyRes5[j].fun:
            topScipyRes5[j] = scipyRes
        
        if round(annealingRes[3],2) != round(scipyRes.fun,2):
            print("Annealing: %f, Scipy: %f Times (annealing,scipy): %f, %f" % (round(annealingRes[3],2),round(scipyRes.fun,2),t2-t1,t3-t2))
        else:
            print("Both: %f. Times (annealing,scipy): %f, %f" % (round(annealingRes[3],2),t2-t1,t3-t2))
    print("Run %d finished" % (i+1))

#%%
plt.close("all")
fig = plt.figure()
for (i,res) in enumerate(topAnnealingRes5):
    r = res[0]
    ax = fig.add_subplot(4,5,i+1, projection='3d')
    for j in range(len(r)-1):
        ax.plot([r[j,0], r[j+1,0]], [r[j,1],r[j+1,1]],zs=[r[j,2],r[j+1,2]])
    ax.set_xlim([-len(r)/4,len(r)/4])
    ax.set_ylim([-len(r)/4,len(r)/4])
    ax.set_zlim([0,len(r)/2])
    ax.set_title("".join([acDict[q] for q in seqs5[i]]))
    ax.view_init(elev=34.)
fig.tight_layout()

optEnergies5 = [min(min(annealingEnergies5[i,:]),min(scipyEnergies5[i,:])) for i in range(len(seqs5))]
propAnnealing5 = [propOfSuccess(optEnergies5[i],annealingEnergies5[i,:]) for i in range(len(seqs5))]
propScipy5 = [propOfSuccess(optEnergies5[i],scipyEnergies5[i,:]) for i in range(len(seqs5))]
#%%
"""One sequence of length 10"""
report_annealing = True

seq10 = [1,1,1,-1,-1,-1,-1,1,1,1]

numOfTries = 300
topAnnealingRes10 = annealing(seq10,10,0.1,np.pi/20,False)
annealingEnergies10 = np.zeros(numOfTries)

for i in range(numOfTries):
    annealingRes = annealing(seq10,8000,0.5,0,np.pi/20,report_annealing)#The annealing time and starting temperature are different here than in the others
    
    annealingEnergies10[i] = annealingRes[3]
    if annealingRes[3] < topAnnealingRes10[3]:
        topAnnealingRes10 = annealingRes
    
    print("Round %d: Energy: %f" % (i+1,annealingRes[3]))
#%%
plt.close("all")
fig = plt.figure()
r = topAnnealingRes10[0]
ax = fig.add_subplot(1,1,1,projection='3d')
for j in range(len(r)-1):
    ax.plot([r[j,0], r[j+1,0]], [r[j,1],r[j+1,1]],zs=[r[j,2],r[j+1,2]],LineWidth=2)
ax.set_xlim([-2.5,1])
ax.set_ylim([-2.5,1])
ax.set_zlim([-1,2.5])
ax.set_title("".join([acDict[q] for q in seq10]))
fig.tight_layout()

propAnnealing10 = propOfSuccess(min(annealingEnergies10),annealingEnergies10,accuracy=0.01)

plt.figure()
plt.hist(annealingEnergies10,40)
plt.xlabel("Optimized energy")
#%%
"""The fancy 3d autostereogram plot"""
plt.close("all")
fig = plt.figure()
r = topAnnealingRes10[0]
elevs = [10,10,10,10,45,45]
azimuths = [-40, -41, 20, 21, 80, 81]
for i in range(6):
    ax = fig.add_subplot(3,2,i+1,projection='3d')
    ax.grid(False)
    for j in range(len(r)-1):
        ax.plot([r[j,0], r[j+1,0]], [r[j,1],r[j+1,1]],zs=[r[j,2],r[j+1,2]],LineWidth=2)
    ax.set_xlim([-2.5,1])
    ax.set_ylim([-2.5,1])
    ax.set_zlim([-1,2.5])
    ax.set_xticks([-1.,0.,1.,2.])
    plt.axis("off")
    ax.view_init(elev=elevs[i],azim=azimuths[i])