# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 23:49:53 2017


"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 21:31:09 2017

@author: CHart
"""
import scipy.cluster as spc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection



def DrawBoard2(mv,sd):
    fig = plt.figure()
    ax=fig.add_subplot(111)
    plt.xticks([x-0.5 for x in xrange(11)])
    plt.yticks([y - 0.5 for y in xrange(11)])
    # Remove whitespace from around the image
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.grid(which='major', axis='both', linestyle='-')
    # Add the image
    ax.imshow(mv,interpolation='nearest')
    L = len(mv)
    for i in xrange(L):
        for j in xrange(L):
            if  (np.abs(mv[i,j]) - sd[i,j])<=0:
               ax.add_patch(
                patches.Rectangle(
                    (j-0.5,i-0.5),   # (x,y)
                    1.0,          # width
                    1.0,          # height
                    facecolor = 'white'))
            elif mv[i,j] < 0:
                h = ax.text(j-0.2,i+0.2,'A')
                h.set_fontsize(20)
            elif mv[i,j] > 0:
                h = ax.text(j-0.2,i+0.2,'D')  
                h.set_fontsize(20)
    ax.set_xticklabels([]# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 23:49:53 2017

@author: e289660
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 21:31:09 2017

@author: CHart
"""
import scipy.cluster as spc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection



def DrawBoard2(next_state,last):
    fig = plt.figure()
    ax=fig.add_subplot(111)
    plt.xticks([x-0.5 for x in xrange(11)])
    plt.yticks([y - 0.5 for y in xrange(11)])
    # Remove whitespace from around the image
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.grid(which='major', axis='both', linestyle='-')
    # Add the image
    ax.imshow(next_state,interpolation='nearest')
    L = len(next_state)
    for i in xrange(L):
        for j in xrange(L):
            if  (np.abs(next_state[i,j]) - last[i,j])<=0:
               ax.add_patch(
                patches.Rectangle(
                    (j-0.5,i-0.5),   # (x,y)
                    1.0,          # width
                    1.0,          # height
                    facecolor = 'white'))
            elif next_state[i,j] < 0:
                h = ax.text(j-0.2,i+0.2,'A')
                h.set_fontsize(20)
            elif next_state[i,j] > 0:
                h = ax.text(j-0.2,i+0.2,'D')  
                h.set_fontsize(20)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return fig,ax


def BoardDistance(m1,m2):
    
    m2r = np.fliplr(m2)
    dists1 = [np.sum(np.abs(m1 - np.rot90(m2,k))) for k in xrange(4)]
    dists2 = [np.sum(np.abs(m1 - np.rot90(m2r,k))) for k in xrange(4)]
    dists = dists1 + dists2
    bestdist = np.min(dists)
    bestrot = dists.index(bestdist)
    return bestdist,bestrot


def FlipBoard(template,test):
    
    testr = np.fliplr(test)
    dists1 = [np.sum(np.abs(template - np.rot90(test,k))) for k in xrange(4)]
    dists2 = [np.sum(np.abs(template - np.rot90(testr,k))) for k in xrange(4)]
    dists = dists1 + dists2
    bestdist = np.min(dists)
    bestrot = dists.index(bestdist)
    if dists.index(bestdist) < 4:
        return np.rot90(test,dists.index(bestdist))
    else:
        return np.rot90(testr,dists.index(bestdist)-4)
    

def BaseMatrix():
    
    M = np.zeros((11,11),int)
    M[0,3:8] = -1
    M[1,5] = -1
    M[3,5] = 1
    M[4,4] = 1
    M[4,5] = 1
   #M= M + np.rot90(M) +  np.rot90( np.rot90(M)) + np.rot90(np.rot90( np.rot90(M))) 
    M2 = np.zeros((11,11),int)
    M2[5,5] = 2
    return M + np.rot90(M) +  np.rot90( np.rot90(M)) + np.rot90(np.rot90( np.rot90(M))) + M2


def CheckMove(initconf,move):
    board = np.matrix.copy(initconf)
    start = move.split('-')[0]  #(l.split(',')[0].split('/')[0].split('-')[0])
    
    tmp = initconf[ord(start[0]) - ord('a'),int(start[1:])-1] 
    ##
    
    finish = move.split('-')[1]
    x0 = ord(start[0]) - ord('a')
    y0 = int(start[1:])-1
    x1 = ord(finish[0]) - ord('a')
    y1 = int(finish[1:])-1
    xdiff = x1 - x0
    ydiff = y1 - y0    
    retval = True
    #check movement along y direction
    if xdiff == 0 and ydiff > 0:
        ysteps = [board[x0,y0 + ydelta]==0 for ydelta in xrange(1,ydiff + 1)]
        print ysteps
        if False in ysteps:
            retval = False
    if xdiff == 0 and ydiff < 0:
        ysteps = [board[xdiff,y0 - ydelta]==0 for ydelta in xrange(1,np.abs(ydiff) + 1)]
        if False in ysteps:
            retval = False
     #check movement along x directio
    if ydiff == 0 and xdiff > 0:
        ysteps = [board[x0 + xdelta,y0]==0 for xdelta in xrange(1,xdiff + 1)]
        if False in ysteps:
            retval = False
    if ydiff == 0 and xdiff < 0:
        ysteps = [board[x0 + xdelta,y0]==0 for xdelta in xrange(1,xdiff + 1)]
        if False in ysteps:
            retval = False
    return retval

def MakeMove(initconf,move):
    zz = np.matrix.copy(initconf)
    start = move.split('-')[0]  #(l.split(',')[0].split('/')[0].split('-')[0])
    
    tmp = initconf[ord(start[0]) - ord('a'),int(start[1:])-1] 
    zz[ord(start[0]) - ord('a'),int(start[1:])-1]  = 0
    finish = move.split('-')[1]
    #initconf[ord(start[0]) - ord('a'),int(start[1:])-1]  = 0
    zz[ord(finish[0]) - ord('a'),int(finish[1:])-1]  = tmp
    currx = ord(finish[0]) - ord('a')
    curry = int(finish[1:])-1
    #white captures
    #check if move
    
    #check if moved piece is sandwiching anyone

    #black captures
    
    #if int(finish[1:])-1 != int(start[1:])-1:        
    if ord(finish[0]) - ord('a') != ord(start[0]) - ord('a'):
        if curry < 9  and curry>=2: 
            
            if np.dot(zz[currx,curry:curry+3],np.array([-1,1,-1])) == 3:
                zz[currx,curry+1] = 0
           
            #if np.dot(zz[currx,curry:curry+3],np.array([-1,1,-1])) == 3:
            #    zz[currx,curry+1] = 0  
            if np.dot(zz[currx,curry-1:curry-4:-1],np.array([-1,1,-1])) == 3:
                zz[currx-1,curry] = 0
            #if np.dot(zz[currx-1:currx-4:-1,curry],np.array([-1,1,-1])) == 3:
            #    zz[currx-1,curry] = 0  
    print ord(finish[0]) - ord('a') , ord(start[0]) - ord('a')
    #if ord(finish[0]) - ord('a') != ord(start[0]) - ord('a'):
    if int(finish[1:])-1 != int(start[1:])-1:      
        
        if currx<9 and currx >= 2:
            
            if np.dot(zz[currx:currx+3,curry],np.array([-1,1,-1])) == 3:
                zz[currx+1,curry] = 0
            #if np.dot(zz[currx:currx+3,curry],np.array([-1,1,-1])) == 3:
            #    zz[currx+1,curry] = 0  
            
            if np.dot(zz[currx,xrange(curry-1,curry-4,-1)],np.array([-1,1,-1])) == 3:
               zz[currx-1,curry] = 0
                
            if np.dot(zz[currx:currx-3:-1,curry],np.array([-1,1,-1])) == 3:
                zz[currx-1,curry] = 0
    #white capture
    if ord(finish[0]) - ord('a') != ord(start[0]) - ord('a'):
        if curry < 9  and curry>=2: 
            
            if np.dot(zz[currx,curry:curry+3],np.array([1,-1,1])) == 3:
                zz[currx,curry+1] = 0
           
            #if np.dot(zz[currx,curry:curry+3],np.array([-1,1,-1])) == 3:
            #    zz[currx,curry+1] = 0  
            if np.dot(zz[currx,curry-1:curry-4:-1],np.array([1,-1,1])) == 3:
                zz[currx-1,curry] = 0
            #if np.dot(zz[currx-1:currx-4:-1,curry],np.array([-1,1,-1])) == 3:
            #    zz[currx-1,curry] = 0  
    
    if int(finish[1:])-1 != int(start[1:])-1:      
        
        if currx<9 and currx >= 2:
            print zz[currx:currx-3:-1,curry],zz[currx,xrange(curry-1,curry-4,-1)]
            print zz[currx:currx+3:,curry],zz[currx,xrange(curry,curry+3,-1)]
            if np.dot(zz[currx:currx+3,curry],np.array([1,-1,-1])) == 3:
                zz[currx+1,curry] = 0
            #if np.dot(zz[currx:currx+3,curry],np.array([-1,1,-1])) == 3:
            #    zz[currx+1,curry] = 0  
            
            if np.dot(zz[currx,xrange(curry-1,curry-4,-1)],np.array([1,-1,1])) == 3:
               zz[currx-1,curry] = 0
                
            if np.dot(zz[currx:currx-3:-1,curry],np.array([1,-1,1])) == 3:
                zz[currx-1,curry] = 0
    
    return zz
    

astercount = 0
move1 = []
move2 = []
move3 = []
move4 = []
whitewin = []
blackwin = []
draws = []
filelastir = 'C:\TEMP\e289660\Documents'
FH = open(filelastir + '\\' + 'hnefatafl_openings.txt', 'r')
#lines = [ i.decode('unicode_escape').encode('ascii','ignore').rstrip() for i in FH.readlines()]

seq = input('Input Moves ')

#for n,l in enumerate(lines):
    #print l

z = BaseMatrix() 
#z = MakeMove(z,firstfourmoves[0])
#        start = (l.split(',')[0].split('/')[0].split('-')[0])
#        tmp = z[ord(start[0]) - ord('a'),int(start[1:])-1] 
#        z[ord(start[0]) - ord('a'),int(start[1:])-1]  = 0
#        finish = (l.split(',')[0].split('/')[0].split('-')[1])
#        z[ord(finish[0]) - ord('a'),int(finish[1:])-1]  = tmp

legal = CheckMove(z,seq)

    ax.set_yticklabels([])
    return fig,ax


def BoardDistance(m1,m2):
    
    m2r = np.fliplr(m2)
    dists1 = [np.sum(np.abs(m1 - np.rot90(m2,k))) for k in xrange(4)]
    dists2 = [np.sum(np.abs(m1 - np.rot90(m2r,k))) for k in xrange(4)]
    dists = dists1 + dists2
    bestdist = np.min(dists)
    bestrot = dists.index(bestdist)
    return bestdist,bestrot


def FlipBoard(template,test):
    
    testr = np.fliplr(test)
    dists1 = [np.sum(np.abs(template - np.rot90(test,k))) for k in xrange(4)]
    dists2 = [np.sum(np.abs(template - np.rot90(testr,k))) for k in xrange(4)]
    dists = dists1 + dists2
    bestdist = np.min(dists)
    bestrot = dists.index(bestdist)
    if dists.index(bestdist) < 4:
        return np.rot90(test,dists.index(bestdist))
    else:
        return np.rot90(testr,dists.index(bestdist)-4)
    

def BaseMatrix():
    
    M = np.zeros((11,11),int)
    M[0,3:8] = -1
    M[1,5] = -1
    M[3,5] = 1
    M[4,4] = 1
    M[4,5] = 1
   #M= M + np.rot90(M) +  np.rot90( np.rot90(M)) + np.rot90(np.rot90( np.rot90(M))) 
    M2 = np.zeros((11,11),int)
    M2[5,5] = 2
    return M + np.rot90(M) +  np.rot90( np.rot90(M)) + np.rot90(np.rot90( np.rot90(M))) + M2

def MakeMove(initconf,move):
    zz = np.matrix.copy(initconf)
    start = move.split('-')[0]  #(l.split(',')[0].split('/')[0].split('-')[0])
    
    tmp = initconf[ord(start[0]) - ord('a'),int(start[1:])-1] 
    zz[ord(start[0]) - ord('a'),int(start[1:])-1]  = 0
    finish = move.split('-')[1]
    #initconf[ord(start[0]) - ord('a'),int(start[1:])-1]  = 0
    zz[ord(finish[0]) - ord('a'),int(finish[1:])-1]  = tmp
    currx = ord(finish[0]) - ord('a')
    curry = int(finish[1:])-1
    #white captures
    #check if moved piece is sandwiching anyone

    #black captures
    
    #if int(finish[1:])-1 != int(start[1:])-1:        
    if ord(finish[0]) - ord('a') != ord(start[0]) - ord('a'):
        if curry < 9  and curry>=2: 
            
            if np.dot(zz[currx,curry:curry+3],np.array([-1,1,-1])) == 3:
                zz[currx,curry+1] = 0
           
            #if np.dot(zz[currx,curry:curry+3],np.array([-1,1,-1])) == 3:
            #    zz[currx,curry+1] = 0  
            if np.dot(zz[currx,curry-1:curry-4:-1],np.array([-1,1,-1])) == 3:
                zz[currx-1,curry] = 0
            #if np.dot(zz[currx-1:currx-4:-1,curry],np.array([-1,1,-1])) == 3:
            #    zz[currx-1,curry] = 0  
    print ord(finish[0]) - ord('a') , ord(start[0]) - ord('a')
    #if ord(finish[0]) - ord('a') != ord(start[0]) - ord('a'):
    if int(finish[1:])-1 != int(start[1:])-1:      
        
        if currx<9 and currx >= 2:
            
            if np.dot(zz[currx:currx+3,curry],np.array([-1,1,-1])) == 3:
                zz[currx+1,curry] = 0
            #if np.dot(zz[currx:currx+3,curry],np.array([-1,1,-1])) == 3:
            #    zz[currx+1,curry] = 0  
            
            if np.dot(zz[currx,xrange(curry-1,curry-4,-1)],np.array([-1,1,-1])) == 3:
               zz[currx-1,curry] = 0
                
            if np.dot(zz[currx:currx-3:-1,curry],np.array([-1,1,-1])) == 3:
                zz[currx-1,curry] = 0
    #white capture
    if ord(finish[0]) - ord('a') != ord(start[0]) - ord('a'):
        if curry < 9  and curry>=2: 
            
            if np.dot(zz[currx,curry:curry+3],np.array([1,-1,1])) == 3:
                zz[currx,curry+1] = 0
           
            #if np.dot(zz[currx,curry:curry+3],np.array([-1,1,-1])) == 3:
            #    zz[currx,curry+1] = 0  
            if np.dot(zz[currx,curry-1:curry-4:-1],np.array([1,-1,1])) == 3:
                zz[currx-1,curry] = 0
            #if np.dot(zz[currx-1:currx-4:-1,curry],np.array([-1,1,-1])) == 3:
            #    zz[currx-1,curry] = 0  
    
    if int(finish[1:])-1 != int(start[1:])-1:      
        
        if currx<9 and currx >= 2:
            print zz[currx:currx-3:-1,curry],zz[currx,xrange(curry-1,curry-4,-1)]
            print zz[currx:currx+3:,curry],zz[currx,xrange(curry,curry+3,-1)]
            if np.dot(zz[currx:currx+3,curry],np.array([1,-1,-1])) == 3:
                zz[currx+1,curry] = 0
            #if np.dot(zz[currx:currx+3,curry],np.array([-1,1,-1])) == 3:
            #    zz[currx+1,curry] = 0  
            
            if np.dot(zz[currx,xrange(curry-1,curry-4,-1)],np.array([1,-1,1])) == 3:
               zz[currx-1,curry] = 0
                
            if np.dot(zz[currx:currx-3:-1,curry],np.array([1,-1,1])) == 3:
                zz[currx-1,curry] = 0
    
    return zz
    

astercount = 0
move1 = []
move2 = []
move3 = []
move4 = []
whitewin = []
blackwin = []
draws = []
filesdir = 'C:\\...'
FH = open(filesdir + '\\' + 'hnefatafl_openings.txt', 'r')
#lines = [ i.decode('unicode_escape').encode('ascii','ignore').rstrip() for i in FH.readlines()]

seq = input('Input your game sequence ')

#for n,l in enumerate(lines):
    #print l
firstfourmoves = seq.split(',')[0].split('/')
z = BaseMatrix() 
#z = MakeMove(z,firstfourmoves[0])
#        start = (l.split(',')[0].split('/')[0].split('-')[0])
#        tmp = z[ord(start[0]) - ord('a'),int(start[1:])-1] 
#        z[ord(start[0]) - ord('a'),int(start[1:])-1]  = 0
#        finish = (l.split(',')[0].split('/')[0].split('-')[1])
#        z[ord(finish[0]) - ord('a'),int(finish[1:])-1]  = tmp

z1 = MakeMove(z,firstfourmoves[0])
move1.append(z1)
z2 = MakeMove(z1,firstfourmoves[1])
move2.append(z2)
z3 = MakeMove(z2,firstfourmoves[2])
move3.append(z3)
z4 = MakeMove(z3,firstfourmoves[3])
move4.append(z4)
