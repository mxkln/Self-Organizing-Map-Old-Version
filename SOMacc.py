# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 09:03:11 2023

@author: mkln01
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import circle_fit as cf

from numba import jit, njit

def normalize_data(data,image_size): #normalizes particle coordinates to 1-0
    data = np.divide(data, image_size)
    return data

@njit(target_backend='cuda',fastmath = True)
def euclidean_distance(a,b): #calculates euclidean distance of all particles in b to all particles in a
    distances = []
    for i in range(len(a)):
        distance = np.sqrt((b[:,0]-a[i,0])**2+(b[:,1]-a[i,1])**2)
        distances.append(distance)
    return distances

@njit(target_backend='cuda',fastmath = True)
def bmu(a,b,dmax): # calculates the nearest neighbour of each particle in a to b, gives out index of bmu and distance
    distances = euclidean_distance(a,b)
    winners = []
    for i in range(len(distances)):
        minimum = np.argmin(distances[i])
        if (distances[i][minimum] > dmax):
            winner = (minimum,0)
        else:
            winner = (minimum,distances[i][minimum])
        winners.append(winner)
    return winners  

@njit(target_backend='cuda',fastmath = True)
def calculate_neighbours_with_gaussian_alpha(b, i, radius, alpha): # calculates all neighbours of ith particle in b within radius and ohmis gaussian alpha function 
    neighbours = []
    for j in range(len(b)):
        dist = b[i]-b[j]
        dist = np.sqrt(dist[0]**2+dist[1]**2)
        if dist <= radius:
            neighbours.append(alpha)
        else:
            gauss_alpha = alpha * np.exp(-((dist-radius)**2)/(2*radius**2))
            neighbours.append(gauss_alpha)
    return neighbours    

@njit(target_backend='cuda',fastmath = True)
def update_weights(a,b,alpha,radius,dmax): #calculates weight delta from "old" weight vectors and winner neutron + neighbors
    best_matching = bmu(a,b,dmax)
    deltaw = np.zeros((len(b),2))
    activated_neutrons = np.zeros((len(b),1))
    for i in range(len(a)):
        winner = best_matching[i][0]
        activated_neutrons[winner] = activated_neutrons[winner] + 1
        if (best_matching[i][1]) > 0:
            neighbours = np.array(calculate_neighbours_with_gaussian_alpha(b,winner,radius,alpha))
        else: 
            neighbours = np.zeros((len(b)))
        deltawx = neighbours*(a[i,0]-b[winner][0])  
        deltawy = neighbours*(a[i,1]-b[winner][1])
        deltawxy = np.column_stack((deltawx,deltawy))#*alpha not used with calculate_neighbours_with_gaussian_alpha
        deltaw =  deltaw + deltawxy
    return deltaw, activated_neutrons

def process(a,b,alpha,radius,dmax):
    deltab, bactivated = update_weights(a,b,alpha,radius,dmax)
    b = b + deltab
    deltaa, aactivated = update_weights(b,a,alpha,radius,dmax)
    a = a + deltaa
    return a, b, aactivated, bactivated

def matching(aweights, bweights, epsilon, afilter, bfilter):
    matches = np.empty((0,2),dtype=int)
    dist = euclidean_distance(aweights,bweights)
    dist = np.multiply(dist,np.expand_dims(afilter, axis=1))
    revdist = euclidean_distance(bweights,aweights)
    revdist = np.multiply(revdist,np.expand_dims(bfilter, axis=1))
    for i in range(len(dist)):
        mindist = dist[i].min()
        index = np.where(dist[i]==mindist)
        minrev = revdist[index[0][0]].min()
        revparticledist = revdist[index[0][0]][i]
        if mindist <= epsilon and mindist >= 0 and revparticledist == minrev and minrev >= 0:
            matched = np.array((i,index[0][0]),dtype=int)#,dtype=float)
            matches = np.vstack((matches,matched))
    return matches

@njit(target_backend='cuda',fastmath = True)
def make_arrows(match, anorm, bnorm):
    arrows = np.zeros((len(match),4))
    for i in range(len(match)):
        arrows[i][0] = anorm[match[i][0]][0]
        arrows[i][1] = anorm[match[i][0]][1]
        arrows[i][2] = bnorm[match[i][1]][0] - anorm[match[i][0]][0]
        arrows[i][3] = bnorm[match[i][1]][1] - anorm[match[i][0]][1]
    return arrows

def plotting(anorm,bnorm,aweights,bweights,arrows):
    plt.figure(figsize=(14,7))
    plt.subplot(1, 2, 1)
    plt.scatter(anorm[:,0], anorm[:,1])
    plt.scatter(bnorm[:,0], bnorm[:,1])
    for i in range(len(arrows)):
        plt.arrow(arrows[i][0],arrows[i][1],arrows[i][2],arrows[i][3], head_width=0.01)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.gca().invert_yaxis()
    plt.subplot(1, 2, 2)
    plt.scatter(aweights[:,0], aweights[:,1])
    plt.scatter(bweights[:,0], bweights[:,1])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.gca().invert_yaxis()
    plt.show()
    
@njit(target_backend='cuda',fastmath = True)   
def radius_function(x,r0,rf,cycles):
    radius = ((rf-r0)/cycles) * x + r0
    return radius

def som(a,b,cycles,startradius,finalradius,alpha,image_size,epsilon,plot,output,dmax):
    
    anorm = normalize_data(a,image_size)
    bnorm = normalize_data(b,image_size)
    aweights = anorm
    bweights = bnorm 
    aact = np.arange(len(a))
    bact = np.arange(len(b))
    afilter = np.ones((len(a)))
    bfilter = np.ones((len(b)))
    
    for i in range(cycles):
        radius = radius_function(i,startradius,finalradius,cycles)
        #alpha = alpha_function(i,a0,af,cycles)
        aweights, bweights, aactivated, bactivated= process(aweights, bweights, alpha,radius,dmax)
        aact = np.column_stack((aact,aactivated))
        bact = np.column_stack((bact,bactivated))
        if i >= 3: 
            for x in range(len(a)):
                if aact[x][i]==0 and aact[x][i-1]==0 and aact[x][i-2]==0:
                    aweights[x] = (-1,-1)
                    afilter[x] = -1
                    #aweights = np.delete(aweights, x, 0)
            for y in range(len(b)):
                if bact[y][i]==0 and bact[y][i-1]==0 and bact[y][i-2]==0:
                    bweights[y] = (-1,-1)
                    #bweights = np.delete(bweights, y, 0)
                    bfilter[y] = -1
                              
    match = matching(aweights, bweights, epsilon,afilter,bfilter)
    
    if plot=="yes" or output=="percentage":
        arrows = make_arrows(match, anorm, bnorm)

    if plot=="yes":
        plotting(anorm,bnorm,aweights,bweights,arrows)
    if output=="percentage":
        precentage = len(arrows)/len(anorm)
        weightdelta, _ = update_weights(aweights, bweights,alpha,finalradius,dmax)
        weightdelta = np.mean(weightdelta)
        return precentage, weightdelta
    elif output=="matches":
        return match
    elif output=="act":
        return afilter,bfilter
    
def tracing(matchlist,min_length): #returns traces of all particles from first image with minimum length
    traces = []
    for particle in np.array(matchlist[0]):
        trace = []
        trace.append(particle[0])
        trace.append(particle[1])
        for i in range(len(matchlist)-1):
            next_image = np.array(matchlist[i+1])
            if trace[-1] in next_image[:,0]:
                where = np.where(next_image[:,0] == trace[-1])[0][0]
                trace.append(next_image[where][1])
            else:
                break                       
        traces.append(trace)
    long_traces=[trace for trace in traces if len(trace)>=min_length]
    return long_traces   

def alltracing(matchlist,min_length): #returns traces of all particles from all images with minimum length; -1 if particle was not yet in image
    traces = [] 
    for j in range(len(matchlist)-min_length-1):      
        for particle in matchlist[j]:
            trace = []
            for x in range(j):
                trace.append(-1)
            trace.append(particle[0])
            trace.append(particle[1])
            already_saved = False
            for sublist in traces:
                if len(sublist) > j:
                    if particle[0] == sublist[j]:
                        already_saved = True
            for i in range(len(matchlist)-j-1):
                next_image = matchlist[i+1+j]
                if trace[-1] in next_image[:,0]:
                    where = np.where(next_image[:,0] == trace[-1])[0][0]
                    trace.append(next_image[where][1])
                else:
                    break
            if already_saved == False:
                traces.append(trace)
    long_traces=[trace for trace in traces if len([x for x in trace if x != -1])>=min_length]
    return long_traces

def traces_to_coords(traces,starting_image,image_size,path): #converts traces back to coordinates
    coords = []
    for trace in traces:
        coord = []
        coords.append(coord)
    maximum_length=len(max(traces, key=len))
    for i in range(maximum_length):
        positions = normalize_data(np.load(path+str(starting_image+i)+".npy"),image_size)
        for trace in traces:  
            if len(trace) > i:
                index = traces.index(trace)
                if trace[i] != -1:
                    coords[index].append(positions[trace[i]])
                #else:
                 #   coords[index].append((-1,-1))
    for trace in traces:
        index = traces.index(trace)
        coords[index] = np.array(coords[index])
    return coords

def plotting_coords(coords):
    plt.figure(figsize=(7,7))
    for i in range(len(coords)):
        #plt.scatter(coords[i][:,0],coords[i][:,1],s=3)
        plt.plot(coords[i][:,0],coords[i][:,1], linewidth=1)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.gca().invert_yaxis()
    plt.show
    
def angular_velocity_all(coords,min_length, radius_region):
    angular_velocities = []
    for i in range(len(coords)):  
        omega, arc_length, radius = angular_velocity_single(coords[i],"no")
        if arc_length > min_length and radius_region[0]<radius<radius_region[1]:
            angular_velocities.append((omega,i))
    
    return angular_velocities

def angular_velocity_single(coords,plot):
    xc,yc,r,_ = cf.least_squares_circle(coords)
    fps = 60
    dt = len(coords)/fps
    c = (xc,yc)
    a = coords[0]
    b = coords[-1]
    ab = math.dist(a,b)
    ac = math.dist(a,c)
    bc = math.dist(b,c)
    theta = np.arccos((bc**2+ac**2-ab**2)/(2*bc*ac)) #radians
    arc_length = theta * r
    theta = theta * (180/np.pi) #degrees
    omega = theta / dt # degrees/s
    
    if plot == "yes":
        plt.figure(figsize=(7,7))
        fig = plt.gcf()
        ax = fig.gca()
        plt.scatter(coords[:,0],coords[:,1],s=7)
        plt.scatter(xc,yc,s=3)
        circle = plt.Circle((xc,yc),r,fill=False, color='green',ls='--')
        ax.add_patch(circle)
        plt.plot(coords[:,0],coords[:,1],lw=5)
        plt.xticks([0,.1953125,.390625,.5859375,.78125,0.9765625],[0,100,200,300,400,500])
        plt.yticks([0,.1953125,.390625,.5859375,.78125,0.9765625],[0,100,200,300,400,500])
        plt.xlabel("x [px]")
        plt.ylabel("y [px]")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.gca().invert_yaxis()
        plt.show
        
    return omega, arc_length, r

def matches_to_coords(matches,starting_image,image_size,path): #converts matches back to coordinates (1. x,y; 2. x,y)
    coords = np.zeros((len(matches),4))
    positions1 = normalize_data(np.load(path+str(starting_image)+".npy"),image_size)
    positions2 = normalize_data(np.load(path+str(starting_image+1)+".npy"),image_size)
    for i in range(len(matches)):
        index1 = matches[i][0]
        index2 = matches[i][1]
        coords[i] = (positions1[index1][0],positions1[index1][1],positions2[index2][0],positions2[index2][1])
    return coords

def coords_to_PIV(coords,boxes,scale=None):
    ###calculating
    boxcoords = np.zeros((len(coords),4))
    for k in range(len(coords)):
        xbox = np.trunc(coords[k][0]*boxes)
        ybox = np.trunc(coords[k][1]*boxes)
        xdist = coords[k][2]-coords[k][0]
        ydist = coords[k][3]-coords[k][1]
        boxcoords[k] = (xbox,ybox,xdist,-ydist)
    
    u = np.zeros((boxes,boxes))
    v = np.zeros((boxes,boxes))
    for i in range(boxes):
        for j in range(boxes):
            meanx = np.empty(0)
            meany = np.empty(0)
            for coord in boxcoords:
                if int(coord[0])==j and int(coord[1])==i:
                    meanx = np.append(meanx, coord[2])
                    meany = np.append(meany, coord[3])
            if len(meanx)>0:
                meanx = np.mean(meanx)
            else: 
                meanx = np.nan
            if len(meany)>0:
                meany = np.mean(meany)
            else:
                meany = np.nan
            u[i][j] = meanx
            v[i][j] = meany
    
    ###plotting    
    plt.figure(figsize=(7,7))
    #for i in range(boxes-1):
    #    plt.axhline(y=(i+1)/boxes, color='k', linestyle='dotted')
    #    plt.axvline(x=(i+1)/boxes, color='k', linestyle='dotted')
    x = np.arange(0,1,1/boxes)
    y = np.arange(0,1,1/boxes)
    X,Y = np.meshgrid(x,y)
    X = X + 1/(2*boxes)
    Y = Y + 1/(2*boxes)
    n = -2
    color_array = np.sqrt(((abs(v)-n)/2)**2 + ((abs(u)-n)/2)**2)
    if scale==None:
        plt.quiver(X,Y,u,v,color_array)
    else:
        plt.quiver(X,Y,u,v,color_array,scale=scale)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.gca().invert_yaxis()
    plt.show