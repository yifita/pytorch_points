import numpy as np
import random

def generatePolygon( ctrX, ctrY, aveRadius, irregularity, spikeyness, numVerts) :
    '''Start with the centre of the polygon at ctrX, ctrY, 
    then creates the polygon by sampling points on a circle around the centre. 
    Randon noise is added by varying the angular spacing between sequential points,
    and by varying the radial distance of each point from the centre.

    Params:
    ctrX, ctrY - coordinates of the "centre" of the polygon, scalars
    aveRadius - in px, the average radius of this polygon, this roughly controls how large the polygon is, really only useful for order of magnitude.
    irregularity - [0,1] indicating how much variance there is in the angular spacing of vertices. [0,1] will map to [0, 2pi/numberOfVerts]
    spikeyness - [0,1] indicating how much variance there is in each vertex from the circle of radius aveRadius. [0,1] will map to [0, aveRadius]
    numVerts - self-explanatory

    Returns a list of vertices, in CCW order.
    '''

    irregularity = np.clip( irregularity, 0,1 ) * 2*np.pi / numVerts
    spikeyness = np.clip( spikeyness, 0,1 ) * aveRadius

    # generate n angle steps
    angleSteps = []
    lower = (2*np.pi / numVerts) - irregularity
    upper = (2*np.pi / numVerts) + irregularity
    sum = 0
    for i in range(numVerts) :
        tmp = np.random.uniform(lower, upper)
        angleSteps.append( tmp )
        sum = sum + tmp

    # normalize the steps so that point 0 and point n+1 are the same
    k = sum / (2*np.pi)
    for i in range(numVerts) :
        angleSteps[i] = angleSteps[i] / k

    # now generate the points
    points = []
    # angle = random.uniform(0, 2*np.pi)
    angle = 5*np.pi/6
    for i in range(numVerts) :
        r_i = np.clip(np.random.normal(aveRadius, spikeyness), 0, 2*aveRadius )
        x = ctrX + r_i*np.cos(angle)
        y = ctrY + r_i*np.sin(angle)
        points.append((x,y))

        angle = angle + angleSteps[i]

    return points