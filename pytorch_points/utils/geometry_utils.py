import numpy as np
import random
import openmesh as om
import os
from matplotlib import cm

def read_trimesh(filename):
    """
    load vertices and faces of a mesh file
    return:
        V (N,3) floats
        F (F,3) int64
    """
    mesh = om.read_trimesh(filename)
    V = mesh.points()
    face_lists = []
    for f in mesh.face_vertex_indices():
        face_lists.append(f)
    F = np.stack(face_lists, axis=0)
    return V, F

def write_trimesh(filename, V, F, v_colors=None, cmap_name="Set1"):
    """
    write a mesh with (N,3) vertices and (F,3) faces to file
    """
    assert(V.ndim==2)
    assert(F.ndim==2)
    assert(F.shape[-1]==3)

    mesh = om.TriMesh()
    if v_colors is not None:
        assert(v_colors.shape[0]==V.shape[0])
        if v_colors.size == V.shape[0]:
            cmap = cm.get_cmap(cmap_name)
            minV, maxV = v_colors.min(), v_colors.max()
            v_colors = (v_colors-minV)/maxV
            v_colors = [cmap(color) for color in v_colors]
        mesh.request_vertex_colors()


    for v in range(V.shape[0]):
        vh = mesh.add_vertex(V[v])
        if mesh.has_vertex_colors() and v_colors is not None:
            mesh.set_color(vh, v_colors[v])

    for f in range(F.shape[0]):
        vh_list = [mesh.vertex_handle(vIdx) for vIdx in F[f]]
        fh0 = mesh.add_face(vh_list)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    om.write_mesh(filename, mesh, vertex_color=mesh.has_vertex_colors())


def generatePolygon( ctrX, ctrY, aveRadius, irregularity, spikeyness, randRot, numVerts) :
    '''Start with the centre of the polygon at ctrX, ctrY,
    then creates the polygon by sampling points on a circle around the centre.
    Randon noise is added by varying the angular spacing between sequential points,
    and by varying the radial distance of each point from the centre.

    Params:
    ctrX, ctrY - coordinates of the "centre" of the polygon, scalars
    aveRadius - in px, the average radius of this polygon, this roughly controls how large the polygon is, really only useful for order of magnitude.
    irregularity - [0,1] indicating how much variance there is in the angular spacing of vertices. [0,1] will map to [0, 2pi/numberOfVerts]
    spikeyness - [0,1] indicating how much variance there is in each vertex from the circle of radius aveRadius. [0,1] will map to [0, aveRadius]
    randRot - [0, 1] indicating how much variance there is in the mean angular position of the first vertex
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
    angle = random.uniform(0, randRot)*2*np.pi
    for i in range(numVerts) :
        r_i = np.clip(np.random.normal(aveRadius, spikeyness), 0, 2*aveRadius )
        x = ctrX + r_i*np.cos(angle)
        y = ctrY + r_i*np.sin(angle)
        points.append((x,y))

        angle = angle + angleSteps[i]

    return points