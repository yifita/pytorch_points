import numpy as np
import random
import openmesh as om
import os
from matplotlib import cm
import torch

def read_trimesh(filename, **kwargs):
    """
    load vertices and faces of a mesh file
    return:
        V (N,3) floats
        F (F,3) int64
    """
    try:
        mesh = om.read_trimesh(filename, **kwargs)
    except RuntimeError as e:
        mesh = om.read_trimesh(filename)

    V = mesh.points()
    face_lists = []
    for f in mesh.face_vertex_indices():
        face_lists.append(f)
    F = np.stack(face_lists, axis=0)

    mesh.request_face_normals()
    if not mesh.has_vertex_normals():
        mesh.request_vertex_normals()
        mesh.update_normals()
    v_normals = mesh.vertex_normals()
    f_normals = mesh.face_normals()
    V = np.concatenate([V, v_normals], axis=-1)
    F = np.concatenate([F, f_normals], axis=-1)

    properties = {}
    if mesh.has_vertex_colors():
        v_colors = mesh.vertex_colors()
        properties["vertex_colors"] = v_colors
    if mesh.has_face_colors():
        f_colors = mesh.face_colors()
        properties["face_colors"] = f_colors

    return V, F, properties

def write_trimesh(filename, V, F, v_colors=None, f_colors=None, v_normals=True, cmap_name="Set1"):
    """
    write a mesh with (N,3) vertices and (F,3) faces to file
    """
    assert(V.ndim==2)
    assert(F.ndim==2)
    assert(F.shape[-1]==3)

    mesh = array_to_mesh(V, F, v_colors=v_colors, f_colors=f_colors, v_normals=v_normals, cmap_name=cmap_name)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    om.write_mesh(filename, mesh, vertex_color=mesh.has_vertex_colors())


def array_to_mesh(V, F, v_colors=None, f_colors=None, v_normals=True, cmap_name="Set1"):
    """
    convert a mesh with (N,3) vertices and (F,3) faces to a trimesh object
    """
    assert(V.ndim==2)
    assert(F.ndim==2)
    assert(F.shape[-1]==3)

    mesh = om.TriMesh()
    if v_colors is not None:
        assert(v_colors.shape[0]==V.shape[0])
        # 1D scalar for each face
        if v_colors.size == V.shape[0]:
            cmap = cm.get_cmap(cmap_name)
            minV, maxV = v_colors.min(), v_colors.max()
            v_colors = (v_colors-minV)/maxV
            v_colors = [cmap(color) for color in v_colors]
        mesh.request_vertex_colors()

    if f_colors is not None:
        assert(f_colors.shape[0]==F.shape[0])
        # 1D scalar for each face
        if f_colors.size == F.shape[0]:
            cmap = cm.get_cmap(cmap_name)
            minV, maxV = f_colors.min(), f_colors.max()
            f_colors = (f_colors-minV)/maxV
            f_colors = [cmap(color) for color in f_colors]
        mesh.request_face_colors()

    for v in range(V.shape[0]):
        vh = mesh.add_vertex(V[v])
        if mesh.has_vertex_colors() and v_colors is not None:
            mesh.set_color(vh, v_colors[v])

    for f in range(F.shape[0]):
        vh_list = [mesh.vertex_handle(vIdx) for vIdx in F[f]]
        fh0 = mesh.add_face(vh_list)
        if mesh.has_face_colors() and f_colors is not None:
            mesh.set_color(vh, f_colors[v])

    mesh.request_face_normals()
    if v_normals:
        mesh.request_vertex_normals()

    mesh.update_normals()
    return mesh


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


def edge_vertex_indices(F):
    """
    Given F return unique edge vertices of a mesh Ex2 tensor
    params:
        F (F,3) tensor or numpy
    return:
        E (E,2) tensor or numpy
    """
    if isinstance(F, torch.Tensor):
        # F,3,2
        edges = torch.stack([F, F[:,[1, 2, 0]]], dim=-1)
        edges = torch.sort(edges, dim=-1)[0]
        # Fx3,2
        edges = edges.reshape(-1, 2)
        # E,2
        edges = torch.unique(edges, dim=0)[0]
    else:
        edges = np.stack([F, F[:,[1,2,0]]], axis=-1)
        edges = np.sort(edges, axis=-1)
        edges = edges.reshape([-1, 2])
        edges = np.unique(edges, axis=0)
    return edges
