import numpy as np
import random
import openmesh as om
import os
from matplotlib import cm
import torch
from collections import abc


def read_trimesh(filename, return_mesh=False, **kwargs):
    """
    load vertices and faces of a mesh file
    return:
        V (N,3) floats
        F (F,3) int64
        properties "vertex_colors" "face_colors" etc
        mesh    trimesh object
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

    if return_mesh:
        return V, F, properties, mesh

    return V, F, properties

def write_trimesh(filename, V, F, v_colors=None, f_colors=None, v_normals=True, cmap_name="Set1", **kwargs):
    """
    write a mesh with (N,3) vertices and (F,3) faces to file
    """
    assert(V.ndim==2)
    assert(F.ndim==2)
    assert(F.shape[-1]==3)

    mesh = array_to_mesh(V, F, v_colors=v_colors, f_colors=f_colors, v_normals=v_normals, cmap_name=cmap_name)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    om.write_mesh(filename, mesh, vertex_color=mesh.has_vertex_colors(), **kwargs)


def array_to_mesh(V, F, v_colors=None, f_colors=None, v_normals=True, cmap_name="Set1"):
    """
    convert a mesh with (N,3) vertices and (F,3) faces to a trimesh object
    """
    assert(V.ndim==2)
    assert(F.ndim==2)
    assert(F.shape[-1]==3)
    if isinstance(V, torch.Tensor):
        V = V.detach().cpu().numpy()
    if isinstance(F, torch.Tensor):
        F = F.detach().cpu().numpy()

    mesh = om.TriMesh()
    if v_colors is not None:
        if isinstance(v_colors, torch.Tensor):
            v_colors = v_colors.detach().cpu().numpy()
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
    Given F matrix of a triangle mesh return unique edge vertices of a mesh Ex2 tensor
    params:
        F (F,L) tensor or numpy
    return:
        E (E,2) tensor or numpy
    """
    if isinstance(F, torch.Tensor):
        # F,L,2
        edges = torch.stack([F, F[:,[1, 2, 0]]], dim=-1)
        edges = torch.sort(edges, dim=-1)[0]
        # FxL,2
        edges = edges.reshape(-1, 2)
        # E,2
        edges = torch.unique(edges, dim=0)
    else:
        edges = np.stack([F, F[:,[1,2,0]]], axis=-1)
        edges = np.sort(edges, axis=-1)
        edges = edges.reshape([-1, 2])
        edges = np.unique(edges, axis=0)
    return edges

def get_edge_lengths(vertices, edge_points=None):
    """
    get edge length using edge_points from get_edge_points(mesh)
    """
    edge_lengths = torch.norm(vertices[edge_points[:, 0]] - mesh.vs[edge_points[:, 1]], ord=2, axis=1)
    return edge_lengths

class Mesh(abc.Mapping):
    """
    create mesh object from vertices and faces with attributes
    ve:            List(List(int64)) vertex - edge idx
    edges:         (E,2) int32 numpy array edges represented as sorted vertex indices
    gemm_edges     (E,4) int64 numpy array indices of the four neighboring edges
    ======
    :param
        vertices   (V,3) float32
        faces      (F,3) int64
    """
    def __init__(self, filepath: str = None, vertices: torch.Tensor = None, faces: torch.Tensor = None):
        if filepath is not None:
            mesh = om.read_trimesh(filepath)
            vertices = mesh.points()

        face_lists = []
        for f in mesh.face_vertex_indices():
            face_lists.append(f)
        faces = np.stack(face_lists, axis=0)

        mesh.request_face_normals()
        if not mesh.has_vertex_normals():
            mesh.request_vertex_normals()
            mesh.update_normals()
        v_normals = mesh.vertex_normals()
        f_normals = mesh.face_normals()
        vertices = np.concatenate([vertices, v_normals], axis=-1)
        faces = np.concatenate([faces, f_normals], axis=-1)

        self.vs = vertices
        self.fs = faces
        build_gemm(self, faces)
        features = []
        edge_points = get_edge_points(self)


        with np.errstate(divide='raise'):
            try:
                for extractor in [dihedral_angle, symmetric_opposite_angles, symmetric_ratios]:
                    feature = extractor(mesh, edge_points)
                    features.append(feature)
                return np.concatenate(features, axis=0)
            except Exception as e:
                print(e)
                raise ValueError(mesh.filename, 'bad features')

    def __getitem__(self, key):
        return getattr(self, key)

    def __len__(self):
        return len(self.keys())

    def __iter__(self):
        return iter(self.keys())

    def funcname(parameter_list):
        pass

def compute_face_normals_and_areas(mesh, faces):
    face_normals = torch.cross(mesh.vs[faces[:, 1]] - mesh.vs[faces[:, 0]],
                            mesh.vs[faces[:, 2]] - mesh.vs[faces[:, 1]])
    face_areas = torch.sqrt((face_normals ** 2).sum(dim=1))
    face_normals /= face_areas.unsqueeze(-1)
    assert (not np.any(face_areas.unsqueeze(-1) == 0)), 'has zero area face: %s' % mesh.filename
    face_areas *= 0.5
    return face_normals, face_areas


def build_gemm(mesh, faces):
    """
    ve:            List(List(int64)) vertex - edge idx
    edges:         (E,2) int32 numpy array edges represented as sorted vertex indices
    gemm_edges     (E,4) int64 numpy array indices of the four neighboring edges
    """
    mesh.ve = [[] for _ in mesh.vs]
    edge_nb = []
    # sides = []
    edge2key = dict()
    edges = []
    edges_count = 0
    nb_count = []
    for face_id, face in enumerate(faces):
        faces_edges = []
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            faces_edges.append(cur_edge)
        for idx, edge in enumerate(faces_edges):
            edge = tuple(sorted(list(edge)))
            faces_edges[idx] = edge
            if edge not in edge2key:
                edge2key[edge] = edges_count
                edges.append(list(edge))
                edge_nb.append([-1, -1, -1, -1])
                # sides.append([-1, -1, -1, -1])
                mesh.ve[edge[0]].append(edges_count)
                mesh.ve[edge[1]].append(edges_count)
                # mesh.edge_areas.append(0)
                nb_count.append(0)
                edges_count += 1
            # mesh.edge_areas[edge2key[edge]] += face_areas[face_id] / 3
        for idx, edge in enumerate(faces_edges):
            edge_key = edge2key[edge]
            edge_nb[edge_key][nb_count[edge_key]] = edge2key[faces_edges[(idx + 1) % 3]]
            edge_nb[edge_key][nb_count[edge_key] + 1] = edge2key[faces_edges[(idx + 2) % 3]]
            nb_count[edge_key] += 2
        # for idx, edge in enumerate(faces_edges):
        #     edge_key = edge2key[edge]
        #     sides[edge_key][nb_count[edge_key] - 2] = nb_count[edge2key[faces_edges[(idx + 1) % 3]]] - 1
        #     sides[edge_key][nb_count[edge_key] - 1] = nb_count[edge2key[faces_edges[(idx + 2) % 3]]] - 2
    mesh.edges = np.array(edges, dtype=np.int32)
    mesh.gemm_edges = np.array(edge_nb, dtype=np.int64)
    # mesh.sides = np.array(sides, dtype=np.int64
    mesh.edges_count = edges_count
    # mesh.edge_areas = vertices(mesh.edge_areas, dtype=np.float32) / np.sum(face_areas) #todo whats the difference between edge_areas and edge_lengths?


def get_edge_points(mesh):
    """
    get 4 edge points
    """
    edge_points = np.zeros([mesh.edges_count, 4], dtype=np.int32)
    for edge_id, edge in enumerate(mesh.edges):
        edge_points[edge_id] = get_side_points(mesh, edge_id)
    return edge_points


def get_side_points(mesh, edge_id):
    """
    return the four vertex indices around an edge
       1
    0 <|> 3
       2
    """
    edge_a = mesh.edges[edge_id]

    if mesh.gemm_edges[edge_id, 0] == -1:
        edge_b = mesh.edges[mesh.gemm_edges[edge_id, 2]]
        edge_c = mesh.edges[mesh.gemm_edges[edge_id, 3]]
    else:
        edge_b = mesh.edges[mesh.gemm_edges[edge_id, 0]]
        edge_c = mesh.edges[mesh.gemm_edges[edge_id, 1]]
    if mesh.gemm_edges[edge_id, 2] == -1:
        edge_d = mesh.edges[mesh.gemm_edges[edge_id, 0]]
        edge_e = mesh.edges[mesh.gemm_edges[edge_id, 1]]
    else:
        edge_d = mesh.edges[mesh.gemm_edges[edge_id, 2]]
        edge_e = mesh.edges[mesh.gemm_edges[edge_id, 3]]
    first_vertex = 0
    second_vertex = 0
    third_vertex = 0
    if edge_a[1] in edge_b:
        first_vertex = 1
    if edge_b[1] in edge_c:
        second_vertex = 1
    if edge_d[1] in edge_e:
        third_vertex = 1
    return [edge_a[first_vertex], edge_a[1 - first_vertex], edge_b[second_vertex], edge_d[third_vertex]]


def get_normals(mesh, edge_points, side):
    """
    return the face normal of 4 edge points on the specified side
    """
    edge_a = mesh.vs[edge_points[:, side // 2 + 2]] - mesh.vs[edge_points[:, side // 2]]
    edge_b = mesh.vs[edge_points[:, 1 - side // 2]] - mesh.vs[edge_points[:, side // 2]]
    normals = np.cross(edge_a, edge_b)
    div = np.linalg.norm(normals, ord=2, axis=1)
    normals /= (div[:, np.newaxis]+1e-5)
    return normals


def dihedral_angle(mesh, edge_points):
    """
    return the face-to-face angle of an edge specified by the 4 edge_points
    """
    normals_a = get_normals(mesh, edge_points, 0)
    normals_b = get_normals(mesh, edge_points, 3)
    dot = np.sum(normals_a * normals_b, axis=1).clip(-1, 1)
    angles = np.expand_dims(np.pi - np.arccos(dot), axis=0)
    return angles