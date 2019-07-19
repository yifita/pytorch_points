""" 
code courtesy of 
https://github.com/erikwijmans/Pointnet2_PyTorch
"""

import torch
import faiss
import numpy as np
from scipy import sparse
from threading import Thread

from .._ext import sampling
from .._ext import linalg

if torch.cuda.is_available():
    from .faiss_setup import GPU_RES


def normalize_point_batch(pc, NCHW=True):
    """
    normalize a batch of point clouds
    :param
        pc      [B, N, 3] or [B, 3, N]
        NCHW    if True, treat the second dimension as channel dimension
    :return
        pc      normalized point clouds, same shape as input
        centroid [B, 1, 3] or [B, 3, 1] center of point clouds
        furthest_distance [B, 1, 1] scale of point clouds
    """
    point_axis = 2 if NCHW else 1
    dim_axis = 1 if NCHW else 2
    centroid = torch.mean(pc, dim=point_axis, keepdim=True)
    pc = pc - centroid
    furthest_distance, _ = torch.max(
        torch.sqrt(torch.sum(pc ** 2, dim=dim_axis, keepdim=True)), dim=point_axis, keepdim=True)
    pc = pc / furthest_distance
    return pc, centroid, furthest_distance


def channel_shuffle(x, groups=2):
    '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
    N, C, H, W = x.size()
    g = groups
    return x.view(N, g, C/g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


def jitter_perturbation_point_cloud(batch_data, sigma=0.005, clip=0.02, is_2D=False, NCHW=True):
    if NCHW:
        batch_data = batch_data.transpose(1, 2)

    batch_size = batch_data.shape[0]
    chn = 2 if is_2D else 3
    jittered_data = sigma * torch.randn_like(batch_data)
    for b in range(batch_size):
        jittered_data[b].clamp_(-clip[b].item(), clip[b].item())
    jittered_data[:, :, chn:] = 0
    jittered_data += batch_data
    if NCHW:
        jittered_data = jittered_data.transpose(1, 2)
    return jittered_data


def __swig_ptr_from_FloatTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.float32
    return faiss.cast_integer_to_float_ptr(x.storage().data_ptr())


def __swig_ptr_from_LongTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.int64, 'dtype=%s' % x.dtype
    return faiss.cast_integer_to_long_ptr(x.storage().data_ptr())


def search_index_pytorch(database, x, k):
    """
    KNN search via Faiss
    :param
        database BxNxC
        x BxMxC
    :return
        D BxMxK
        I BxMxK
    """
    Dptr = database.storage().data_ptr()
    if not (x.is_cuda or database.is_cuda):
        index = faiss.IndexFlatL2(database.size(-1))
    else:
        index = faiss.GpuIndexFlatL2(
            GPU_RES, database.size(-1))  # dimension is 3
    index.add_c(database.size(0), faiss.cast_integer_to_float_ptr(Dptr))

    assert x.is_contiguous()
    n, d = x.size()
    assert d == index.d

    D = torch.empty((n, k), dtype=torch.float32, device=x.device)
    I = torch.empty((n, k), dtype=torch.int64, device=x.device)

    torch.cuda.synchronize()
    xptr = __swig_ptr_from_FloatTensor(x)
    Iptr = __swig_ptr_from_LongTensor(I)
    Dptr = __swig_ptr_from_FloatTensor(D)
    index.search_c(n, xptr,
                   k, Dptr, Iptr)
    torch.cuda.synchronize()
    index.reset()
    return D, I


class KNN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, query, points):
        """
        :param k: k in KNN
               query: BxMxC
               points: BxNxC
        :return:
            neighbors_points: BxMxK
            index_batch: BxMxK
        """
        # selected_gt: BxkxCxM
        # process each batch independently.
        index_batch = []
        distance_batch = []
        for i in range(points.shape[0]):
            D_var, I_var = search_index_pytorch(points[i], query[i], k)
            GPU_RES.syncDefaultStreamCurrentDevice()
            index_batch.append(I_var)  # M, k
            distance_batch.append(D_var)  # M, k

        # B, M, K
        index_batch = torch.stack(index_batch, dim=0)
        distance_batch = torch.stack(distance_batch, dim=0)
        ctx.mark_non_differentiable(index_batch, distance_batch)
        return index_batch, distance_batch


def faiss_knn(k, query, points, NCHW=True):
    """
    group batch of points to neighborhoods
    :param
        k: neighborhood size
        query: BxCxM or BxMxC
        points: BxCxN or BxNxC
        NCHW: if true, the second dimension is the channel dimension
    :return
        neighbor_points BxCxMxk (if NCHW) or BxMxkxC (otherwise)
        index_batch     BxMxk
        distance_batch  BxMxk
    """
    if NCHW:
        batch_size, channels, num_points = points.size()
        points_trans = points.transpose(2, 1).contiguous()
        query_trans = query.transpose(2, 1).contiguous()
    else:
        points_trans = points.contiguous()
        query_trans = query.contiguous()

    batch_size, num_points, _ = points_trans.size()
    # BxMxk
    index_batch, distance_batch = KNN.apply(k, query_trans, points_trans)
    # BxNxC -> BxMxNxC
    points_expanded = points_trans.unsqueeze(dim=1).expand(
        (-1, query.size(1), -1, -1))
    # BxMxk -> BxMxkxC
    index_batch_expanded = index_batch.unsqueeze(dim=-1).expand(
        (-1, -1, -1, points_trans.size(-1)))
    # BxMxkxC
    neighbor_points = torch.gather(points_expanded, 2, index_batch_expanded)
    index_batch = index_batch
    if NCHW:
        # BxCxMxk
        neighbor_points = neighbor_points.permute(0, 3, 1, 2).contiguous()
    return neighbor_points, index_batch, distance_batch


def __batch_distance_matrix_general(A, B):
    """
    :param
        A, B [B,N,C], [B,M,C]
    :return
        D [B,N,M]
    """
    r_A = torch.sum(A * A, dim=2, keepdim=True)
    r_B = torch.sum(B * B, dim=2, keepdim=True)
    m = torch.matmul(A, B.permute(0, 2, 1))
    D = r_A - 2 * m + r_B.permute(0, 2, 1)
    return D


def group_knn(k, query, points, unique=True, NCHW=True):
    """
    group batch of points to neighborhoods
    :param
        k: neighborhood size
        query: BxCxM or BxMxC
        points: BxCxN or BxNxC
        unique: neighborhood contains *unique* points
        NCHW: if true, the second dimension is the channel dimension
    :return
        neighbor_points BxCxMxk (if NCHW) or BxMxkxC (otherwise)
        index_batch     BxMxk
        distance_batch  BxMxk
    """
    if NCHW:
        batch_size, channels, num_points = points.size()
        points_trans = points.transpose(2, 1).contiguous()
        query_trans = query.transpose(2, 1).contiguous()
    else:
        points_trans = points.contiguous()
        query_trans = query.contiguous()

    batch_size, num_points, _ = points_trans.size()
    assert(num_points >= k
           ), "points size must be greater or equal to k"

    D = __batch_distance_matrix_general(query_trans, points_trans)
    if unique:
        # prepare duplicate entries
        points_np = points_trans.detach().cpu().numpy()
        indices_duplicated = np.ones(
            (batch_size, 1, num_points), dtype=np.int32)

        for idx in range(batch_size):
            _, indices = np.unique(points_np[idx], return_index=True, axis=0)
            indices_duplicated[idx, :, indices] = 0

        indices_duplicated = torch.from_numpy(
            indices_duplicated).to(device=D.device, dtype=torch.float32)
        D += torch.max(D) * indices_duplicated

    # (B,M,k)
    distances, point_indices = torch.topk(-D, k, dim=-1, sorted=True)
    # (B,N,C)->(B,M,N,C), (B,M,k)->(B,M,k,C)
    knn_trans = torch.gather(points_trans.unsqueeze(1).expand(-1, query_trans.size(1), -1, -1),
                             2,
                             point_indices.unsqueeze(-1).expand(-1, -1, -1, points_trans.size(-1)))

    if NCHW:
        knn_trans = knn_trans.permute(0, 3, 1, 2)

    return knn_trans, point_indices, -distances


class GatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, idx):
        r"""
        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor
        idx : torch.Tensor
            (B, npoint) tensor of the features to gather
        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """
        features = features.contiguous()
        idx = idx.contiguous()
        idx = idx.to(dtype=torch.int32)

        B, npoint = idx.size()
        _, C, N = features.size()

        output = torch.empty(
            B, C, npoint, dtype=features.dtype, device=features.device)
        output = sampling.gather_forward(
            B, C, N, npoint, features, idx, output
        )

        ctx.save_for_backward(idx)
        ctx.C = C
        ctx.N = N
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, = ctx.saved_tensors
        B, npoint = idx.size()

        grad_features = torch.zeros(
            B, ctx.C, ctx.N, dtype=grad_out.dtype, device=grad_out.device)
        grad_features = sampling.gather_backward(
            B, ctx.C, ctx.N, npoint, grad_out.contiguous(), idx, grad_features
        )

        return grad_features, None


gather_points = GatherFunction.apply


class BallQuery(torch.autograd.Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        r"""
        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query
        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        return sampling.ball_query(new_xyz, xyz, radius, nsample)

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class GroupingOperation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, idx):
        r"""
        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of features to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of features to group with
        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()

        ctx.for_backwards = (idx, N)

        return sampling.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward
        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, N = ctx.for_backwards

        grad_features = sampling.group_points_grad(grad_out.contiguous(), idx, N)

        return grad_features, None


grouping_operation = GroupingOperation.apply


class QueryAndGroup(torch.nn.Module):
    r"""
    Groups with a ball query of radius
    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True):
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz, new_xyz, features=None):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)
        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        # (B, npoint, k)
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        # (B, 3, N)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features


class FurthestPointSampling(torch.autograd.Function):

    @staticmethod
    def forward(ctx, xyz, npoint, seedIdx):
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set
        Returns
        -------
        torch.LongTensor
            (B, npoint) tensor containing the indices

        """
        B, N, _ = xyz.size()

        idx = torch.empty([B, npoint], dtype=torch.int32, device=xyz.device)
        temp = torch.full([B, N], 1e10, dtype=torch.float32, device=xyz.device)

        sampling.furthest_sampling(
            npoint, seedIdx, xyz, temp, idx
        )
        ctx.mark_non_differentiable(idx)
        return idx


__furthest_point_sample = FurthestPointSampling.apply


def furthest_point_sample(xyz, npoint, NCHW=True, seedIdx=0):
    """
    :param
        xyz (B, 3, N) or (B, N, 3)
        npoint a constant
    :return
        torch.LongTensor
            (B, npoint) tensor containing the indices
        torch.FloatTensor
            (B, npoint, 3) or (B, 3, npoint) point sets"""
    assert(xyz.dim() == 3), "input for furthest sampling must be a 3D-tensor, but xyz.size() is {}".format(xyz.size())
    # need transpose
    if NCHW:
        xyz = xyz.transpose(2, 1).contiguous()

    assert(xyz.size(2) == 3), "furthest sampling is implemented for 3D points"
    idx = __furthest_point_sample(xyz, npoint, seedIdx)
    sampled_pc = gather_points(xyz.transpose(2, 1).contiguous(), idx)
    if not NCHW:
        sampled_pc = sampled_pc.transpose(2, 1).contiguous()
    return idx, sampled_pc


class BatchSVDFunction(torch.autograd.Function):
    """
    batched svd implemented by https://github.com/KinglittleQ/torch-batch-svd
    """
    @staticmethod
    def forward(ctx, x):
        ctx.device = x.device
        if not torch.cuda.is_available():
            assert(RuntimeError), "BatchSVDFunction only runs on gpu"
        x = x.cuda()
        U, S, V = linalg.batch_svd_forward(x, True, 1e-7, 100)
        k = S.size(1)
        U = U[:, :, :k]
        V = V[:, :, :k]
        ctx.save_for_backward(x, U, S, V)
        U = U.to(ctx.device)
        S = S.to(ctx.device)
        V = V.to(ctx.device)
        return U, S, V

    @staticmethod
    def backward(ctx, grad_u, grad_s, grad_v):
        x, U, S, V = ctx.saved_variables

        grad_out = linalg.batch_svd_backward(
            [grad_u, grad_s, grad_v],
            x, True, True, U, S, V
        )

        return grad_out.to(device=ctx.device)


def batch_svd(x):
    """
    input:
        x --- shape of [B, M, N], k = min(M,N)
    return:
        U, S, V = batch_svd(x) where x = USV^T
        U [M, k]
        V [N, k]
        S [B, k] in decending order
    """
    assert(x.dim() == 3)
    return BatchSVDFunction.apply(x)


def batch_normals(points, base=None, nn_size=20, NCHW=True):
    """
    compute normals vectors for batched points [B, C, M]
    If base is given, compute the normals of points using the neighborhood in base
    The direction of normal could flip.
    
    Args:
        points: (B,C,M)
        base:   (B,C,N)
    Returns:
        normals: (B,C,M)
    """
    if base is None:
        base = points

    if NCHW:
        points = points.transpose(2, 1).contiguous()
        base = base.transpose(2, 1).contiguous()

    assert(nn_size < base.shape[1])
    batch_size, M, C = points.shape
    # B,M,k,C
    grouped_points, group_idx, _ = group_knn(nn_size, points, base, unique=True, NCHW=False)
    group_center = torch.mean(grouped_points, dim=2, keepdim=True)
    points = grouped_points - group_center
    allpoints = points.view(-1, nn_size, C).contiguous()
    # MB,C,k
    U, S, V = batch_svd(allpoints)
    # V is MBxCxC, last_u MBxC
    normals = V[:, :, -1]
    normals = normals.view(batch_size, M, C)
    if NCHW:
        normals = normals.transpose(1, 2)
    return normals

def normalize(tensor, dim=-1):
    """normalize tensor in specified dimension"""
    return torch.nn.functional.normalize(tensor, p=2, dim=dim, eps=1e-12, out=None)


class UniformLaplacian(torch.nn.Module):
    """
    uniform laplacian for mesh
    vertex B,N,D
    faces  B,F,L
    """
    def __init__(self, faces, nv):
        super().__init__()
        self.batch, self.nf, self.face_deg = faces.shape
        self.faces = faces
        self.nv = nv
    
        offset = torch.arange(0, self.batch).reshape(-1, 1, 1) * self.nv
        faces = self.faces + offset
        faces = faces.reshape(-1, 3)
        # offset index by batch
        row = faces[:, [i for i in range(self.face_deg)]].reshape(-1)
        col = faces[:, [i for i in range(1, self.face_deg)]+[0]].reshape(-1)
        indices = torch.stack([row, col], dim=0)
        L = torch.sparse_coo_tensor(indices, -torch.ones_like(col, dtype=torch.float), size=[self.nv*self.batch, self.nv*self.batch])
        L = L.t() + L
        self.Lii = -torch.sparse.sum(L, dim=[1]).to_dense()
        M = torch.sparse_coo_tensor(torch.arange(self.nv*self.batch).unsqueeze(0).expand(2, -1), self.Lii, size=(self.nv*self.batch, self.nv*self.batch))
        L = L + M
        # need to divide by diagonal, but can't do it in sparse

        self.register_buffer('L', L)

    def forward(self, verts):
        assert(verts.shape[0] == self.batch)
        assert(verts.shape[1] == self.nv)
        
        verts = verts.reshape(-1, verts.shape[-1])
        x = self.L.mm(verts)
        x = x / (self.Lii.unsqueeze(-1)+1e-12)
        x = x.reshape([self.batch, self.nv, -1])
        return x
        
#############
### cotangent laplacian from 3D-coded ###
#############
def convert_as(src, trg):
    src = src.type_as(trg)
    if src.is_cuda:
        src = src.cuda(device=trg.get_device())
    return src

class CotLaplacian(torch.autograd.Function):
    def __init__(self, faces):
        """
        Faces is B x F x 3, cuda torch Variabe.
        Reuse faces.
        """
        self.F_np = faces.data.cpu().numpy()
        self.F = faces.data
        self.L = None
    
    def forward(self, V):
        """
        If forward is explicitly called, V is still a Parameter or Variable
        But if called through __call__ it's a tensor.
        This assumes __call__ was used.
        
        Input:
           V: B x N x 3
           F: B x F x 3
        Outputs: L x B x N x 3
        
        Numpy also doesnt support sparse tensor, so stack along the batch
        """

        V_np = V.cpu().numpy()
        batchV = V_np.reshape(-1, 3)

        if self.L is None:
            print('Computing the Laplacian!')
            # Compute cotangents
            C = cotangent(V, self.F)
            C_np = C.cpu().numpy()
            batchC = C_np.reshape(-1, 3)
            # Adjust face indices to stack:
            offset = np.arange(0, V.size(0)).reshape(-1, 1, 1) * V.size(1)
            F_np = self.F_np + offset
            batchF = F_np.reshape(-1, 3)

            rows = batchF[:, [1, 2, 0]].reshape(-1) #1,2,0 i.e to vertex 2-3 associate cot(23)
            cols = batchF[:, [2, 0, 1]].reshape(-1) #2,0,1 This works because triangles are oriented ! (otherwise 23 could be associated to more than 1 cot))

            # Final size is BN x BN
            BN = batchV.shape[0]
            L = sparse.csr_matrix((batchC.reshape(-1), (rows, cols)), shape=(BN,BN))
            L = L + L.T
            # np.sum on sparse is type 'matrix', so convert to np.array
            M = sparse.diags(np.array(np.sum(L, 1)).reshape(-1), format='csr')
            L = L - M
            # remember this
            self.L = L
            # TODO The normalization by the size of the voronoi cell is missing.
            # import matplotlib.pylab as plt
            # plt.ion()
            # plt.clf()
            # plt.spy(L)
            # plt.show()
            # import ipdb; ipdb.set_trace()

        Lx = self.L.dot(batchV).reshape(V_np.shape)

        return convert_as(torch.Tensor(Lx), V)

    def backward(self, grad_out):
        """
        Just L'g = Lg
        Args:
           grad_out: B x N x 3
        Returns:
           grad_vertices: B x N x 3
        """
        g_o = grad_out.cpu().numpy()
        # Stack
        g_o = g_o.reshape(-1, 3)
        Lg = self.L.dot(g_o).reshape(grad_out.shape)

        return convert_as(torch.Tensor(Lg), grad_out)

#############
### cotangent laplacian from 3D-coded ###
#############

def cotangent(V, F):
    """
    Input:
      V: B x N x 3
      F: B x F x 3
    Outputs:
      C: B x F x 3 list of cotangents corresponding
        angles for triangles, columns correspond to edges 23,31,12
    B x F x 3 x 3
    """
    indices_repeat = torch.stack([F, F, F], dim=2)

    #v1 is the list of first triangles B*F*3, v2 second and v3 third
    v1 = torch.gather(V, 1, indices_repeat[:, :, :, 0].long())
    v2 = torch.gather(V, 1, indices_repeat[:, :, :, 1].long())
    v3 = torch.gather(V, 1, indices_repeat[:, :, :, 2].long())

    l1 = torch.sqrt(((v2 - v3)**2).sum(2)) #distance of edge 2-3 for every face B*F
    l2 = torch.sqrt(((v3 - v1)**2).sum(2))
    l3 = torch.sqrt(((v1 - v2)**2).sum(2))

    # semiperimieters
    sp = (l1 + l2 + l3) * 0.5

    # Heron's formula for area #FIXME why the *2 ? Heron formula is without *2 It's the 0.5 than appears in the (0.5(cotalphaij + cotbetaij))
    A = 2*torch.sqrt( sp * (sp-l1)*(sp-l2)*(sp-l3))

    # Theoreme d Al Kashi : c2 = a2 + b2 - 2ab cos(angle(ab))
    cot23 = (l2**2 + l3**2 - l1**2)
    cot31 = (l1**2 + l3**2 - l2**2)
    cot12 = (l1**2 + l2**2 - l3**2)

    # 2 in batch #proof page 98 http://www.cs.toronto.edu/~jacobson/images/alec-jacobson-thesis-2013-compressed.pdf
    C = torch.stack([cot23, cot31, cot12], 2) / torch.unsqueeze(A, 2) / 4

    return C


def mean_value_coordinates(points, polygon):
    """
    compute wachspress MVC of points wrt a polygon
    Args: 
        points: (B,D,N)
        polygon: (B,D,M)
    Returns:
        phi: (B,M,N)
    """
    D = polygon.shape[1]
    N = points.shape[-1]
    M = polygon.shape[-1]
    # (B,D,M,1) - (B,D,1,N) = (B,D,M,N)
    e = normalize(polygon.unsqueeze(3)-points.unsqueeze(2), dim=1)
    # B,M,N
    r = torch.norm(polygon.unsqueeze(3)-points.unsqueeze(2), p=2, dim=1)
    eplus = torch.cat([e[:,:,1:,:], e[:,:,:1,:]], dim=2)
    # (B,M,N)
    cos = dot_product(e, eplus, dim=1)
    sin = cross_product_2D(e, eplus, dim=1)
    tanhalf = sin / (1+cos+1e-12)
    tanhalf_minus  = torch.cat([tanhalf[:,-1:,:], tanhalf[:,:-1,:]], dim=1)
    w = (tanhalf_minus + tanhalf)/(r+1e-12)
    
    # special case: on boundary
    mask = ((torch.abs(sin) == 0) & (cos <= 0)| (cos == -1))
    mask_plus = torch.cat([mask[:,-1:,:], mask[:,:-1,:]], dim=1)
    mask_point = torch.any(mask, dim=1, keepdim=True)
    w = torch.where(mask_point, torch.zeros_like(w), w)
    pe = polygon - torch.cat([polygon[:,:,1:], polygon[:,:,:1]],dim=2)
    # (B,M,1)
    dL = torch.norm(pe, p=2, dim=1).unsqueeze(-1)
    w = torch.where(mask, 1-r/dL, w)
    w = torch.where(mask_plus, 1-r/dL, w)
    
    # special case: close to polygon vertex
    # (B,N)
    mask = torch.lt(r, 1e-10)
    mask_point = torch.any(mask, dim=1, keepdim=True)
    # set all weights of those points to zero
    w = torch.where(mask_point, torch.zeros_like(w), w)
    # set vertex weight of those points to 1
    w = torch.where(mask, torch.ones_like(w), w)
    
    # finally, normalize
    sumW = torch.sum(w, dim=1, keepdim=True)
    # sometimes sumw is 0?!
    torch.where(sumW==0, torch.ones_like(w), w)
    phi = w/sumW
    return phi


def dot_product(tensor1, tensor2, dim=-1):
    return torch.sum(tensor1*tensor2, dim=dim)

def cross_product_2D(tensor1, tensor2, dim=1):
    assert(tensor1.shape[dim] == tensor2.shape[dim] and tensor1.shape[dim] == 2)
    output = torch.narrow(tensor1, dim, 0, 1) * torch.narrow(tensor2, dim, 1, 1) - torch.narrow(tensor1, dim, 1, 1) * torch.narrow(tensor2, dim, 0, 1)
    return output.squeeze(dim)

def barycentric_coordinates(points, ref_points, triangulation):
    """
    compute barycentric coordinates of N points wrt M 2D triangles/3D tetrahedrons
    Args: 
        points: (B,D,N)
        ref_points: (B,D,M)
        triangulation: (B,D+1,L) L triangles (D=2) or L tetrahedra (D=3) indices
    Returns:
        epsilon: (B,N,L,D+1) weights for all triangles
        simplexMask: (B,N,L) mask the enclosing triangle
        pointMask: (B,N) mask the valid points
    """
    L = triangulation.shape[-1]
    D = ref_points.shape[1]
    N = points.shape[-1]
    M = ref_points.shape[-1]
    # find enclosing triangle
    ref_points = ref_points.transpose(2, 1).contiguous()
    triangulation = triangulation.transpose(2, 1).contiguous()
    # (B,L,D+1,D)
    simplexes = torch.gather(ref_points.unsqueeze(1).expand(-1, L, -1, -1), 2, triangulation.unsqueeze(-1).expand(-1, -1, -1, ref_points.shape[-1]))
    # (B,L,D,D+1)
    simplexes = simplexes.transpose(2,3)
    # (B,N,1,D) - (B,1,L,D) = (B,N,L,D)
    B = points.transpose(1,2).unsqueeze(2) - simplexes[:, :, :, -1].unsqueeze(1)
    # (B,L,D,D+1) - (B,L,D,1) = (B,L,D,D+1)
    T = (simplexes - simplexes[:,:,:,-1:])[:,:,:,:D]
    # (B,N,L,D,D)epsilon = (B,N,L,D,1), epsilon (B,N,L,D,1)
    epsilon, _ = torch.solve(B.unsqueeze(-1), T.unsqueeze(1).expand(-1, N, -1, -1, -1))
    epsilon_last = 1 - torch.sum(epsilon, dim=-2, keepdim=True)
    # (B,N,L,D+1)
    epsilon = torch.cat([epsilon, epsilon_last], dim=-2).squeeze(-1)
    # (B,N,L) enclosing triangle has positive coordinates
    simplexMask = torch.all((epsilon < 1) & (epsilon > 0), dim=-1)
    # cannot be enclosed in multiple simplexes
    assert(torch.all(torch.sum(simplexMask, dim=-1) <= 1)), "detected points enclosed in multiple triangles"
    # (B,N,L,D+1)
    epsilon = epsilon * simplexMask.unsqueeze(-1).to(dtype=epsilon.dtype)
    # (B,N)
    pointMask = torch.eq(torch.sum(simplexMask, dim=-1), 1)
    return epsilon, simplexMask, pointMask

def barycentric_map(points, epsilon, cage, triangulation, simplexMask):
    """
    Args:
        points: (B,D,N)
        epsilon: (B,N,L,D+1) weights
        cage: (B,D,M) cage points
        triangulation: (B,D+1,L) L triangles (D=2) or L tetrahedra (D=3) indices
        simplexMask: (B,N,L) mask for enclosing simplex
    Return:
        mapped: (B,D,N) mapped points, invalid points is mapped to zero
        epsilon_filtered: (B,N,D+1)
    """
    L = triangulation.shape[-1]
    N = points.shape[-1]
    cage_trans = cage.transpose(2, 1).contiguous()
    triangulation = triangulation.transpose(2,1).contiguous()
    # (B,L,D+1,D)
    simplexes = torch.gather(cage_trans.unsqueeze(1).expand(-1, L, -1, -1), 2, triangulation.unsqueeze(-1).expand(-1, -1, -1, cage_trans.shape[-1]))
    epsilon_filtered = simplexMask.to(dtype=epsilon.dtype).unsqueeze(-1)*epsilon
    # (B,N,L,D+1,D) * (B,N,L,1,1) * (B,N,L,D+1,1) = (B,N,L,D+1,D)
    mapped = simplexes.unsqueeze(1).expand(-1,N,-1,-1,-1)*epsilon_filtered.unsqueeze(-1)
    # (B,N,D)
    mapped = torch.sum(torch.sum(mapped, dim=2),dim=2)
    epsilon_filtered = torch.sum(epsilon_filtered, dim=2)
    return mapped.transpose(1,2), epsilon_filtered


if __name__ == '__main__':
    from ..utils import pc_utils
    cuda0 = torch.device('cuda:0')
    pc = pc_utils.read_ply("/home/ywang/Documents/points/point-upsampling/3PU/prepare_data/polygonmesh_base/build/data_PPU_output/training/112/angel4_aligned_2.ply")
    pc = pc[:, :3]
    print("{} input points".format(pc.shape[0]))
    pc_utils.save_ply(pc, "./input.ply", colors=None, normals=None)
    pc = torch.from_numpy(pc).requires_grad_().to(cuda0).unsqueeze(0)
    pc = pc.transpose(2, 1)

    # test furthest point
    idx, sampled_pc = furthest_point_sample(pc, 1250)
    output = sampled_pc.transpose(2, 1).cpu().squeeze()
    pc_utils.save_ply(output.detach(), "./output.ply", colors=None, normals=None)

    # test KNN
    knn_points, _, _ = group_knn(10, sampled_pc, pc, NCHW=True)  # B, C, M, K
    labels = torch.arange(0, knn_points.size(2)).unsqueeze_(
        0).unsqueeze_(0).unsqueeze_(-1)  # 1, 1, M, 1
    labels = labels.expand(knn_points.size(0), -1, -1,
                           knn_points.size(3))  # B, 1, M, K
    # B, C, P
    labels = torch.cat(torch.unbind(labels, dim=-1), dim=-1).squeeze().detach().cpu().numpy()
    knn_points = torch.cat(torch.unbind(knn_points, dim=-1),
                           dim=-1).transpose(2, 1).squeeze(0).detach().cpu().numpy()
    pc_utils.save_ply_property(knn_points, labels, "./knn_output.ply", cmap_name='jet')

    from torch.autograd import gradcheck
    # test = gradcheck(furthest_point_sample, [pc, 1250], eps=1e-6, atol=1e-4)
    # print(test)
    test = gradcheck(gather_points, [pc.to(
        dtype=torch.float64), idx], eps=1e-6, atol=1e-4)

    print(test)
