"""
code courtesy of
https://github.com/erikwijmans/Pointnet2_PyTorch
"""

import torch
import faiss
import numpy as np
from scipy import sparse

from .._ext import sampling
from .._ext import linalg
from ..utils.pytorch_utils import check_values, save_grad, saved_variables
from torch_scatter import scatter_add

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
    return faiss.cast_integer_to_float_ptr(
        x.storage().data_ptr() + x.storage_offset() * 4)


def __swig_ptr_from_LongTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.int64, 'dtype=%s' % x.dtype
    return faiss.cast_integer_to_long_ptr(
        x.storage().data_ptr() + x.storage_offset() * 8)


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
    Dptr = database.data_ptr()
    is_cuda = False
    if not (x.is_cuda or database.is_cuda):
        index = faiss.IndexFlatL2(database.size(-1))
    else:
        is_cuda = True
        index = faiss.GpuIndexFlatL2(GPU_RES, database.size(-1))  # dimension is 3
    index.add_c(database.size(0), faiss.cast_integer_to_float_ptr(Dptr))

    assert x.is_contiguous()
    n, d = x.size()
    assert d == index.d

    D = torch.empty((n, k), dtype=torch.float32, device=x.device)
    I = torch.empty((n, k), dtype=torch.int64, device=x.device)

    if is_cuda:
        torch.cuda.synchronize()
    xptr = __swig_ptr_from_FloatTensor(x)
    Iptr = __swig_ptr_from_LongTensor(I)
    Dptr = __swig_ptr_from_FloatTensor(D)
    index.search_c(n, xptr,
                   k, Dptr, Iptr)
    if is_cuda:
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
        (-1, query_trans.size(1), -1, -1))
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


gather_points = GatherFunction.apply  # type: ignore


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


ball_query = BallQuery.apply  # type: ignore


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


grouping_operation = GroupingOperation.apply  # type: ignore


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


__furthest_point_sample = FurthestPointSampling.apply  # type: ignore


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

def pointUniformLaplacian(points, knn_idx=None, nn_size=3):
    """
    Args:
        points: (B, N, 3)
        knn_idx: (B, N, K)
    Returns:
        laplacian: (B, N, 1)
    """
    batch_size, num_points, _ = points.shape
    if knn_idx is None:
        # find neighborhood, (B,N,K,3), (B,N,K)
        group_points, knn_idx, _ = faiss_knn(nn_size+1, points, points, NCHW=False)
        knn_idx = knn_idx[:, :, 1:]
        group_points = group_points[:, :, 1:, :]
    else:
        points_expanded = points.unsqueeze(dim=1).expand(
            (-1, num_points, -1, -1))
        # BxNxk -> BxNxNxC
        index_batch_expanded = knn_idx.unsqueeze(dim=-1).expand(
            (-1, -1, -1, points.size(-1)))
        # BxMxkxC
        group_points = torch.gather(points_expanded, 2, index_batch_expanded)

    lap = -torch.sum(group_points, dim=2)/knn_idx.shape[2] + points
    return lap, knn_idx


class UniformLaplacian(torch.nn.Module):
    """
    uniform laplacian for mesh
    vertex B,N,D
    faces  B,F,L
    """
    def __init__(self):
        super().__init__()
        self.L = None

    def computeLaplacian(self, V, F):
        batch, nv = V.shape[:2]
        V = V.reshape(-1, V.shape[-1])
        face_deg = F.shape[-1]
        offset = torch.arange(0, batch).reshape(-1, 1, 1) * nv
        faces = F + offset.to(device=F.device)
        faces = faces.reshape(-1, face_deg)
        # offset index by batch
        row = faces[:, [i for i in range(face_deg)]].reshape(-1)
        col = faces[:, [i for i in range(1, face_deg)]+[0]].reshape(-1)
        indices = torch.stack([row, col], dim=0)

        # (BN,BN)
        L = torch.sparse_coo_tensor(indices, -torch.ones_like(col, dtype=V.dtype, device=V.device), size=[nv*batch, nv*batch])
        L = L.t() + L
        # (BN)
        Lii = -torch.sparse.sum(L, dim=[1]).to_dense()
        M = torch.sparse_coo_tensor(torch.arange(nv*batch).unsqueeze(0).expand(2, -1), Lii, size=(nv*batch, nv*batch))
        L = L + M
        self.L = L
        self.Lii = Lii

    def forward(self, verts, faces):
        batch, nv = verts.shape[:2]
        assert(verts.shape[0] == batch)
        assert(verts.shape[1] == nv)

        if self.L is None:
            self.computeLaplacian(verts, faces)
        verts = verts.reshape(-1, verts.shape[-1])
        x = self.L.mm(verts)
        x = x / (self.Lii.unsqueeze(-1)+1e-12)
        x = x.reshape([batch, nv, -1])
        return x

#############
### cotangent laplacian from 3D-coded ###
#############
def convert_as(src, trg):
    src = src.type_as(trg)
    if src.is_cuda:
        src = src.cuda(device=trg.get_device())
    return src

class CotLaplacian(torch.nn.Module):
    def __init__(self):
        """
        Faces is B x F x 3, cuda torch Variabe.
        Reuse faces.
        """
        super().__init__()
        self.L = None

    def computeLaplacian(self, V, F):
        print('Computing the Laplacian!')
        F_np = F.data.cpu().numpy()
        F = F.data
        B,N,_ = V.shape
        # Compute cotangents
        C = cotangent(V, F)
        C_np = C.cpu().numpy()
        batchC = C_np.reshape(-1, 3)
        # Adjust face indices to stack:
        offset = np.arange(0, V.size(0)).reshape(-1, 1, 1) * V.size(1)
        F_np = F_np + offset
        batchF = F_np.reshape(-1, 3)

        rows = batchF[:, [1, 2, 0]].reshape(-1) #1,2,0 i.e to vertex 2-3 associate cot(23)
        cols = batchF[:, [2, 0, 1]].reshape(-1) #2,0,1 This works because triangles are oriented ! (otherwise 23 could be associated to more than 1 cot))

        # Final size is BN x BN
        BN = B*N
        L = sparse.csr_matrix((batchC.reshape(-1), (rows, cols)), shape=(BN,BN))
        L = L + L.T
        # np.sum on sparse is type 'matrix', so convert to np.array
        M = sparse.diags(np.array(np.sum(L, 1)).reshape(-1), format='csr')
        L = L - M
        # remember this
        self.L = L

    def forward(self, V, F=None):
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
        if self.L is None:
            assert(F is not None)
            self.computeLaplacian(V, F)

        Lx = _cotLx(V, self.L)
        return Lx


class _CotLaplacianBatchLx(torch.autograd.Function):
    @staticmethod
    def forward(ctx, V, L):
        """V: (B,N,3), L: numpy sparse matrix (BN, 3)"""
        ctx.L = L
        batchV = V.reshape(-1, 3).cpu().numpy()
        Lx = L.dot(batchV)
        return convert_as(torch.Tensor(Lx), V)

    @staticmethod
    def backward(ctx, grad_out):
        """
        Just L'g = Lg
        Args:
           grad_out: B x N x 3
        Returns:
           grad_vertices: B x N x 3
        """
        L = ctx.L
        sh = grad_out.shape
        g_o = grad_out.cpu().numpy()
        # Stack
        g_o = g_o.reshape(-1, 3)
        Lg = L.dot(g_o).reshape(sh)
        return convert_as(torch.Tensor(Lg), grad_out), None

_cotLx = _CotLaplacianBatchLx.apply  # type: ignore

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
    indices_repeat = torch.stack([F, F, F], dim=2).to(device=V.device)

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


def mean_value_coordinates_3D(query, vertices, faces):
    """
    Tao Ju et.al. MVC for 3D triangle meshes
    params:
        query    (B,P,3)
        vertices (B,N,3)
        faces    (B,F,3)
    return:
        wj       (B,P,N)
    """
    PI = 3.1415927
    B, F, _ = faces.shape
    _, P, _ = query.shape
    _, N, _ = vertices.shape
    # u_i = p_i - x (B,P,N,3)
    uj = vertices.unsqueeze(1) - query.unsqueeze(2)
    # \|u_i\| (B,P,N,1)
    dj = torch.norm(uj, dim=-1, p=2, keepdim=True)
    # TODO: if lu < epsilon, wj to delta_ij (B,P,N,3)
    uj = normalize(uj, dim=-1)
    # gather triangle B,P,F,3,3
    triangle_points = torch.gather(uj.unsqueeze(2).expand(-1,-1,F,-1,-1),
                                   3,
                                   faces.unsqueeze(1).unsqueeze(-1).expand(-1,P,-1,-1,3))
    # li = \|u_{i+1}-u_{i-1}\| (B,P,F,3)
    li = torch.norm(triangle_points[:,:,:,[1, 2, 0],:] - triangle_points[:, :, :,[2, 0, 1],:], dim=-1, p=2)
    eps = 2e-5
    li = torch.where(li>=2, li-(li.detach()-(2-eps)), li)
    li = torch.where(li<=-2, li-(li.detach()+(2-eps)), li)
    # asin(x) is inf at +/-1
    # θi =  2arcsin[li/2] (B,P,F,3)
    theta_i = 2*torch.asin(li/2)
    assert(check_values(theta_i))
    # B,P,F,1
    h = torch.sum(theta_i, dim=-1, keepdim=True)/2
    # TODO if π −h < ε, x lies on t, use 2D barycentric coordinates
    # wi← sin[θi]d{i−1}d{i+1}
    # (B,P,F,3) ci ← (2sin[h]sin[h−θi])/(sin[θ_{i+1}]sin[θ_{i−1}])−1
    ci = 2*torch.sin(h)*torch.sin(h-theta_i)/(torch.sin(theta_i[:,:,:,[1, 2, 0]])*torch.sin(theta_i[:,:,:,[2, 0, 1]]))-1

    # NOTE: because of floating point ci can be slightly larger than 1, causing problem with sqrt(1-ci^2)
    # NOTE: sqrt(x)' is nan for x=0, hence use eps
    eps = 1e-5
    ci = torch.where(ci>=1, ci-(ci.detach()-(1-eps)), ci)
    ci = torch.where(ci<=-1, ci-(ci.detach()+(1-eps)), ci)
    # si← sign[det[u1,u2,u3]]sqrt(1-ci^2)
    # (B,P,F)*(B,P,F,3)
    si = torch.sign(torch.det(triangle_points)).unsqueeze(-1)*torch.sqrt(1-ci**2)  # sqrt gradient nan for 0
    assert(check_values(si))
    # TODO if ∃i,|si| ≤ ε, set wi to 0. coplaner with T but outside
    # (B,P,F,3)
    di = torch.gather(dj.unsqueeze(2).squeeze(-1).expand(-1,-1,F,-1), 3,
                      faces.unsqueeze(1).expand(-1,P,-1,-1))
    assert(check_values(di))
    # if si.requires_grad:
    #     saved_variables["di"] = di.detach()
    #     saved_variables["si"] = si.detach()
    #     saved_variables["ci"] = ci.detach()
    #     saved_variables["thetai"] = theta_i.detach()
    #     saved_variables["li"] = li.detach()
    #     li.register_hook(save_grad("dli"))
    #     theta_i.register_hook(save_grad("dtheta"))
    #     ci.register_hook(save_grad("dci"))
    #     si.register_hook(save_grad("dsi"))
    #     di.register_hook(save_grad("ddi"))
    # wi← (θi −c[i+1]θ[i−1] −c[i−1]θ[i+1])/(disin[θi+1]s[i−1])
    # B,P,F,3
    wi = (theta_i-ci[:,:,:,[1,2,0]]*theta_i[:,:,:,[2,0,1]]-ci[:,:,:,[2,0,1]]*theta_i[:,:,:,[1,2,0]])/(di*torch.sin(theta_i[:,:,:,[1,2,0]])*si[:,:,:,[2,0,1]])

    # ignore coplaner outside triangle
    wi = torch.where(torch.any(torch.abs(si) <= 1e-5, keepdim=True, dim=-1), torch.zeros_like(wi), wi)

    # inside triangle
    inside_triangle = (PI-h).squeeze(-1)<1e-3
    # set all F for this P to zero
    wi = torch.where(torch.any(inside_triangle, dim=-1, keepdim=True).unsqueeze(-1), torch.zeros_like(wi), wi)
    wi = torch.where(inside_triangle.unsqueeze(-1), torch.sin(theta_i)*di[:,:,:,[2,0,1]]*di[:,:,:,[1,2,0]], wi)

    # per face -> vertex (B,P,F*3) -> (B,P,N)
    wj = scatter_add(wi.view(B,P,-1), faces.unsqueeze(1).expand(-1,P,-1,-1).reshape(B,P,-1),2)

    # close to vertex (B,P,N)
    close_to_point = dj.squeeze(-1) < 1e-8
    # set all F for this P to zero
    wj = torch.where(torch.any(close_to_point, dim=-1, keepdim=True), torch.zeros_like(wj), wj)
    wj = torch.where(close_to_point, torch.ones_like(wj), wj)

    # (B,P,1)
    sumWj = torch.sum(wj, dim=-1, keepdim=True)
    sumWj = torch.where(sumWj==0, torch.ones_like(sumWj), sumWj)

    wj = wj / sumWj
    # if wj.requires_grad:
    #     saved_variables["dwi"] = wi.detach()
    #     wi.register_hook(save_grad("dwi"))
    #     wj.register_hook(save_grad("dwj"))
    return wj


def mean_value_coordinates(points, polygon):
    """
    compute wachspress MVC of points wrt a polygon
    https://www.mn.uio.no/math/english/people/aca/michaelf/papers/barycentric.pdf
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
    si = polygon.unsqueeze(3)-points.unsqueeze(2)
    # ei = normalize(s, dim=1)
    # B,M,N
    ri = torch.norm(si, p=2, dim=1)
    rip = torch.cat([ri[:,1:,:], ri[:,:1,:]], dim=1)
    sip = torch.cat([si[:,:,1:,:], si[:,:,:1,:]], dim=2)
    # sip = si[:,:,[i%M+1 for i in range(M)], :]
    # (B,M,N)
    # cos = dot_product(e, eplus, dim=1)
    # sin = cross_product_2D(e, eplus, dim=1)
    # (r_i*r_{i+1}-D_i)/A_i
    # D_i <e_i, e_{i+1}>
    # A_i det(e_i, e_{i+1})/2
    Ai = cross_product_2D(si, sip, dim=1)/2
    Aim = torch.cat([Ai[:,-1:,:], Ai[:,:-1,:]], dim=1)
    Di = dot_product(si, sip, dim=1)
    # Dim = torch.cat([Di[:,-1:,:], Di[:,:-1,:]], dim=1)
    # tanhalf = sin / (1+cos+1e-12)
    # w = torch.where(Ai!=0, (rip - Di/ri)/Ai, torch.zeros_like(Ai))+ torch.where(Aim!=0, (rim-Dim/ri)/Aim)
    tanhalf = torch.where(torch.abs(Ai) > 1e-5, (rip*ri-Di)/(Ai+1e-10), torch.zeros_like(Ai))
    tanhalf_minus  = torch.cat([tanhalf[:,-1:,:], tanhalf[:,:-1,:]], dim=1)
    w = (tanhalf_minus + tanhalf)/(ri+1e-10)

    # special case: on boundary
    # mask = ((torch.abs(sin) == 0) & (cos <= 0)| (cos == -1))
    mask = (torch.abs(Ai) <= 1e-5) & (Di < 0.0)
    mask_plus = torch.cat([mask[:,-1:,:], mask[:,:-1,:]], dim=1)
    mask_point = torch.any(mask, dim=1, keepdim=True)
    w = torch.where(mask_point, torch.zeros_like(w), w)
    pe = polygon - torch.cat([polygon[:,:,1:], polygon[:,:,:1]],dim=2)
    # (B,M,1)
    dL = torch.norm(pe, p=2, dim=1).unsqueeze(-1)
    w = torch.where(mask, 1-ri/(dL+1e-10), w)
    # w = torch.where(mask_plus, 1-ri/dL, w)
    w = torch.where(mask_plus, 1-torch.sum(w, dim=1, keepdim=True), w)
    # special case: close to polygon vertex
    # (B,N)
    mask = torch.lt(ri, 1e-8)
    # if an cage edge is very very short, can happen that this is true for both vertices
    mask_point = torch.any(mask, dim=1, keepdim=True)
    # set all weights of those points to zero
    w = torch.where(mask_point, torch.zeros_like(w), w)
    # set vertex weight of those points to 1
    w = torch.where(mask, torch.ones_like(w), w)

    # finally, normalize
    sumW = torch.sum(w, dim=1, keepdim=True)
    # sometimes sumw is 0?!
    if torch.nonzero(sumW==0).numel() > 0:
        sumW = torch.where(sumW==0, torch.ones_like(w), w)
    phi = w/sumW
    return phi


def dot_product(tensor1, tensor2, dim=-1):
    return torch.sum(tensor1*tensor2, dim=dim)

def cross_product_2D(tensor1, tensor2, dim=1):
    assert(tensor1.shape[dim] == tensor2.shape[dim] and tensor1.shape[dim] == 2)
    output = torch.narrow(tensor1, dim, 0, 1) * torch.narrow(tensor2, dim, 1, 1) - torch.narrow(tensor1, dim, 1, 1) * torch.narrow(tensor2, dim, 0, 1)
    return output.squeeze(dim)



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
    test = gradcheck(gather_points, [pc.to(  # type: ignore
        dtype=torch.float64), idx], eps=1e-6, atol=1e-4)

    print(test)
