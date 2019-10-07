import torch
import numpy as np
from .._ext import losses
from .operations import faiss_knn
from . import geo_operations as geo_op
from matplotlib import cm


class UniformLaplacianSmoothnessLoss(torch.nn.Module):
    """
    Encourages minimal mean curvature shapes.
    """
    def __init__(self, num_point, faces, metric):
        super().__init__()
        self.laplacian = geo_op.UniformLaplacian()
        self.metric = metric
        self.faces = faces

    def __call__(self, vert, vert_ref=None):
        lap = self.laplacian(vert, self.faces)
        curve = torch.norm(lap, p=2, dim=-1)
        if vert_ref is not None:
            lap_ref = self.laplacian(vert, self.faces)
            curve_gt = torch.norm(lap_ref, p=2, dim=-1)
            loss = self.metric(curve, curve_gt)
        else:
            loss = curve
        return loss

class MeshLaplacianLoss(torch.nn.Module):
    """
    compare laplacian of two meshes with the same connectivity assuming known correspondence
    metric: an instance of a module e.g. L1Loss
    use_cot: cot laplacian is used instead of uniformlaplacian
    consistent_topology: assume face matrix is the same during the entire use
    """
    def __init__(self, metric, use_cot=False, use_norm=False, consistent_topology=False):
        super().__init__()
        if use_cot:
            self.laplacian = geo_op.CotLaplacian()
        else:
            self.laplacian = geo_op.UniformLaplacian()

        self.use_norm = use_norm
        self.consistent_topology = consistent_topology
        self.metric = metric

    def forward(self, vert1, vert2=None, face=None):
        if not self.consistent_topology:
            self.laplacian.L = None

        lap1 = self.laplacian(vert1, face)
        if self.use_norm:
            lap1 = torch.norm(lap1, dim=-1, p=2)
        if vert2 is not None:
            lap2 = self.laplacian(vert2, face)
            if self.use_norm:
                lap2 = torch.norm(lap2, dim=-1, p=2)
            return self.metric(lap1,lap2)
        else:
            return lap1.mean()

class PointLaplacianLoss(torch.nn.Module):
    """
    compare uniform laplacian of two point clouds assuming known correspondence
    metric: an instance of a module e.g. L1Loss
    """
    def __init__(self, nn_size, metric, use_norm=False):
        super().__init__()
        self.metric = metric
        self.nn_size = nn_size
        self.use_norm = use_norm

    def forward(self, point1, point2):
        """
        point1: (B,N,D) ref points (where connectivity is computed)
        point2: (B,N,D) pred points, uses connectivity of point1
        """
        lap1, knn_idx = geo_op.pointUniformLaplacian(point1, nn_size=self.nn_size)
        lap2, _ = geo_op.pointUniformLaplacian(point2, knn_idx=knn_idx)
        if self.use_norm:
            lap1 = torch.norm(lap1, dim=-1, p=2)
            lap2 = torch.norm(lap2, dim=-1, p=2)
        return self.metric(lap1, lap2)


class PointEdgeLengthLoss(torch.nn.Module):
    """
    Penalize edge length change
    metric: an instance of a module e.g. L1Loss
    """
    def __init__(self, nn_size, metric):
        super().__init__()
        self.metric = metric
        self.nn_size = nn_size

    def forward(self, points_ref, points):
        """
        point1: (B,N,D) ref points (where connectivity is computed)
        point2: (B,N,D) pred points, uses connectivity of point1
        """
        # find neighborhood, (B,N,K,3), (B,N,K)
        group_points, knn_idx, _ = faiss_knn(self.nn_size+1, points_ref, points_ref, NCHW=False)
        knn_idx = knn_idx[:, :, 1:]
        group_points= group_points[:,:,1:,:]
        dist_ref = torch.norm(group_points - points_ref.unsqueeze(2), dim=-1, p=2)
        # dist_ref = torch.sqrt(dist_ref)
        # B,N,K,D
        group_points = torch.gather(points.unsqueeze(1).expand(-1, knn_idx.shape[1], -1, -1), 2, knn_idx.unsqueeze(-1).expand(-1, -1, -1, points.shape[-1]))
        dist = torch.norm(group_points - points.unsqueeze(2), dim=-1, p=2)
        # print(group_points, group_points2)
        return self.metric(dist_ref, dist)


class PointStretchLoss(torch.nn.Module):
    """
    penalize stretch only max(d/d_ref-1, 0)
    """
    def __init__(self, nn_size, reduction="mean"):
        super().__init__()
        self.nn_size = nn_size
        self.reduction = reduction

    def forward(self, points_ref, points):
        """
        point1: (B,N,D) ref points (where connectivity is computed)
        point2: (B,N,D) pred points, uses connectivity of point1
        """
        # find neighborhood, (B,N,K,3), (B,N,K), (B,N,K)
        group_points_ref, knn_idx, _ = faiss_knn(self.nn_size+1, points_ref, points_ref, NCHW=False)
        knn_idx = knn_idx[:, :, 1:]
        group_points_ref = group_points_ref[:,:,1:,:]
        dist_ref = torch.norm(group_points_ref - points_ref.unsqueeze(2), dim=-1, p=2)
        group_points = torch.gather(points.unsqueeze(1).expand(-1, knn_idx.shape[1], -1, -1), 2, knn_idx.unsqueeze(-1).expand(-1, -1, -1, points.shape[-1]))
        dist = torch.norm(group_points - points.unsqueeze(2), dim=-1, p=2)
        stretch = torch.max(dist/dist_ref-1, torch.zeros_like(dist))
        if self.reduction == "mean":
            return torch.mean(stretch)
        elif self.reduction == "sum":
            return torch.mean(torch.sum(stretch, dim=-1))
        elif self.reduction == "none":
            return stretch
        elif self.reduction == "max":
            return torch.mean(torch.max(stretch, dim=-1)[0])
        else:
            raise NotImplementedError


class MeshEdgeLengthLoss(torch.nn.Module):
    """
    Penalize large edge deformation for meshes of the same topology (assuming correspondance)
    faces (B,F,L)
    """
    def __init__(self, metric, consistent_topology=False):
        super().__init__()
        self.metric = metric
        self.E = None
        self.consistent_topology = consistent_topology

    @staticmethod
    def getEV(faces, n_vertices):
        """return a list of B (E, 2) int64 tensor"""
        B, F, _ = faces.shape
        EV = []
        for b in range(B):
            EV.append(geo_op.edge_vertex_indices(faces[b]))
        return EV

    def forward(self, vert1, vert2, face=None):
        """
        vert1: (B, N, 3)
        vert2: (B, N, 3)
        faces:  (B, F, L)
        """
        assert(vert1.shape == vert2.shape)
        B, P, _ = vert1.shape
        F = face.shape[1]
        if (not self.consistent_topology) or (self.E is None):
            assert(face is not None), "Face is required"
            self.E = self.getEV(face, P)

        # (B, E, 2, 3)
        loss = []
        for b in range(B):
            edge_length1 = geo_op.get_edge_lengths(vert1[b], self.E[b])
            edge_length1 = geo_op.get_edge_lengths(vert2[b], self.E[b])
            loss.append(self.metric(edge_length1, edge_length2))

        loss = torch.stack(loss, dim=0)
        loss = torch.mean(loss)

        return loss


class MeshStretchLoss(torch.nn.Module):
    """
    Penalize increase of edge length max(len2/len1-1, 0)
    assuming the same triangulation
    ======
    Input:
        vert1 reference vertices (B,N,3)
        vert2 vertices (B,N,3)
        faces face vertex indices (same between vert1 and vert2)
    """
    def __init__(self, reduction="mean", consistent_topology=False):
        self.E = None
        self.reduction = reduction
        self.consistent_topology = consistent_topology
        super().__init__()


    @staticmethod
    def getEV(faces, n_vertices):
        """return a list of B (E, 2) int64 tensor"""
        B, F, _ = faces.shape
        EV = []
        for b in range(B):
            EV.append(geo_op.edge_vertex_indices(faces[b]))
        return EV

    def forward(self, vert1, vert2, face=None):
        assert(vert1.shape == vert2.shape)
        B, P, _ = vert1.shape
        F = face.shape[1]
        if (not self.consistent_topology) or self.E is None:
            assert(face is not None), "Face is required"
            self.E = self.getEV(face, P)

        # (B, E, 2, 3)
        loss = []
        for b in range(B):
            edge_length1 = geo_op.get_edge_lengths(vert1[b], self.E[b])
            edge_length2 = geo_op.get_edge_lengths(vert2[b], self.E[b])

            stretch = torch.max(edge_length2/edge_length1-1, torch.zeros_like(edge_length1))
            if self.reduction in ("mean", "none"):
                loss.append(stretch.mean())
            elif self.reduction == "max":
                loss.append(stretch.max())
            elif self.reduction == "sum":
                loss.append(stretch.sum())
            else:
                raise NotImplementedError

        loss = torch.stack(loss, dim=0)
        if self.reduction != "none":
            loss = loss.mean()

        return loss


class SimpleMeshRepulsionLoss(MeshStretchLoss):
    """
    Penalize very short mesh edges
    """
    def __init__(self, threshold, reduction="mean", consistent_topology=False):
        super().__init__(reduction=reduction, consistent_topology=consistent_topology)
        self.threshold2 = threshold*threshold

    def forward(self, verts1, face=None):
        """
        verts1: (B, N, 3)
        faces:  (B, F, L)
        """
        B, P, _ = verts1.shape
        F = face.shape[1]
        if (not self.consistent_topology) or self.E is None:
            assert(face is not None), "Face is required"
            self.E = self.getEV(face, P)

        # (B, E, 2, 3)
        loss = []
        for b in range(B):
            edge_length1 = geo_op.get_edge_lengths(verts[b], self.E[b])
            tmp = 1/(edge_length1+1e-6)
            tmp = torch.where(edge_length1 < self.threshold2, tmp, torch.zeros_like(tmp))
            if self.reduction in ("mean", "none"):
                tmp = tmp.mean()
            elif self.reduction == "max":
                tmp = tmp.max()
            elif self.reduction == "sum":
                tmp = tmp.sum()
            else:
                raise NotImplementedError
            loss.append(tmp)

        loss = torch.stack(loss, dim=0)
        if self.reduction != "none":
            loss = loss.mean()

        return loss


class MeshDihedralAngleLoss(torch.nn.Module):
    """
    vert1           (B,N,3)
    vert2           (B,N,3)
    edge_points     List(torch.Tensor(E, 4))
    """
    def __init__(self, metric: torch.nn.Module, consistent_topology: bool=False):
        super().__init__()
        self.metric = metric
        self.consistent_topology = consistent_topology
        # List(Ex4)
        self.edge_points = None

    def forward(self, vert1, vert2, edge_points):
        B = vert1.shape[0]
        loss = []
        for b in range(B):
            angles1 = geo_op.dihedral_angle(vert1[b], edge_points[b])
            angles2 = geo_op.dihedral_angle(vert2[b], edge_points[b])
            tmp = self.metric(angles1, angles2)
            loss.append(tmp)

        loss = torch.stack(loss, dim=0)
        if self.reduction != "none":
            loss = loss.mean()

        return loss

class SmapeLoss(torch.nn.Module):
    """
    relative L1 norm
    http://drz.disneyresearch.com/~jnovak/publications/KPAL/KPAL.pdf eq(2)
    """
    def __init__(self, epsilon=1e-8):
        super(SmapeLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, y):
        """
        x pred  (N,3)
        y label (N,3)
        """
        return torch.mean(torch.abs(x-y)/(torch.abs(x)+torch.abs(y)+self.epsilon))

class NormalLoss(torch.nn.Module):
    def __init__(self, metric, nn_size=10):
        self.nn_size = nn_size
        self.metric = metric

    def forward(self, pred, gt):
        pred_normals = geo_op.batch_normals(pred, nn_size=10, NCHW=True)
        gt_normals = geo_op.batch_normals(gt, nn_size=10, NCHW=True)
        # compare the normal with the closest point
        return self.metric(pred_normals, gt_normals)


class SimplePointRepulsionLoss(torch.nn.Module):
    """
    Penalize point-to-point distance which is smaller than a threshold
    params:
        points:  (B,N,C)
        nn_size: neighborhood size
    """
    def __init__(self, nn_size, radius, reduction="mean"):
        super().__init__()
        self.nn_size = nn_size
        self.reduction = reduction
        self.radius2 = radius*radius

    def forward(self, points, knn_idx=None):
        batchSize, PN, _ = points.shape
        if knn_idx is None:
            knn_points, knn_idx, distance2 = faiss_knn(self.nn_size+1, points, points, NCHW=False)
            knn_points = knn_points[:, :, 1:, :].contiguous().detach()
            knn_idx = knn_idx[:, :, 1:].contiguous()
        else:
            knn_points = torch.gather(points.unsqueeze(1).expand(-1, PN, -1, -1), 2, knn_idx.unsqueeze(-1).expand(-1, -1, -1, points.shape[-1]))

        knn_v = knn_points - points.unsqueeze(dim=2)
        distance2 = torch.sum(knn_v * knn_v, dim=-1)
        loss = 1/torch.sqrt(distance2+1e-4)
        loss = torch.where(distance2 < self.radius2, loss, torch.zeros_like(loss))
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "max":
            return torch.mean(torch.max(loss, dim=-1)[0])
        elif self.reduction == "sum":
            return loss.mean(torch.sum(loss, dim=-1))
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError
        return loss


class NmDistanceFunction(torch.autograd.Function):
    """3D point set to 3D point set distance"""
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        assert(xyz1.is_contiguous())
        assert(xyz2.is_contiguous())
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        assert(xyz1.dtype==xyz2.dtype)
        dist1 = torch.zeros(batchsize, n, dtype=xyz1.dtype)
        dist2 = torch.zeros(batchsize, m, dtype=xyz1.dtype)

        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)

        dist1 = dist1.cuda()
        dist2 = dist2.cuda()
        idx1 = idx1.cuda()
        idx2 = idx2.cuda()
        losses.nmdistance_forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        ctx.mark_non_differentiable(idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, gradNone1, gradNone2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros_like(xyz1)
        gradxyz2 = torch.zeros_like(xyz2)

        gradxyz1 = gradxyz1.cuda()
        gradxyz2 = gradxyz2.cuda()
        losses.nmdistance_backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        return gradxyz1, gradxyz2


nndistance = NmDistanceFunction.apply  # type: ignore


class LabeledNmdistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2, label1, label2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        assert(xyz1.dtype==xyz2.dtype)
        label1 = label1.to(dtype=xyz1.dtype)
        label2 = label2.to(dtype=xyz1.dtype)
        dist1 = torch.zeros(batchsize, n, dtype=xyz1.dtype)
        dist2 = torch.zeros(batchsize, m, dtype=xyz1.dtype)

        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)

        dist1 = dist1.cuda()
        dist2 = dist2.cuda()
        idx1 = idx1.cuda()
        idx2 = idx2.cuda()
        losses.labeled_nmdistance_forward(xyz1, xyz2, label1, label2,  dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        ctx.mark_non_differentiable(idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, gradNone1, gradNone2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros_like(xyz1)
        gradxyz2 = torch.zeros_like(xyz2)

        gradxyz1 = gradxyz1.cuda()
        gradxyz2 = gradxyz2.cuda()
        losses.nmdistance_backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        return gradxyz1, gradxyz2, None, None

labeled_nndistance = LabeledNmdistanceFunction.apply

class ChamferLoss(torch.nn.Module):
    """
    chamfer loss. bidirectional nearest neighbor distance of two point sets.
    mean_{xyz1}(nd_{1to2})+\beta*max_{xyz1}(nd_{1to2})+(\gamma+\delta|xyz1|)mean_{xyz2}(nd_{2to1})
    Args:
        threshold (float): distance beyond threshold*average_distance not be considered
        percentage (float): consider a percentage of inner points
        max (bool): use hausdorf, i.e. use max instead of mean
    """

    def __init__(self, threshold=None, beta=1.0, gamma=1, delta=0, percentage=1.0):
        super(ChamferLoss, self).__init__()
        # only consider distance smaller than threshold*mean(distance) (remove outlier)
        self.__threshold = threshold
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.percentage = percentage

    def set_threshold(self, value):
        self.__threshold = value

    def unset_threshold(self):
        self.__threshold = None

    def forward(self, pred, gt, pred_mask=None, gt_mask=None):
        """
        chamfer disntance between (B,N,3) and (B,M,3) points
        if pred_mask and gt_mask is given, then set unmasked loss to zero
        """
        assert(pred.dim() == 3 and gt.dim() == 3), \
            "input for ChamferLoss must be a 3D-tensor, but pred.size() is {} gt.size() is {}".format(pred.size(), gt.size())
        if pred_mask is not None:
            assert(pred.shape[:2] == pred_mask.shape), "Mask and input must have the same shape"
        if gt_mask is not None:
            assert(gt.shape[:2] == gt_mask.shape), "Mask and input must have the same shape"

        assert(pred.size(2) == gt.size(2)), "input and output must be (B,N,D) and (B,M,D)"
        assert(pred.is_contiguous())
        assert(gt.is_contiguous())
        B,N,_ = pred.shape
        B,M,_ = gt.shape

        # discard border points
        if self.percentage < 1.0:
            pred_center = torch.mean(pred, dim=1, keepdim=True)
            pred, _, _ = faiss_knn(int(self.percentage * N), pred_center, pred, unique=False, NCHW=False)
            pred = torch.squeeze(pred, dim=1)
            # # BxN
            # dist_sqr = torch.sum((pred - pred_center)**2, dim=-1)
            # # Bx1
            # dist_sqrm = torch.max(dist_sqr, dim=1, keepdim=True)
            # weight = torch.exp(-dist_sqr / 1.5 * dist_sqrm)
            # weight = weight / torch.max(weight)
            # pred2gt = pred2gt * weight

            gt_center = torch.mean(gt, dim=1, keepdim=True)
            gt, _, _ = faiss_knn(int(self.percentage * M), gt_center, gt, unique=False, NCHW=False)
            gt = torch.squeeze(gt, dim=1)
            # # BxN
            # dist_sqr = torch.sum((label - label_center)**2, dim=-1)
            # # Bx1
            # dist_sqrm = torch.max(dist_sqr, dim=1, keepdim=True)
            # weight = torch.exp(-dist_sqr / 1.5 * dist_sqrm)
            # weight = weight / torch.max(weight)
            # gt2pred = gt2pred * weight
        if pred_mask is not None:
            # (B,N)
            pred = torch.where(pred_mask.unsqueeze(-1), pred, torch.full(pred.shape, float("Inf"), device=pred.device, dtype=pred.dtype))
        if gt_mask is not None:
            gt = torch.where(gt_mask.unsqueeze(-1), gt, torch.full(gt.shape, float("Inf"), device=gt.device, dtype=gt.dtype))

        pred2gt, gt2pred, _, _ = nndistance(pred, gt)

        if self.__threshold is not None:
            threshold = self.__threshold
            forward_threshold = torch.mean(
                pred2gt, dim=1, keepdim=True) * threshold
            backward_threshold = torch.mean(
                gt2pred, dim=1, keepdim=True) * threshold
            # only care about distance within threshold (ignore strong outliers)
            pred2gt = torch.where(
                pred2gt < forward_threshold, pred2gt, torch.zeros_like(pred2gt))
            gt2pred = torch.where(
                gt2pred < backward_threshold, gt2pred, torch.zeros_like(gt2pred))

        if pred_mask is not None:
            pred2gt = torch.where(pred_mask, pred2gt, torch.zeros_like(pred2gt))
        if gt_mask is not None:
            gt2pred = torch.where(gt_mask, gt2pred, torch.zeros_like(gt2pred))

        # pred2gt is for each element in gt, the closest distance to this element
        loss = torch.mean(pred2gt, dim=-1) + torch.mean(gt2pred, dim=-1)*(self.delta*N+self.gamma)+torch.max(pred2gt, dim=-1)[0]*self.beta
        loss = torch.mean(loss)
        return loss


if __name__ == '__main__':
    from .geo_operations import normalize_point_batch
    from ..utils.geometry_utils import array_to_mesh, write_trimesh, generatePolygon
    from ..utils.pytorch_utils import saved_variables, save_grad
    from torch.autograd import gradcheck
    # pc1 = torch.randn([2, 600, 2], dtype=torch.float32,
    #                   requires_grad=True).cuda()
    # pc2 = torch.randn([2, 600, 2], dtype=torch.float32,
    #                   requires_grad=True).cuda()
    # chamfer = ChamferLoss()
    # edgeLoss = PointEdgeLengthLoss(nn_size=3, metric=torch.nn.MSELoss())
    # edgeShouldBeZero = edgeLoss(pc1, pc1)
    # print(edgeShouldBeZero)
    # assert(torch.all(edgeShouldBeZero == 0))

    # shape_laplacian = MeshLaplacianLoss(torch.nn.MSELoss(reduction="none"), use_cot=False, use_norm=True, consistent_topology=True)
    # faces = torch.from_numpy(np.loadtxt("/home/yifan/Data/Chicken-3D-movie/chicken_T.txt", dtype=np.int64)).unsqueeze(0)
    # sequence = torch.from_numpy(np.load("/home/yifan/Data/Chicken-3D-movie/vertices.npy")).to(dtype=torch.float32)
    # sequence, _, _ = normalize_point_batch(sequence, NCHW=False)
    # mesh = array_to_mesh(sequence[0].numpy(), faces[0].numpy(), v_normals=True)
    # source_shape = torch.narrow(sequence, 0, 0, 1)
    # target_shape = torch.narrow(sequence, 0, 354, 1)
    # loss = shape_laplacian(vert1=source_shape, vert2=target_shape, face=faces)
    # print("min lap", loss.min(), "max lap", loss.max())

    # write_trimesh("./test_mesh_laplacian1.ply", source_shape[0], faces[0], v_colors=loss[0], cmap_name="rainbow")
    # write_trimesh("./test_mesh_laplacian2.ply", target_shape[0], faces[0], v_colors=loss[0], cmap_name="rainbow")
    # _v = mesh.points()
    # _v[:] = source_shape.numpy()
    # mesh.update_normals()
    # target_normals = torch.from_numpy(mesh.vertex_normals()).to(dtype=torch.float32).unsqueeze(0)

    ###### Labeled chamfer loss ######
    from ..utils.pc_utils import save_ply_property, save_ply
    from .geo_operations import normalize_point_batch

    pnl1 = np.loadtxt("/home/mnt/points/data/Coseg_Wang/Coseg_Wang_processed/Vase300Points/15.pts", dtype=np.float32, converters={6: lambda x: np.float32(x[1:-1])})
    pnl2 = np.loadtxt("/home/mnt/points/data/Coseg_Wang/Coseg_Wang_processed/Vase300Points/16.pts", dtype=np.float32, converters={6: lambda x: np.float32(x[1:-1])})

    V1 = torch.from_numpy(pnl1[:,:3]).cuda().unsqueeze(0)
    V1, _, _ = normalize_point_batch(V1, NCHW=False)
    V1 = V1.detach()
    V1.requires_grad_(True)
    # seems that the normals are inverted
    V1_n = -pnl1[:,3:6]
    V1_l = torch.from_numpy(pnl1[:,6:]).cuda().unsqueeze(0)

    V2 = torch.from_numpy(pnl2[:,:3]).cuda().unsqueeze(0)
    V2, _, _ = normalize_point_batch(V2, NCHW=False)
    V2 = V2.detach()
    # V2.requires_grad_(True)
    # seems that the normals are inverted
    V2_n = -pnl2[:,3:6]
    V2_l = torch.from_numpy(pnl2[:,6:]).cuda().unsqueeze(0)
    d12, d21, _, _ = nndistance(V1, V2)
    loss = torch.mean(d12) + torch.mean(d21)
    loss.backward(torch.ones_like(loss))
    print(V1.grad)
    print(V2.grad)

    d12, d21, idx1, idx2 = labeled_nndistance(V1, V2, V1_l, V2_l)
    save_ply_property(V1[0].cpu().detach().numpy(), V1_l[0,:,0].cpu().numpy(), "./test_labeled_nmdistance_input1.ply", normals=V1_n, cmap_name="Set1")
    save_ply_property(V2[0].cpu().detach().numpy(), V2_l[0,:,0].cpu().numpy(), "./test_labeled_nmdistance_input2.ply", normals=V2_n, cmap_name="Set1")
    save_ply_property(V1[0].cpu().detach().numpy(), d12[0].cpu().detach().numpy(), "./test_labeled_nmdistance1.ply", normals=V1_n, cmap_name="rainbow")
    save_ply_property(V2[0].cpu().detach().numpy(), d21[0].cpu().detach().numpy(), "./test_labeled_nmdistance2.ply", normals=V2_n, cmap_name="rainbow")
    cmap = cm.get_cmap("rainbow")
    zValue = V1[0,:,2]
    colors1 = cmap(zValue.detach().cpu().numpy())[:,:3]
    colors2 = np.take(colors1, idx2.detach().cpu().numpy()[0], axis=0)
    save_ply(V1[0].cpu().detach().numpy(), "./test_labeled_nmdistance_reg11.ply", colors=colors1)
    save_ply(V2[0].cpu().detach().numpy(), "./test_labeled_nmdistance_reg12.ply", colors=colors2)
    zValue = V2[0,:,2]
    colors2 = cmap(zValue.detach().cpu().numpy())[:,:3]
    colors1 = np.take(colors2, idx1.detach().cpu().numpy()[0], axis=0)
    save_ply(V1[0].cpu().detach().numpy(), "./test_labeled_nmdistance_reg21.ply", colors=colors1)
    save_ply(V2[0].cpu().detach().numpy(), "./test_labeled_nmdistance_reg22.ply", colors=colors2)

    loss = torch.mean(d12) + torch.mean(d21)
    d12.backward(torch.ones_like(d12))
    # loss.backward(torch.cuda.FloatTensor([1.0]))
    print(V1.grad)
    print(V2.grad)
    # test = gradcheck(nndistance, [pc1, pc2], eps=1e-3, atol=1e-4)
    # print(test)
    # pc2 = pc2.detach()
    # test = gradcheck(nndistance, [pc1, pc2], eps=1e-3, atol=1e-4)
    # print(test)
