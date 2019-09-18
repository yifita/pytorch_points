import torch
import numpy as np
from .._ext import losses
from . import operations
from ..utils.pytorch_utils import save_grad
from ..utils.geometry_utils import edge_vertex_indices

class MeshLaplacianLoss_old(torch.nn.Module):
    """
    compare uniform laplacian of two meshes assuming known correspondence
    num_point: number of vertices
    faces: (B,F,L) face indices
    metric: an instance of a module e.g. L1Loss
    """
    def __init__(self, num_point, faces, metric):
        super().__init__()
        self.laplacian1 = operations.UniformLaplacian()
        self.laplacian2 = operations.UniformLaplacian()
        self.metric = metric
        self.faces = faces

    def forward(self, vert1, vert2):
        lap1 = self.laplacian1(vert1, self.faces)
        lap2 = self.laplacian2(vert2, self.faces)
        return self.metric(lap1, lap2)


class UniformLaplacianSmoothnessLoss(torch.nn.Module):
    """
    Encourages minimal mean curvature shapes.
    """
    def __init__(self, num_point, faces, metric):
        super().__init__()
        self.laplacian = operations.UniformLaplacian()
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
            self.laplacian = operations.CotLaplacian()
        else:
            self.laplacian = operations.UniformLaplacian()

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
        lap1, knn_idx = operations.pointUniformLaplacian(point1, nn_size=self.nn_size)
        lap2, _ = operations.pointUniformLaplacian(point2, knn_idx=knn_idx)
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
        group_points, knn_idx, _ = operations.faiss_knn(self.nn_size+1, points_ref, points_ref, NCHW=False)
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
        group_points_ref, knn_idx, _ = operations.faiss_knn(self.nn_size+1, points_ref, points_ref, NCHW=False)
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
            EV.append(edge_vertex_indices(faces[b]))
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
            # (P,3) (Ex2) -> (Ex2,3)
            edge1 = torch.gather(vert1[b], 0, self.E[b].view(-1, 1).expand(-1, vert1.shape[-1])).view(-1, 2, vert1.shape[-1])
            edge2 = torch.gather(vert2[b], 0, self.E[b].view(-1, 1).expand(-1, vert2.shape[-1])).view(-1, 2, vert2.shape[-1])

            edge1 = edge1[:,0,:]-edge1[:,1,:]
            edge2 = edge2[:,0,:]-edge2[:,1,:]

            edge_length1 = torch.sum(edge1*edge1, dim=-1)
            edge_length2 = torch.sum(edge2*edge2, dim=-1)
            loss.append(self.metric(edge_length1, edge_length2))

        loss = torch.stack(loss, dim=0)
        loss = torch.mean(loss)

        return loss


class MeshStretchLoss(torch.nn.Module):
    """
    Penalize increase of edge length max(len2/len1-1, 0)
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
            EV.append(edge_vertex_indices(faces[b]))
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
            # (P,3) (Ex2) -> (Ex2,3)
            edge1 = torch.gather(vert1[b], 0, self.E[b].view(-1, 1).expand(-1, vert1.shape[-1])).view(-1, 2, vert1.shape[-1])
            edge2 = torch.gather(vert2[b], 0, self.E[b].view(-1, 1).expand(-1, vert2.shape[-1])).view(-1, 2, vert2.shape[-1])

            edge1 = edge1[:,0,:]-edge1[:,1,:]
            edge2 = edge2[:,0,:]-edge2[:,1,:]

            edge_length1 = torch.sum(edge1*edge1, dim=-1)
            edge_length2 = torch.sum(edge2*edge2, dim=-1)
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
            # (P,3) (Ex2) -> (Ex2,3)
            edge1 = torch.gather(verts1[b], 0, self.E[b].view(-1, 1).expand(-1, verts1.shape[-1])).view(-1, 2, verts1.shape[-1])
            edge1 = edge1[:,0,:]-edge1[:,1,:]
            edge_length1 = torch.sum(edge1*edge1, dim=-1)
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
        pred_normals = operations.batch_normals(pred, nn_size=10, NCHW=True)
        gt_normals = operations.batch_normals(gt, nn_size=10, NCHW=True)
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
            knn_points, knn_idx, distance2 = operations.faiss_knn(self.nn_size+1, points, points, NCHW=False)
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
        return dist1, dist2

    @staticmethod
    def backward(ctx, graddist1, graddist2):
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

class ChamferLoss(torch.nn.Module):
    """
    chamfer loss. bidirectional nearest neighbor distance of two point sets.
    Args:
        threshold (float): distance beyond threshold*average_distance not be considered
        forward_weight (float): if != 1, different weight for chamfer distance forward and backward
        percentage (float): consider a percentage of inner points
        max (bool): use hausdorf, i.e. use max instead of mean
    """

    def __init__(self, threshold=None, forward_weight=1.0, percentage=1.0, reduction="mean"):
        super(ChamferLoss, self).__init__()
        # only consider distance smaller than threshold*mean(distance) (remove outlier)
        self.__threshold = threshold
        self.forward_weight = forward_weight
        self.percentage = percentage
        self.reduction = reduction

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

        # discard border points
        if self.percentage < 1.0:
            pred_center = torch.mean(pred, dim=1, keepdim=True)
            num_point = pred.size(1)
            pred, _, _ = operations.faiss_knn(int(self.percentage * num_point), pred_center, pred, unique=False, NCHW=False)
            pred = torch.squeeze(pred, dim=1)
            # # BxN
            # dist_sqr = torch.sum((pred - pred_center)**2, dim=-1)
            # # Bx1
            # dist_sqrm = torch.max(dist_sqr, dim=1, keepdim=True)
            # weight = torch.exp(-dist_sqr / 1.5 * dist_sqrm)
            # weight = weight / torch.max(weight)
            # pred2gt = pred2gt * weight

            gt_center = torch.mean(gt, dim=1, keepdim=True)
            num_point = gt.size(1)
            gt, _, _ = operations.faiss_knn(int(self.percentage * num_point), gt_center, gt, unique=False, NCHW=False)
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

        pred2gt, gt2pred = nndistance(pred, gt)

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
        if self.reduction == "mean":
            loss = torch.mean(pred2gt, dim=-1) * self.forward_weight + torch.mean(gt2pred, dim=-1)
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(pred2gt, dim=-1) * self.forward_weight + torch.sum(gt2pred, dim=-1)
            loss = torch.mean(loss)
        elif self.reduction == "max":
            loss = torch.max(pred2gt, dim=-1)[0] * self.forward_weight + torch.max(gt2pred, dim=-1)[0]
            loss = torch.mean(loss)
        elif self.reduction == "none":
            loss = torch.mean(pred2gt, dim=-1) * self.forward_weight + torch.mean(gt2pred, dim=-1)
        else:
            raise NotImplementedError
        return loss


if __name__ == '__main__':
    from .operations import normalize_point_batch
    from ..utils.geometry_utils import array_to_mesh, write_trimesh, generatePolygon
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

    shape_laplacian = MeshLaplacianLoss(torch.nn.MSELoss(reduction="none"), use_cot=False, use_norm=True, consistent_topology=True)
    face = torch.arange(0, 20).reshape(1, 1, -1).cuda()
    smooth_old = UniformLaplacianSmoothnessLoss(20, face, torch.nn.MSELoss(reduction="mean"))
    shape_laplacian_old = MeshLaplacianLoss_old(20, face, torch.nn.MSELoss(reduction="none"))

    cage = generatePolygon(0, 0, 1.5, 0, 0, 0, 20)
    cage = torch.tensor([(x, y) for x, y in cage], dtype=torch.float).unsqueeze(0).cuda().requires_grad_(True)

    cage2 = generatePolygon(0, 0, 1.5, 0.1, 0.1, 0.1, 20)
    cage2 = torch.tensor([(x, y) for x, y in cage2], dtype=torch.float).unsqueeze(0).cuda().requires_grad_(True)

    loss1 = shape_laplacian(vert1=cage, face=face)
    loss2 = smooth_old(cage)
    loss1.mean().backward()
    grad1 = cage.grad
    loss2.mean().backward()
    grad2 = cage.grad

    print(torch.eq(grad1, grad2))

    shape_laplacian.use_norm = False
    loss1 = shape_laplacian(vert1=cage, vert2=cage2, face=face)
    loss2 = shape_laplacian_old(cage, cage2)
    loss1.mean().backward()
    grad1_1 = cage.grad
    grad1_2 = cage2.grad
    loss2.mean().backward()
    grad2_1 = cage.grad
    grad2_2 = cage2.grad
    print(torch.eq(grad1_1,grad2_1))
    print(torch.eq(grad1_2,grad2_2))
    # write_trimesh("./test_mesh_laplacian1.ply", source_shape[0], faces[0], v_colors=loss[0], cmap_name="rainbow")
    # write_trimesh("./test_mesh_laplacian2.ply", target_shape[0], faces[0], v_colors=loss[0], cmap_name="rainbow")
    # _v = mesh.points()
    # _v[:] = source_shape.numpy()
    # mesh.update_normals()
    # target_normals = torch.from_numpy(mesh.vertex_normals()).to(dtype=torch.float32).unsqueeze(0)

    # test = gradcheck(nndistance, [pc1, pc2], eps=1e-3, atol=1e-4)
    # print(test)
    # from torch.autograd import gradcheck
    # pc2 = pc2.detach()
    # test = gradcheck(nndistance, [pc1, pc2], eps=1e-3, atol=1e-4)
    # print(test)
