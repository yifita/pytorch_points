import torch
from threading import Thread
from .._ext import losses
from . import operations
from ..utils.pytorch_utils import save_grad

class MeshLaplacianLoss(torch.nn.Module):
    """
    compare uniform laplacian of two meshes assuming known correspondence
    num_point: number of vertices
    faces: (B,F,L) face indices
    metric: an instance of a module e.g. L1Loss
    """
    def __init__(self, num_point, faces, metric):
        super().__init__()
        self.laplacian1 = operations.UniformLaplacian(faces, num_point)
        self.laplacian2 = operations.UniformLaplacian(faces, num_point)
        self.metric = metric
    
    def forward(self, vert1, vert2):
        lap1 = self.laplacian1(vert1)
        lap2 = self.laplacian2(vert2)
        return self.metric(lap1, lap2)

class LaplacianSmoothnessLoss(object):
    """
    Encourages minimal mean curvature shapes.
    """
    def __init__(self, faces, vert, toref=True):
        # Input:
        #  faces: B x F x 3
        self.toref = toref
        # V x V
        self.laplacian = operations.CotLaplacian(faces)
        self.Lx = None
        tmp = self.laplacian(vert)
        self.curve_gt = torch.norm(tmp.view(-1, tmp.size(2)), p=2, dim=1).float()
        if not self.toref:
            self.curve_gt = self.curve_gt*0
    
    def __call__(self, verts):
        self.Lx = self.laplacian(verts)
        # Reshape to BV x 3
        Lx = self.Lx.view(-1, self.Lx.size(2))
        loss = (torch.norm(Lx, p=2, dim=1).float()-self.curve_gt).mean()
        return loss

class UniformLaplacianSmoothnessLoss(torch.nn.Module):
    """
    Encourages minimal mean curvature shapes.
    """
    def __init__(self, num_point, faces, metric):
        super().__init__()
        self.laplacian = operations.UniformLaplacian(faces, num_point)
        self.metric = metric
    
    def __call__(self, vert, vert_ref=None):
        lap = self.laplacian(vert)
        curve = torch.norm(lap, p=2, dim=-1)
        if vert_ref is not None:
            lap_ref = self.laplacian(vert)
            curve_gt = torch.norm(lap_ref, p=2, dim=-1)
            loss = self.metric(curve, curve_gt)
        else:
            loss = curve
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


class PointLaplacianLoss(torch.nn.Module):
    """
    compare uniform laplacian of two point clouds assuming known correspondence
    metric: an instance of a module e.g. L1Loss
    """
    def __init__(self, nn_size, metric):
        super().__init__()
        self.metric = metric
        self.nn_size = nn_size

    def forward(self, point1, point2):
        """
        point1: (B,N,D) ref points (where connectivity is computed)
        point2: (B,N,D) pred points, uses connectivity of point1
        """
        lap1, knn_idx = operations.pointUniformLaplacian(point1, nn_size=nn_size)
        lap2, _ = operations.pointUniformLaplacian(point2, knn_idx=knn_idx)
        return self.metric(lap1, lap2)


class PointEdgeLengthLoss(torch.nn.Module):
    """
    compare uniform laplacian of two point clouds assuming known correspondence
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


class StretchLoss(torch.nn.Module):
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
            return torch.sum(stretch)
        elif "none":
            return stretch
        else:
            raise NotImplementedError


class MeshEdgeLengthLoss(torch.nn.Module):
    """
    Penalize large edge deformation for meshes of the same topology (assuming correspondance)
    """
    def __init__(self, metric):
        super().__init__(self)
        self.metric = metric 

    def forward(self, verts1, verts2, faces=None):
        """
        verts1: (B, N, 3)
        verts2: (B, N, 3)
        faces:  (B, F, L)
        """
        F = faces.shape[1]
        # (B, F, N, 3)
        face_verts1 = torch.gather(verts1.unsqueeze(1).expand(-1, F, -1, -1), 2, faces.unsqueeze(-1).expand(-1, -1, -1, verts1.shape[-1]))
        face_verts2 = torch.gather(verts2.unsqueeze(1).expand(-1, F, -1, -1), 2, faces.unsqueeze(-1).expand(-1, -1, -1, verts2.shape[-1]))
        edge1 = face_verts1[:, :, [i for i in range(F)]]-face_verts1[:, :, [i for i in range(1, self.face_deg)]+[0]]
        edge2 = face_verts2[:, :, [i for i in range(F)]]-face_verts2[:, :, [i for i in range(1, self.face_deg)]+[0]]
        # distance
        d1 = edge1 * edge1 
        d2 = edge2 * edge2
        return self.metric(d1, d2)

class NormalLoss(torch.nn.Module):
    def __init__(self, nn_size=10):
        self.nn_size = nn_size

    def forward(self, pred, gt):
        pred_normals = operations.batch_normals(pred, nn_size=10, NCHW=True)
        gt_normals = operations.batch_normals(gt, nn_size=10, NCHW=True)
        # compare the normal with the closest point


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


nndistance = NmDistanceFunction.apply


class ChamferLoss(torch.nn.Module):
    """
    chamfer loss. bidirectional nearest neighbor distance of two point sets.
    Args:
        threshold (float): distance beyond threshold*average_distance not be considered
        forward_weight (float): if != 1, different weight for chamfer distance forward and backward
        percentage (float): consider a percentage of inner points
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
            pred, _, _ = operations.group_knn(int(self.percentage * num_point), pred_center, pred, unique=False, NCHW=False)
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
            gt, _, _ = operations.group_knn(int(self.percentage * num_point), gt_center, gt, unique=False, NCHW=False)
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
        pred2gt = torch.mean(pred2gt, dim=1)
        gt2pred = torch.mean(gt2pred, dim=1)
        CD_dist = self.forward_weight * pred2gt + gt2pred
        # CD_dist_norm = CD_dist/radius
        if self.reduction == "mean":
            cd_loss = torch.mean(CD_dist)
        elif self.reduction == "sum":
            cd_loss = torch.sum(CD_dist)
        elif self.reduction is "none":
            cd_loss = CD_dist
        else:
            raise NotImplementedError
        return cd_loss


if __name__ == '__main__':
    pc1 = torch.randn([2, 600, 2], dtype=torch.float32,
                      requires_grad=True).cuda()
    pc2 = torch.randn([2, 600, 2], dtype=torch.float32,
                      requires_grad=True).cuda()
    chamfer = ChamferLoss()
    edgeLoss = PointEdgeLengthLoss(nn_size=3, metric=torch.nn.MSELoss())
    edgeShouldBeZero = edgeLoss(pc1, pc1)
    print(edgeShouldBeZero)
    assert(torch.all(edgeShouldBeZero == 0))
    # test = gradcheck(nndistance, [pc1, pc2], eps=1e-3, atol=1e-4)
    # print(test)
    # from torch.autograd import gradcheck
    # pc2 = pc2.detach()
    # test = gradcheck(nndistance, [pc1, pc2], eps=1e-3, atol=1e-4)
    # print(test)
