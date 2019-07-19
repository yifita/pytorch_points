import torch
from threading import Thread
from .._ext import losses
from . import operations


class MeshLaplacianLoss(torch.nn.Module):
    """
    compare uniform laplacian of two meshes
    num_point: number of vertices
    faces: (B,F,L) face indices
    metric: a module e.g. L1Loss
    """
    def __init__(self, num_point, faces, metric):
        super().__init__()
        self.laplacian = operations.UniformLaplacian(faces, num_point)
        self.metric = metric
    
    def forward(self, vert1, vert2):
        return self.metric(self.laplacian(vert1), self.laplacian(vert2))

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

    def __init__(self, threshold=None, forward_weight=1.0, percentage=1.0):
        super(ChamferLoss, self).__init__()
        # only consider distance smaller than threshold*mean(distance) (remove outlier)
        self.__threshold = threshold
        self.forward_weight = forward_weight
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
        cd_loss = torch.mean(CD_dist)
        return cd_loss


if __name__ == '__main__':
    pc1 = torch.randn([2, 600, 2], dtype=torch.float64,
                      requires_grad=True).cuda()
    pc2 = torch.randn([2, 600, 2], dtype=torch.float64,
                      requires_grad=True).cuda()
    chamfer = ChamferLoss()
    from torch.autograd import gradcheck
    # test = gradcheck(nndistance, [pc1, pc2], eps=1e-3, atol=1e-4)
    # print(test)
    pc2 = pc2.detach()
    test = gradcheck(nndistance, [pc1, pc2], eps=1e-3, atol=1e-4)
    print(test)
