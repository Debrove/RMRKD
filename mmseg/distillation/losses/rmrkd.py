import torch.nn as nn
import torch.nn.functional as F
import torch
from ..builder import DISTILL_LOSSES


@DISTILL_LOSSES.register_module()
class RMRKDLoss(nn.Module):
    """PyTorch version of `Masked Generative Distillation`

    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        name (str): the loss name of the layer
        alpha_mgd (float, optional): Weight of dis_loss. Defaults to 0.00002
        lambda_mgd (float, optional): masked ratio. Defaults to 0.75
    """

    def __init__(self, name, distance_type='euclidean', student_channels=19, teacher_channels=19,
                 loss_weight=0.5, sigma=1.0, tau=4.0, loss_type='kl', num_classes=19, decoupled=True):
        super(RMRKDLoss, self).__init__()
        self.name = name
        self.distance_type = distance_type
        self.loss_weight = loss_weight
        self.sigma = sigma
        self.tau = tau
        self.loss_type = loss_type
        self.num_classes = num_classes
        self.ignore_label = 255
        self.decoupled = decoupled

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

        self.embed_s = Embed(student_channels, student_channels)
        self.embed_t = Embed(teacher_channels, student_channels)

    def forward(self, preds_S, preds_T, gt):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]

        if self.align is not None:
            preds_S = self.align(preds_S)

        N, C, H, W = preds_S.shape
        gt = F.interpolate(gt.float(), size=[H, W], mode='bilinear', align_corners=False).int()

        if self.distance_type == 'euclidean':
            # IDD
            loss = self.euclidean_distance(preds_S, preds_T, gt, self.loss_type)
        elif self.distance_type == 'cosine':
            loss = self.cosine_distance(preds_S, preds_T, gt, self.loss_type)
        elif self.distance_type == 'cross_sim':
            loss = self.cross_similarity(preds_S, preds_T, gt, self.loss_type)

        return self.loss_weight * loss

    def euclidean_distance(self, preds_S, preds_T, gt, loss_type):
        loss = 0
        for i in range(self.num_classes):
            mask_feat_S = (gt == i).float()
            mask_feat_T = (gt == i).float()
            vi_S = (mask_feat_S * preds_S).sum(-1).sum(-1) / (mask_feat_S.sum(-1).sum(-1) + 1e-6)
            vi_T = (mask_feat_T * preds_T).sum(-1).sum(-1) / (mask_feat_T.sum(-1).sum(-1) + 1e-6)

            for j in range(self.num_classes):
                if i < j:
                    mask_feat_S = (gt == j).float()
                    mask_feat_T = (gt == j).float()
                    vj_S = (mask_feat_S * preds_S).sum(-1).sum(-1) / (mask_feat_S.sum(-1).sum(-1) + 1e-6)
                    vj_T = (mask_feat_T * preds_T).sum(-1).sum(-1) / (mask_feat_T.sum(-1).sum(-1) + 1e-6)

                    e_ij_S = F.pairwise_distance(vi_S, vj_S, p=2)
                    e_ij_T = F.pairwise_distance(vi_T, vj_T, p=2)
                    loss += 0.5 * (e_ij_T - e_ij_S) ** 2

        loss = loss.mean()

        return loss


    def cosine_distance(self, preds_S, preds_T, gt, loss_type):
        preds_S = self.embed_s(preds_S)
        preds_T = self.embed_s(preds_T)

        N, C, H, W = preds_S.shape
        loss_rec = 0
        list_S = []
        list_T = []
        cos = nn.CosineSimilarity(dim=1)

        for i in range(self.num_classes):
            mask_feat_S = (gt == i).float()
            mask_feat_T = (gt == i).float()
            vi_S = (mask_feat_S * preds_S).sum(-1).sum(-1) / (mask_feat_S.sum(-1).sum(-1) + 1e-6)
            vi_T = (mask_feat_T * preds_T).sum(-1).sum(-1) / (mask_feat_T.sum(-1).sum(-1) + 1e-6)

            loss_rec += cos(vi_S, vi_T)

            list_S.append(cos(preds_S, vi_S.unsqueeze(2).unsqueeze(3))/self.tau)
            list_T.append(cos(preds_T, vi_T.unsqueeze(2).unsqueeze(3))/self.tau)

        p_S = torch.stack(list_S, dim=1) / torch.stack(list_S, dim=1).sum(1).unsqueeze(1)
        p_T = torch.stack(list_T, dim=1) / torch.stack(list_T, dim=1).sum(1).unsqueeze(1)

        softmax_pred_T = F.softmax(p_T.view(-1, H * W), dim=0)
        logsoftmax = torch.nn.LogSoftmax(dim=0)
        loss = torch.sum(softmax_pred_T *
                         logsoftmax(p_T.view(-1, H * W)) -
                         softmax_pred_T *
                         logsoftmax(p_S.view(-1, H * W))) / (N * H * W)

        loss_rec = 1 - torch.sum(loss_rec) / (N * self.num_classes)

        return loss + loss_rec

    def cross_similarity(self, preds_S, preds_T, gt, loss_type):
        N, C, H, W = preds_T.shape
        loss = 0
        cos = nn.CosineSimilarity(dim=1)

        for i in range(self.num_classes):
            # if i not in gt:
            #     continue
            gt_mask = (gt == i).float()
            other_mask = (gt != i).float()

            gt_mask_feat_S = gt_mask * preds_S
            gt_mask_feat_T = gt_mask * preds_T
            other_mask_feat_S = other_mask * preds_S
            other_mask_feat_T = other_mask * preds_T

            vi_S = gt_mask_feat_S.sum(-1).sum(-1) / (gt_mask.sum(-1).sum(-1) + 1e-6)
            vi_T = gt_mask_feat_T.sum(-1).sum(-1) / (gt_mask.sum(-1).sum(-1) + 1e-6)

            if self.decoupled:
                cross_sim_other_ST = cos(vi_S.unsqueeze(2).unsqueeze(3), other_mask_feat_T).detach()
                sim_other_SS = cos(vi_S.unsqueeze(2).unsqueeze(3), other_mask_feat_S)

                cross_sim_other_TS = cos(vi_T.unsqueeze(2).unsqueeze(3), other_mask_feat_S)
                sim_other_TT = cos(vi_T.unsqueeze(2).unsqueeze(3), other_mask_feat_T).detach()

                # OOM
                cross_sim_gt_ST = cos(vi_S.unsqueeze(2).unsqueeze(3), gt_mask_feat_T).detach()
                sim_gt_SS = cos(vi_S.unsqueeze(2).unsqueeze(3), gt_mask_feat_S)

                cross_sim_gt_TS = cos(vi_T.unsqueeze(2).unsqueeze(3), gt_mask_feat_S)
                sim_gt_TT = cos(vi_T.unsqueeze(2).unsqueeze(3), gt_mask_feat_T).detach()
                
                loss += (self.contrast_sim_kd(sim_other_SS, cross_sim_other_ST) +
                         self.contrast_sim_kd(cross_sim_other_TS, sim_other_TT) +
                         self.contrast_sim_kd(sim_gt_SS, cross_sim_gt_ST) +
                         self.contrast_sim_kd(cross_sim_gt_TS, sim_gt_TT)) / 4

        return loss
    
    def contrast_sim_kd(self, s_logits, t_logits, dim=1, reduction='batchmean'):
        b, h, w = s_logits.shape
        p_s = F.log_softmax(s_logits / self.tau, dim=dim)
        p_t = F.softmax(t_logits / self.tau, dim=dim)
        sim_dis = F.kl_div(p_s, p_t, reduction=reduction) * self.tau ** 2
        return sim_dis

class Embed(nn.Module):
    """Embedding module"""

    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        # self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=1)
        self.proj = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1),
            nn.SyncBatchNorm(dim_in),
            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=1)
        )

    def forward(self, x):
        x = self.proj(x)
        return F.normalize(x, p=2, dim=1)

