import torch.nn as nn
import torch.nn.functional as F
import torch
from ..builder import DISTILL_LOSSES


@DISTILL_LOSSES.register_module()
class IFVDLoss(nn.Module):
    """PyTorch version of `Masked Generative Distillation`

    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        name (str): the loss name of the layer
        alpha_mgd (float, optional): Weight of dis_loss. Defaults to 0.00002
        lambda_mgd (float, optional): masked ratio. Defaults to 0.75
    """

    def __init__(self, name, student_channels=19, teacher_channels=19, loss_weight=10.0,):
        super(IFVDLoss, self).__init__()
        self.name = name
        self.loss_weight = loss_weight
        self.num_classes = 19

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

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

        center_feat_S = preds_S.clone()
        center_feat_T = preds_T.clone()

        for i in range(self.num_classes):
            mask_feat_S = (gt == i).float()
            mask_feat_T = (gt == i).float()
            center_feat_S = (1 - mask_feat_S) * center_feat_S + mask_feat_S * (
                        (mask_feat_S * preds_S).sum(-1).sum(-1) / (mask_feat_S.sum(-1).sum(-1) + 1e-6)).unsqueeze(
                -1).unsqueeze(-1)
            center_feat_T = (1 - mask_feat_T) * center_feat_T + mask_feat_T * (
                        (mask_feat_T * preds_T).sum(-1).sum(-1) / (mask_feat_T.sum(-1).sum(-1) + 1e-6)).unsqueeze(
                -1).unsqueeze(-1)
            # vi_S = (1 - mask_feat_S) * center_feat_S + (mask_feat_S * preds_S[:, i].unsqueeze(1)).sum(-1).sum(-1) / (
            #             mask_feat_S.sum(-1).sum(-1) + 1e-6)
            # vi_T = (1 - mask_feat_T) * center_feat_S +T(mask_feat_T * preds_T[:, i].unsqueeze(1)).sum(-1).sum(-1) / (mask_feat_T.sum(-1).sum(-1) + 1e-6)

        cos = nn.CosineSimilarity(dim=1)
        pcsim_feat_S = cos(preds_S, center_feat_S)
        pcsim_feat_T = cos(preds_T, center_feat_T)

        # mseloss
        mse = nn.MSELoss()
        loss = mse(pcsim_feat_S, pcsim_feat_T)

        return self.loss_weight * loss
