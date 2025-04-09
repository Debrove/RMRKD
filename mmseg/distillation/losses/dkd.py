# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import DISTILL_LOSSES


@DISTILL_LOSSES.register_module()
class DecoupledKD(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.

    <https://arxiv.org/abs/2011.13256>`_.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        name,
        tau=1.0,
        loss_weight1=0.5,
        loss_weight2=0.5,
        student_channels=19,
        teacher_channels=19,
        features=False,
        segment=False,
        loss_type="kl",
        merge_label=None,
        dim=1,
        num_classes=19,
    ):
        super(DecoupledKD, self).__init__()
        self.name = name
        self.tau = tau
        self.loss_weight1 = loss_weight1
        self.loss_weight2 = loss_weight2
        self.features = features
        self.segment = segment
        self.loss_type = loss_type
        self.merge_label = merge_label
        self.dim = dim
        self.num_classes = num_classes
        self.ignore_label = 255

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

        self.generation = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1))

    def forward(self, preds_S, preds_T, gt):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W).

        Return:
            torch.Tensor: The calculated loss value.
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]

        if self.align is not None:
            preds_S = self.align(preds_S)

        if self.loss_type == "kl":
            loss = self.forward_kl(preds_T, preds_S, gt)
        elif self.loss_type == "mse":
            loss = self.forward_mse(preds_T, preds_S, gt)

        return loss

    def forward_kl(self, preds_T, preds_S, gt):
        N, C, H, W = preds_S.shape
        # gt = F.interpolate(gt.float(), size=[H, W], mode='bilinear', align_corners=False).int()
        gt = gt.float().clone()
        gt = torch.nn.functional.interpolate(gt, (H, W), mode='nearest').long()

        # device = preds_S.device
        # mat = torch.rand((N, 1, H * W)).to(device)
        # mat = torch.where(mat > 1 - 0.75, 0, 1).to(device)

        # masked_fea = torch.mul(preds_S, mat)
        # new_fea = self.generation(masked_fea)

        gt_mask = self._get_gt_mask(preds_S, gt, self.merge_label)
        other_mask = self._get_other_mask(preds_S, gt, self.merge_label)

        # gt_mat = torch.mul(gt_mask, mat)
        # other_mat = torch.mul(other_mask, mat)

        if self.features:
            segment_loss = 0

            for i in range(gt_mask.shape[1]):
                gt_mask_T = torch.mul(preds_T.view(N, C, W * H), gt_mask[:, i, :].unsqueeze(1)).view(-1, W * H)
                gt_mask_S = torch.mul(preds_S.view(N, C, W * H), gt_mask[:, i, :].unsqueeze(1)).view(-1, W * H)
                other_mask_T = torch.mul(preds_T.view(N, C, W * H), other_mask[:, i, :].unsqueeze(1)).view(-1, W * H)
                other_mask_S = torch.mul(preds_S.view(N, C, W * H), other_mask[:, i, :].unsqueeze(1)).view(-1, W * H)

                segment_loss += self.segment_loss(gt_mask_S, gt_mask_T, other_mask_S, other_mask_T, gt, N, C)

            #
            #     gt_softmax_pred_T = F.softmax(gt_mask_T / self.tau, dim=1)
            #
            #     logsoftmax = torch.nn.LogSoftmax(dim=1)
            #     target_loss = torch.sum(gt_softmax_pred_T *
            #                             logsoftmax(gt_mask_T.view(-1, W * H) / self.tau) -
            #                             gt_softmax_pred_T *
            #                             logsoftmax(gt_mask_S.view(-1, W * H) / self.tau)) * (self.tau ** 2)
            #
            #     non_terget_softmax_pred_T = F.softmax(other_mask_T / self.tau, dim=1)
            #     non_target_loss = torch.sum(non_terget_softmax_pred_T *
            #                                 logsoftmax(other_mask_T.view(-1, W * H) / self.tau) -
            #                                 non_terget_softmax_pred_T *
            #                                 logsoftmax(other_mask_S.view(-1, W * H) / self.tau)) * (self.tau ** 2)
            #
            #     loss += (self.loss_weight1 * target_loss + self.loss_weight2 * non_target_loss) / (C * N)
            # loss = loss / gt_mask.shape[1]
        else:
            gt_mask_T = gt_mask.view(-1, W * H) * preds_T.view(-1, W * H)
            gt_mask_S = gt_mask.view(-1, W * H) * preds_S.view(-1, W * H)
            other_mask_T = other_mask.view(-1, W * H) * preds_T.view(-1, W * H)
            other_mask_S = other_mask.view(-1, W * H) * preds_S.view(-1, W * H)

            loss_segment = 0
            if self.segment:
                loss_segment = self.old_segment_loss(gt_mask_S, gt_mask_T, other_mask_S, other_mask_T, gt, N, C)

            gt_softmax_pred_T = F.softmax(gt_mask_T / self.tau, dim=self.dim)

            logsoftmax = torch.nn.LogSoftmax(dim=self.dim)
            target_loss = torch.sum(gt_softmax_pred_T *
                                    logsoftmax(gt_mask_T.view(-1, W * H) / self.tau) -
                                    gt_softmax_pred_T *
                                    logsoftmax(gt_mask_S.view(-1, W * H) / self.tau)) * (self.tau ** 2)

            non_target_softmax_pred_T = F.softmax(other_mask_T / self.tau, dim=self.dim)
            non_target_loss = torch.sum(non_target_softmax_pred_T *
                                        logsoftmax(other_mask_T.view(-1, W * H) / self.tau) -
                                        non_target_softmax_pred_T *
                                        logsoftmax(other_mask_S.view(-1, W * H) / self.tau)) * (self.tau ** 2)

            n_dim = C if self.dim == 1 else H * W
            loss = (self.loss_weight1 * target_loss + self.loss_weight2 * non_target_loss) / (n_dim * N)
        return loss + loss_segment

    def forward_mse(self, preds_T, preds_S, gt):
        loss_mse = nn.MSELoss(reduction='mean')
        N, C, H, W = preds_S.shape

        # device = preds_S.device
        # mat = torch.rand((N, 1, H, W)).to(device)
        # mat = torch.where(mat > 1 - 0.75, 0, 1).to(device)
        #
        # masked_fea = torch.mul(preds_S, mat)
        # new_fea = self.generation(masked_fea)

        gt_mask = _get_gt_mask(preds_S, gt)
        other_mask = _get_other_mask(preds_S, gt)

        loss = 0
        for i in range(gt_mask.shape[1]):
            gt_mask_T = torch.mul(preds_T, gt_mask[:, i, :].view(-1, W, H).unsqueeze(1))
            gt_mask_S = torch.mul(preds_S, gt_mask[:, i, :].view(-1, W, H).unsqueeze(1))
            other_mask_T = torch.mul(preds_T, other_mask[:, i, :].view(-1, W, H).unsqueeze(1))
            other_mask_S = torch.mul(preds_S, other_mask[:, i, :].view(-1, W, H).unsqueeze(1))

            target_loss = loss_mse(gt_mask_S, gt_mask_T)
            non_target_loss = loss_mse(other_mask_S, other_mask_T)

            loss += (self.loss_weight1 * target_loss + self.loss_weight2 * non_target_loss) / N

        return loss

    def segment_loss(self, pred_S, pred_T, gt):
        N, C, H, W = pred_S.shape
        target_segment_loss = 0
        non_target_segment_loss = 0

        for bs in range(N):
            this_feat_gt_S = pred_S[bs].contiguous().view(C, -1)
            this_feat_gt_T = gt_mask_T.view(N, C, -1)[bs].contiguous().view(C, -1)
            this_feat_other_S = other_mask_S.view(N, C, -1)[bs].contiguous().view(C, -1)
            this_feat_other_T = other_mask_T.view(N, C, -1)[bs].contiguous().view(C, -1)

            this_label = gt[bs].contiguous().view(-1)
            this_label_ids = torch.unique(this_label)
            this_label_ids = [x for x in this_label_ids if x != self.ignore_label]

            if len(this_label_ids) == 0:
                continue

            list_segment_gt_S = []
            list_segment_gt_T = []
            list_segment_other_S = []
            list_segment_other_T = []

            for lb in this_label_ids:
                gt_idxs = (this_label == lb).nonzero()
                other_idxs = (this_label != lb).nonzero()

                segment_gt_S = torch.mean(this_feat_gt_S[:, gt_idxs], dim=1).squeeze(1)
                segment_gt_T = torch.mean(this_feat_gt_T[:, gt_idxs], dim=1).squeeze(1)
                segment_other_S = torch.mean(this_feat_other_S[:, other_idxs], dim=1).squeeze(1)
                segment_other_T = torch.mean(this_feat_other_T[:, other_idxs], dim=1).squeeze(1)

                list_segment_gt_S.append(segment_gt_S)
                list_segment_gt_T.append(segment_gt_T)
                list_segment_other_S.append(segment_other_S)
                list_segment_other_T.append(segment_other_T)

            segment_gt_S = torch.stack(list_segment_gt_S, dim=0)
            segment_gt_T = torch.stack(list_segment_gt_T, dim=0)
            segment_other_S = torch.stack(list_segment_other_S, dim=0)
            segment_other_T = torch.stack(list_segment_other_T, dim=0)

            gt_softmax_segment_T = F.softmax(segment_gt_T / self.tau, dim=self.dim)

            logsoftmax = torch.nn.LogSoftmax(dim=self.dim)
            target_segment_loss += torch.sum(gt_softmax_segment_T *
                                             logsoftmax(segment_gt_T / self.tau) -
                                             gt_softmax_segment_T *
                                             logsoftmax(segment_gt_S / self.tau)) * (self.tau ** 2) / len(
                this_label_ids)

            non_target_softmax_segment_T = F.softmax(segment_other_T / self.tau, dim=self.dim)
            non_target_segment_loss += torch.sum(non_target_softmax_segment_T *
                                                 logsoftmax(segment_other_T / self.tau) -
                                                 non_target_softmax_segment_T *
                                                 logsoftmax(segment_other_S / self.tau)) * (
                                               self.tau ** 2) / len(this_label_ids)

        loss_segment = (target_segment_loss + self.loss_weight2 * non_target_segment_loss) / N
        return loss_segment

    def old_segment_loss(self, gt_mask_S, gt_mask_T, other_mask_S, other_mask_T, gt, N, C):
        target_segment_loss = 0
        non_target_segment_loss = 0
        for bs in range(N):
            this_feat_gt_S = gt_mask_S.view(N, C, -1)[bs].contiguous().view(C, -1)
            this_feat_gt_T = gt_mask_T.view(N, C, -1)[bs].contiguous().view(C, -1)
            this_feat_other_S = other_mask_S.view(N, C, -1)[bs].contiguous().view(C, -1)
            this_feat_other_T = other_mask_T.view(N, C, -1)[bs].contiguous().view(C, -1)

            this_label = gt[bs].contiguous().view(-1)
            this_label_ids = torch.unique(this_label)
            this_label_ids = [x for x in this_label_ids if x != self.ignore_label]

            if len(this_label_ids) == 0:
                continue

            list_segment_gt_S = []
            list_segment_gt_T = []
            list_segment_other_S = []
            list_segment_other_T = []

            for lb in this_label_ids:
                idxs = (this_label == lb).nonzero()
                segment_gt_S = torch.mean(this_feat_gt_S[:, idxs], dim=1).squeeze(1)
                segment_gt_T = torch.mean(this_feat_gt_T[:, idxs], dim=1).squeeze(1)
                segment_other_S = torch.mean(this_feat_other_S[:, idxs], dim=1).squeeze(1)
                segment_other_T = torch.mean(this_feat_other_T[:, idxs], dim=1).squeeze(1)

                list_segment_gt_S.append(segment_gt_S)
                list_segment_gt_T.append(segment_gt_T)
                list_segment_other_S.append(segment_other_S)
                list_segment_other_T.append(segment_other_T)

            segment_gt_S = torch.stack(list_segment_gt_S, dim=0)
            segment_gt_T = torch.stack(list_segment_gt_T, dim=0)
            segment_other_S = torch.stack(list_segment_other_S, dim=0)
            segment_other_T = torch.stack(list_segment_other_T, dim=0)

            gt_softmax_segment_T = F.softmax(segment_gt_T / self.tau, dim=self.dim)

            logsoftmax = torch.nn.LogSoftmax(dim=self.dim)
            target_segment_loss += torch.sum(gt_softmax_segment_T *
                                             logsoftmax(segment_gt_T / self.tau) -
                                             gt_softmax_segment_T *
                                             logsoftmax(segment_gt_S / self.tau)) * (self.tau ** 2) / len(
                this_label_ids)

            non_target_softmax_segment_T = F.softmax(segment_other_T / self.tau, dim=self.dim)
            non_target_segment_loss += torch.sum(non_target_softmax_segment_T *
                                                 logsoftmax(segment_other_T / self.tau) -
                                                 non_target_softmax_segment_T *
                                                 logsoftmax(segment_other_S / self.tau)) * (
                                               self.tau ** 2) / len(this_label_ids)

        loss_segment = (target_segment_loss + self.loss_weight2 * non_target_segment_loss) / N
        return loss_segment

    def _get_gt_mask(self, pred, gt, merge_label):
        N, C, H, W = pred.shape
        gt_flatten = gt.reshape(-1)
        mask = []
        if merge_label is None:
            for i in range(self.num_classes):
                idx = torch.nonzero(gt_flatten == i)
                m = torch.zeros_like(gt_flatten).scatter_(0, idx.squeeze(1), 1)
                m = m.view(N, 1, -1)
                mask.append(m)
        else:
            for i in merge_label:
                # for r in range(i[0], i[1] + 1):
                lst_idx = []
                for j in range(i[0], i[1] + 1):
                    idx = torch.nonzero(gt_flatten == j)
                    lst_idx.append(idx)
                idx = torch.cat(lst_idx)
                m = torch.zeros_like(gt_flatten).scatter_(0, idx.squeeze(1), 1)
                m = m.view(N, 1, -1)
                mask.append(m)
        mask = torch.cat(mask, dim=1)

        return mask

    def _get_other_mask(self, pred, gt, merge_label):
        N, C, H, W = pred.shape
        # gt = F.interpolate(gt.float(), size=[H, W], mode='bilinear', align_corners=False).int()
        gt_flatten = gt.reshape(-1)
        mask = []
        if merge_label is None:
            for i in range(self.num_classes):
                idx = torch.nonzero(gt_flatten == i)
                m = torch.ones_like(gt_flatten).scatter_(0, idx.squeeze(1), 0)
                m = m.view(N, 1, -1)
                mask.append(m)
        else:
            for i in merge_label:
                # for r in range(i[0], i[1] + 1):
                lst_idx = []
                for j in range(i[0], i[1] + 1):
                    idx = torch.nonzero(gt_flatten == j)
                    lst_idx.append(idx)
                idx = torch.cat(lst_idx)
                m = torch.ones_like(gt_flatten).scatter_(0, idx.squeeze(1), 0)
                m = m.view(N, 1, -1)
                mask.append(m)

        mask = torch.cat(mask, dim=1)

        return mask
