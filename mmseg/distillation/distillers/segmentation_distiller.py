import os
import torch.nn as nn
import torch.nn.functional as F
import torch
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.models import build_segmentor
from mmcv.runner import  load_checkpoint, _load_checkpoint, load_state_dict
from ..builder import DISTILLER,build_distill_loss
from collections import OrderedDict
from mmseg.core import add_prefix


@DISTILLER.register_module()
class SegmentationDistiller(BaseSegmentor):
    """Base distiller for detectors.

    It typically consists of teacher_model and student_model.
    """
    def __init__(self,
                 teacher_cfg,
                 student_cfg,
                 distill_cfg=None,
                 teacher_pretrained=None,
                 init_student=False,
                 use_logit=True,
                 use_adv=False,
                 num_classes=19,):

        super(SegmentationDistiller, self).__init__()

        self.use_adv = use_adv
        self.num_classes = num_classes

        self.teacher = build_segmentor(teacher_cfg.model,
                                        train_cfg=teacher_cfg.get('train_cfg'),
                                        test_cfg=teacher_cfg.get('test_cfg'))
        self.init_weights_teacher(teacher_pretrained)
        self.teacher.eval()

        self.use_logit = use_logit
        self.student= build_segmentor(student_cfg.model,
                                        train_cfg=student_cfg.get('train_cfg'),
                                        test_cfg=student_cfg.get('test_cfg'))
        self.student.init_weights()
        if init_student:
            t_checkpoint = _load_checkpoint(teacher_pretrained)
            all_name = []
            for name, v in t_checkpoint["state_dict"].items():
                if name.startswith("backbone."):
                    continue
                else:
                    all_name.append((name, v))

            state_dict = OrderedDict(all_name)
            load_state_dict(self.student, state_dict)
        self.distill_losses = nn.ModuleDict()
        self.distill_cfg = distill_cfg      
        for item_loc in distill_cfg:
            for item_loss in item_loc.methods:
                loss_name = item_loss.name
                self.distill_losses[loss_name] = build_distill_loss(item_loss)
    
    def base_parameters(self):
        return nn.ModuleList([self.student,self.distill_losses])

    @property
    def with_neck(self):
        """bool: whether the segmentor has neck"""
        return hasattr(self.student, 'neck') and self.student.neck is not None

    @property
    def with_auxiliary_head(self):
        """bool: whether the segmentor has auxiliary head"""
        return hasattr(self.student,
                       'auxiliary_head') and self.student.auxiliary_head is not None

    @property
    def with_decode_head(self):
        """bool: whether the segmentor has decode head"""
        return hasattr(self.student, 'decode_head') and self.student.decode_head is not None

    def init_weights_teacher(self, path=None):
        """Load the pretrained model in teacher detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        checkpoint = load_checkpoint(self.teacher, path, map_location='cpu')

    def forward_train(self, img, img_metas, gt_semantic_seg):

        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components(student's losses and distiller's losses).
        """
        with torch.no_grad():
            self.teacher.eval()
            fea_t = self.teacher.extract_feat(img)
            if self.use_logit:
                logit_t = self.teacher._decode_head_forward_test(fea_t, img_metas)

        student_feat = self.student.extract_feat(img)
        logit_s = self.student._decode_head_forward_test(student_feat, img_metas)

        losses = self.student.decode_head.losses(logit_s, gt_semantic_seg)
        loss_decode = dict()
        loss_decode.update(add_prefix(losses, 'decode'))

        student_loss = dict()
        student_loss.update(loss_decode)

        if self.student.with_auxiliary_head:
            loss_aux = self.student._auxiliary_head_forward_train(
                student_feat, img_metas, gt_semantic_seg)
            student_loss.update(loss_aux)

        loss_name1 = 'loss_mgd_fea1'
        student_loss[loss_name1] = self.distill_losses[loss_name1](student_feat[-1], fea_t[-1].detach())

        # student_loss['loss_rmrkd_f3'] = self.distill_losses['loss_rmrkd_f3'](student_feat[-2], fea_t[-2].detach(), gt_semantic_seg)
        student_loss['loss_rmrkd_f4'] = self.distill_losses['loss_rmrkd_f4'](student_feat[-1], fea_t[-1].detach(), gt_semantic_seg)
        
        if self.use_logit:
            num_samples = dict()
            student_loss['loss_dkd_channel'] = self.distill_losses['loss_dkd_channel'](logit_s, logit_t.detach(), gt_semantic_seg)

        return student_loss, num_samples
    
    def simple_test(self, img, img_metas, **kwargs):
        return self.student.simple_test(img, img_metas, **kwargs)
    def aug_test(self, imgs, img_metas, **kwargs):
        return self.student.aug_test(imgs, img_metas, **kwargs)
    def extract_feat(self, imgs):
        """Extract features from images."""
        return self.student.extract_feat(imgs)
    def encode_decode(self, img, img_metas):
        return self.student.encode_decode(img, img_metas)

