"""
kd_losses.py — Knowledge Distillation Loss Functions
=====================================================
This file contains all the loss functions used in knowledge distillation.

There are two types of KD loss:
  1. logit_kd_loss   : BCE-based distillation between student and teacher class logits
  2. feature_kd_loss : MSE loss between L2-normalized intermediate feature maps

The combined loss is assembled in kd_trainer.py:
  L_total = α * L_det  +  β * L_kd_logit  +  γ * L_feat

These functions are pure (no model state, no side effects) and easy to test.

IMPORTANT — What each function expects:
  logit_kd_loss   : class-channel-only tensors [B, nc, H, W].
                    The caller (kd_trainer.py) must split the Detect head output
                    [B, 4*reg_max+nc, H, W] → cls [B, nc, H, W] before calling.
  feature_kd_loss : neck feature tensors [B, C, H, W] after ChannelAdapter.
"""

import torch
import torch.nn.functional as F


def logit_kd_loss(student_cls: list, teacher_cls: list, temperature: float = 4.0) -> torch.Tensor:
    """
    Logit-based Knowledge Distillation Loss (BCE with Temperature-Scaled Sigmoids).

    What this does:
    ---------------
    Instead of just learning from hard labels (0 or 1), the student also learns
    from the teacher's soft per-class confidence at each spatial location.
    Example: if the teacher outputs high confidence for "short" and moderate for
    "spur" at a given location, the student learns this nuanced signal.

    Why BCE instead of Softmax KL-Divergence:
    ------------------------------------------
    YOLOv8 uses Binary Cross Entropy (BCE) for classification — each class is an
    independent binary prediction. Classes are NOT mutually exclusive. Using
    Softmax would force a sum-to-1 constraint that distorts the teacher's output
    and is structurally incompatible with YOLOv8's loss function. Instead, we use
    BCE with sigmoid targets: each class is distilled independently as a binary
    variable, matching the architecture's design.

    Temperature (T):
    ----------------
    - T=1 : standard sigmoid (sharp probabilities, like hard labels)
    - T>1 : softer probabilities — more "dark knowledge" is transferred
    - T=4 is the recommended default
    - Too high (T=8+) can make the signal too noisy

    Math:
    -----
    At each spatial location, for each class independently:
      t_prob = sigmoid(t / T)               (teacher soft target per class)
      L = BCE_with_logits(s / T, t_prob)    (student learns teacher's per-class confidence)
      L_scaled = L * T²                     (gradient scale correction)
    Final loss = mean over all FPN scales.

    Args:
        student_cls : list of student class tensors [P3, P4, P5], each [B, nc, H, W]
                      Class channels ONLY — box regression channels are excluded.
        teacher_cls : list of teacher class tensors [P3, P4, P5], each [B, nc, H, W]
        temperature : softening temperature T (default 4.0)

    Returns:
        Scalar loss tensor (mean across all FPN scales)
    """
    total_loss = torch.tensor(0.0, device=student_cls[0].device, dtype=student_cls[0].dtype)
    num_scales = len(student_cls)

    for s_cls, t_cls in zip(student_cls, teacher_cls):
        # s_cls, t_cls shape: [B, nc, H, W]

        # Match dtypes — handles AMP float16/float32 mismatch between student and teacher
        t_cls = t_cls.to(dtype=s_cls.dtype)

        # Align spatial size if needed (shouldn't happen for same imgsz, but safe to check)
        if t_cls.shape[2:] != s_cls.shape[2:]:
            t_cls = F.interpolate(t_cls, size=s_cls.shape[2:], mode='bilinear', align_corners=False)

        # Temperature-scaled BCE distillation (per-class independent binary)
        # Teacher: sigmoid targets — each class is an independent soft probability
        # Student: raw logits divided by T — F.binary_cross_entropy_with_logits
        #          applies sigmoid internally for numerical stability
        t_soft = torch.sigmoid(t_cls / temperature)     # [B, nc, H, W] — per-class soft targets
        s_scaled = s_cls / temperature                   # [B, nc, H, W] — temperature-scaled student logits

        # BCE: each class is distilled independently (matches YOLOv8's BCE loss design)
        # T² corrects the gradient scale reduction caused by dividing logits by T
        scale_loss = F.binary_cross_entropy_with_logits(
            s_scaled, t_soft, reduction='mean'
        ) * (temperature ** 2)

        total_loss = total_loss + scale_loss

    return total_loss / num_scales


def feature_kd_loss(student_feats_adapted: list, teacher_feats: list) -> torch.Tensor:
    """
    Feature-based Knowledge Distillation Loss (MSE on L2-normalized neck features).

    What this does:
    ---------------
    Forces the student's internal feature maps to mimic the teacher's feature maps
    at the same network stages (P3, P4, P5 neck outputs). This is especially helpful
    for detecting small defects because the teacher's spatial features encode richer
    location and context information.

    The student features are first projected to match the teacher's channel dimensions
    via a ChannelAdapter (1×1 convolution) — this projection happens in kd_trainer.py
    BEFORE calling this function. Only L2-normalized MSE is computed here.

    Why L2 normalization:
    ----------------------
    Without normalization, feature maps in deeper layers have much larger magnitudes
    than shallow ones. This would cause the P5 (deepest) scale to dominate the loss
    and prevent meaningful gradient signal from reaching P3 (where small defects are
    best detected). L2 normalization per channel makes each scale contribute equally.

    Math:
    -----
    s_norm = L2_normalize(s_adapted, dim=1)    # per-channel unit norm
    t_norm = L2_normalize(t_feat,    dim=1)    # per-channel unit norm
    L_scale = MSE(s_norm, t_norm.detach())
    Final loss = mean over all FPN scales.

    Args:
        student_feats_adapted : list of adapted student tensors [P3, P4, P5]
                                after passing through ChannelAdapter.
                                Each tensor: [B, teacher_C, H, W]
        teacher_feats         : list of teacher neck tensors [P3, P4, P5]
                                Each tensor: [B, teacher_C, H, W]
                                (should already be detached — no teacher gradients)

    Returns:
        Scalar loss tensor (mean across all FPN scales)
    """
    total_loss = torch.tensor(0.0, device=student_feats_adapted[0].device,
                              dtype=student_feats_adapted[0].dtype)
    num_scales = len(student_feats_adapted)

    for s_adapted, t_feat in zip(student_feats_adapted, teacher_feats):
        # Match dtypes — handles AMP float16/float32 mismatch between student and teacher
        t_feat = t_feat.detach().to(dtype=s_adapted.dtype)

        # L2 normalize along channel dimension (dim=1)
        # After normalization, all feature vectors have unit norm — scale invariant
        s_norm = F.normalize(s_adapted, p=2, dim=1)
        t_norm = F.normalize(t_feat,    p=2, dim=1)

        # Sum over channels (dim=1) so loss is independent of channel count,
        # then average over batch and spatial dims
        scale_loss = F.mse_loss(s_norm, t_norm, reduction='none').sum(dim=1).mean()
        total_loss = total_loss + scale_loss

    return total_loss / num_scales