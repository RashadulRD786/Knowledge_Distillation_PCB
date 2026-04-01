"""
kd_trainer.py — Knowledge Distillation Trainer
===============================================
Defines KDTrainer, which extends Ultralytics' DetectionTrainer to add
knowledge distillation during student training.

Also defines:
  FeatureExtractor  — persistent forward hooks to capture P3/P4/P5 neck features
  ChannelAdapter    — 1×1 conv projecting student channels → teacher channels

You do NOT need to call this file directly.
  - train_kd.py    uses it for the main KD training run
  - run_ablation.py uses it for all KD ablation experiments

How it works (one training step):
----------------------------------
1. preprocess_batch()
     a. Teacher forward (train mode, BN frozen) → cache raw logits + neck features
2. Ultralytics calls self.model(batch['img']) → student forward
     b. Student neck hooks fire → student P3/P4/P5 features captured automatically
3. criterion(preds, batch)
     c. detection loss  = super().criterion(preds, batch)  [box + cls + DFL]
     d. logit_kd_loss   = KL(teacher cls || student cls)  [cls channels only]
     e. feature_kd_loss = MSE(L2-norm student adapted, L2-norm teacher neck)
     f. total = α*det + β*logit + γ*feat
4. Backprop updates student weights + ChannelAdapter weights only.
   Teacher is fully frozen throughout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import de_parallel

from kd_losses import logit_kd_loss, feature_kd_loss


def _unwrap_model(model):
    """Safely unwrap a model from DDP/DataParallel wrappers."""
    return de_parallel(model)


# =============================================================================
# FeatureExtractor — persistent forward hooks on neck layers
# =============================================================================

class FeatureExtractor:
    """
    Captures intermediate feature maps from specified model layers via persistent hooks.

    Attach to the P3/P4/P5 neck layers — those whose outputs feed directly into
    the Detect head. The layer indices come from model.model[-1].f (the Detect
    head's source index list), e.g. [15, 18, 21] for YOLOv8n.

    Usage:
        extractor = FeatureExtractor(model.model, [15, 18, 21])
        model(imgs)                      # hooks fire automatically
        feats = extractor.get_feats()    # returns [P3_feat, P4_feat, P5_feat]
        extractor.remove()               # clean up when done training

    Note: hooks overwrite on every forward pass, so always read immediately
    after the forward you care about.
    """

    def __init__(self, model_sequential: nn.Sequential, layer_indices: list):
        """
        Args:
            model_sequential : model.model — the nn.Sequential backbone+neck layers
            layer_indices    : absolute layer indices to hook, e.g. [15, 18, 21]
        """
        self._feats   = {}
        self._handles = []
        self._indices = sorted(layer_indices)
        self._model_sequential = model_sequential

        self.register()

    def _make_hook(self, idx: int):
        def hook(module, inp, output):
            # Some layers (C2f, SPPF) return a list/tuple; take the first element
            feat = output[0] if isinstance(output, (list, tuple)) else output
            self._feats[idx] = feat
        return hook

    def register(self):
        """Attach hooks to the model layers. Safe to call after remove()."""
        self.remove()  # clear any existing hooks first
        for idx in self._indices:
            handle = self._model_sequential[idx].register_forward_hook(
                self._make_hook(idx)
            )
            self._handles.append(handle)

    def get_feats(self) -> list:
        """Return captured features in layer-index order as a Python list of tensors."""
        return [self._feats[idx] for idx in self._indices]

    def remove(self):
        """Remove all hooks from the model. Safe to call multiple times."""
        for h in self._handles:
            h.remove()
        self._handles.clear()


# =============================================================================
# ChannelAdapter — 1×1 conv student→teacher channel projection
# =============================================================================

class ChannelAdapter(nn.Module):
    """
    Projects student neck feature maps to teacher's channel dimensions.

    Why this is needed:
    -------------------
    YOLOv8n and YOLOv8m have different neck channel widths at the same FPN stage:
      YOLOv8n  [P3, P4, P5]:  [128, 256, 512]  channels
      YOLOv8m  [P3, P4, P5]:  [192, 384, 576]  channels (approximate)

    We cannot compute MSE between tensors with different channel counts.
    A 1×1 convolution (no spatial mixing) learns to project student channels
    into teacher channel space so the MSE comparison is valid.

    This adapter is training-only — it is NOT saved in the final student checkpoint.
    At inference, only student weights matter.
    """

    def __init__(self, student_channels: list, teacher_channels: list):
        """
        Args:
            student_channels : channel counts for student [P3, P4, P5], e.g. [128, 256, 512]
            teacher_channels : channel counts for teacher [P3, P4, P5], e.g. [192, 384, 576]
        """
        super().__init__()
        self.adapters = nn.ModuleList([
            nn.Conv2d(s_ch, t_ch, kernel_size=1, bias=False)
            for s_ch, t_ch in zip(student_channels, teacher_channels)
        ])

    def forward(self, feats: list) -> list:
        """
        Args:
            feats : list of student tensors [P3, P4, P5], each [B, student_C, H, W]
        Returns:
            list of projected tensors [P3, P4, P5], each [B, teacher_C, H, W]
        """
        return [adapter(feat) for adapter, feat in zip(self.adapters, feats)]


# =============================================================================
# Helper functions
# =============================================================================

def _get_neck_layer_indices(model) -> list:
    """
    Returns the absolute indices of the three neck output layers that feed the
    Detect head (P3, P4, P5 FPN outputs).

    Reads model.model[-1].f — the Detect head's registered source indices.
    Example: YOLOv8n has detect.f = [15, 18, 21] (22-layer backbone+neck).

    Negative indices are converted to absolute using the total layer count.
    The Detect head itself (last layer) is always excluded.

    DDP-safe: unwraps DistributedDataParallel if present.
    """
    raw_model = _unwrap_model(model)
    detect = raw_model.model[-1]
    n      = len(raw_model.model)   # total layers, including the Detect head

    f = detect.f
    if not isinstance(f, (list, tuple)):
        f = [f]

    abs_indices = []
    for idx in f:
        # Negative indices: convert using standard Python convention (relative to n)
        abs_idx = idx if idx >= 0 else (n + idx)
        if 0 <= abs_idx < n - 1:   # exclude Detect head itself
            abs_indices.append(abs_idx)

    return sorted(set(abs_indices))


def _measure_channels(model, layer_indices: list, device, imgsz: int = 640) -> list:
    """
    Runs a single dummy forward pass to measure feature channel counts at the
    specified layer indices. Returns [C_P3, C_P4, C_P5].

    Uses temporary hooks — does NOT reload the model from disk.
    Restores the model's original training/eval mode after measurement.
    """
    captured = {}
    handles  = []

    for idx in layer_indices:
        def make_hook(i):
            def hook(module, inp, output):
                feat = output[0] if isinstance(output, (list, tuple)) else output
                captured[i] = feat.shape[1]
            return hook
        handles.append(model.model[idx].register_forward_hook(make_hook(idx)))

    was_training = model.training
    model.eval()
    dtype = next(model.parameters()).dtype
    dummy = torch.zeros(1, 3, imgsz, imgsz, device=device, dtype=dtype)
    with torch.no_grad():
        model(dummy)

    for h in handles:
        h.remove()

    if was_training:
        model.train()

    return [captured.get(idx, 0) for idx in layer_indices]


# =============================================================================
# KDTrainer
# =============================================================================

class KDTrainer(DetectionTrainer):
    """
    Knowledge Distillation Trainer.

    Extends Ultralytics DetectionTrainer to:
      - Load and freeze the teacher model
      - Register persistent hooks on P3/P4/P5 neck layers of both models
      - Run teacher forward at each training step (preprocess_batch)
      - Compute combined loss: α·L_det + β·L_kd_logit + γ·L_feat

    All standard Ultralytics features are preserved:
      - Mosaic/flip/scale data augmentation
      - Mixed precision training (AMP)
      - Cosine LR scheduling with linear warmup
      - EMA model weights
      - Checkpoint saving (best.pt, last.pt)
      - TensorBoard / CSV / WandB logging
    """

    def __init__(
        self,
        teacher_path: str,
        alpha: float = 0.7,
        beta: float  = 0.2,
        gamma: float = 0.1,
        temperature: float = 4.0,
        cfg=None,
        overrides: dict = None,
        _callbacks=None,
    ):
        """
        Args:
            teacher_path  : path to the trained teacher .pt checkpoint
            alpha         : weight for the standard YOLO detection loss (default 0.7)
            beta          : weight for the logit KD loss / KL divergence (default 0.2)
            gamma         : weight for the feature KD loss / MSE (default 0.1)
                            Set to 0.0 to run logit-only KD (no ChannelAdapter needed)
            temperature   : temperature T for softening predictions (default 4.0)
            cfg           : Ultralytics config (leave None for defaults)
            overrides     : dict of training overrides — model, data, epochs, batch, etc.
        """
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

        self.teacher_path    = teacher_path
        self.kd_alpha        = alpha
        self.kd_beta         = beta
        self.kd_gamma        = gamma
        self.kd_temperature  = temperature

        # Initialized in _setup_kd() once device is ready
        self.teacher_model       = None
        self.feat_adapter        = None
        self._student_extractor  = None   # FeatureExtractor for student neck
        self._teacher_extractor  = None   # FeatureExtractor for teacher neck

        # Cached once in _setup_kd() for efficient per-batch BN freeze
        self._teacher_bn_layers  = []     # list of BN modules in teacher

        # Cached each batch in preprocess_batch(), consumed in criterion()
        self._teacher_logits     = None   # list[Tensor] — raw Detect head outputs
        self._teacher_neck_feats = None   # list[Tensor] — neck P3/P4/P5 features

        LOGGER.info(
            f"\n{'='*60}\n"
            f"  Knowledge Distillation Configuration\n"
            f"{'='*60}\n"
            f"  Teacher      : {teacher_path}\n"
            f"  alpha (det)  : {alpha}\n"
            f"  beta  (logit): {beta}\n"
            f"  gamma (feat) : {gamma}\n"
            f"  temperature  : {temperature}\n"
            f"{'='*60}\n"
        )

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    def _setup_kd(self):
        """
        Load teacher model, validate nc, register neck hooks, build ChannelAdapter.
        Called from _setup_train() after self.model and self.device are ready.
        """
        device = self.device
        imgsz  = self.args.imgsz if hasattr(self.args, 'imgsz') else 640

        # ── 1. Load teacher and freeze all weights ────────────────────────────
        LOGGER.info(f"Loading teacher model from: {self.teacher_path}")
        teacher_yolo = YOLO(self.teacher_path)
        self.teacher_model = teacher_yolo.model.to(device).float()

        for p in self.teacher_model.parameters():
            p.requires_grad = False

        LOGGER.info("Teacher model loaded, cast to float32, and frozen (requires_grad=False).")

        # ── 1b. Warn if teacher and student imgsz may differ ─────────────────
        teacher_imgsz = getattr(self.teacher_model, 'args', {}) or {}
        if hasattr(teacher_imgsz, 'get'):
            t_imgsz = teacher_imgsz.get('imgsz', None)
            if t_imgsz is not None and t_imgsz != imgsz:
                LOGGER.warning(
                    f"[KD] Teacher was trained with imgsz={t_imgsz} but student is "
                    f"using imgsz={imgsz}. Feature map sizes will differ, which "
                    f"degrades KD quality. Consider using the same imgsz for both."
                )

        # ── 2. Validate that both models use the same number of classes ───────
        teacher_nc = _unwrap_model(self.teacher_model).model[-1].nc
        student_nc = _unwrap_model(self.model).model[-1].nc
        if teacher_nc != student_nc:
            raise ValueError(
                f"[KD] Teacher nc={teacher_nc} != student nc={student_nc}.\n"
                f"     Both models must be trained on the same dataset with "
                f"the same class list.\n"
                f"     Check that your teacher checkpoint matches pcb_defect.yaml."
            )
        LOGGER.info(f"nc validated: teacher={teacher_nc}, student={student_nc} ✓")

        # ── 3. Resolve neck layer indices from Detect head's source list ──────
        student_indices = _get_neck_layer_indices(_unwrap_model(self.model))
        teacher_indices = _get_neck_layer_indices(self.teacher_model)
        LOGGER.info(f"Student neck layer indices (from detect.f): {student_indices}")
        LOGGER.info(f"Teacher neck layer indices (from detect.f): {teacher_indices}")

        if len(student_indices) != 3 or len(teacher_indices) != 3:
            LOGGER.warning(
                f"[KD] Expected exactly 3 neck indices per model; "
                f"got student={student_indices}, teacher={teacher_indices}.\n"
                f"     Feature KD disabled (gamma → 0.0)."
            )
            self.kd_gamma = 0.0

        # ── 4. Register persistent hooks and build ChannelAdapter ─────────────
        if self.kd_gamma > 0:
            LOGGER.info("Measuring neck feature channel counts via dummy forward pass...")
            student_channels = _measure_channels(_unwrap_model(self.model),  student_indices, device, imgsz)
            teacher_channels = _measure_channels(self.teacher_model,         teacher_indices, device, imgsz)

            LOGGER.info(f"  Student neck channels: {student_channels}")
            LOGGER.info(f"  Teacher neck channels: {teacher_channels}")

            if 0 in student_channels or 0 in teacher_channels:
                LOGGER.warning(
                    "[KD] Channel measurement failed for one or more layers. "
                    "Feature KD disabled (gamma → 0.0)."
                )
                self.kd_gamma = 0.0
            else:
                # ChannelAdapter: 1×1 convs projecting student → teacher channel space
                self.feat_adapter = ChannelAdapter(student_channels, teacher_channels).to(device)

                # Persistent hooks — fire on EVERY forward pass of each model
                self._student_extractor = FeatureExtractor(_unwrap_model(self.model).model, student_indices)
                self._teacher_extractor = FeatureExtractor(self.teacher_model.model,        teacher_indices)

                LOGGER.info(
                    f"ChannelAdapter created: {student_channels} → {teacher_channels}\n"
                    f"Persistent neck hooks registered on student and teacher."
                )

        # Cache teacher BN layers once — avoids iterating all modules every batch
        self._teacher_bn_layers = [
            m for m in self.teacher_model.modules()
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm))
        ]

    def _setup_train(self, world_size):
        """
        Called by Ultralytics before training starts.
        Sets up model, dataloaders, optimizer, then initialises KD components.
        """
        super()._setup_train(world_size)
        self._setup_kd()

        # Add ChannelAdapter parameters to the optimizer so they are trained
        if self.feat_adapter is not None and self.optimizer is not None:
            self.optimizer.add_param_group({
                'params'      : list(self.feat_adapter.parameters()),
                'lr'          : self.optimizer.param_groups[0]['lr'],
                'weight_decay': self.optimizer.param_groups[0].get('weight_decay', 0.0),
            })
            LOGGER.info("ChannelAdapter parameters added to optimizer.")

            # Patch LR scheduler to track the new param group.
            # The scheduler was created in super()._setup_train() before this
            # param group was added, so its internal base_lrs list is shorter
            # than the optimizer's param_groups. Without this patch, the adapter
            # would train at a constant LR with no warmup or cosine annealing.
            if self.scheduler is not None:
                new_base_lr = self.optimizer.param_groups[-1]['lr']
                self.scheduler.base_lrs.append(new_base_lr)
                # For LambdaLR or similar, extend the lambda list
                if hasattr(self.scheduler, 'lr_lambdas') and self.scheduler.lr_lambdas:
                    # Reuse the same lambda as the first param group
                    self.scheduler.lr_lambdas.append(self.scheduler.lr_lambdas[0])
                LOGGER.info("LR scheduler patched to include ChannelAdapter param group.")

    # -------------------------------------------------------------------------
    # Training step hooks
    # -------------------------------------------------------------------------

    def preprocess_batch(self, batch):
        """
        Runs the teacher forward pass and caches its logits and neck features.

        Called BEFORE the student forward pass each training step.
        Results are stored in self._teacher_logits and self._teacher_neck_feats
        and consumed in criterion() later the same step.

        Teacher mode:
          - model.train()  so the Detect head returns raw logits [B, 4*reg_max+nc, H, W]
            (in eval mode the Detect head post-processes predictions for inference)
          - All BatchNorm layers forced to eval() to use running statistics,
            not the current batch statistics (which would update BN state undesirably)
        """
        batch = super().preprocess_batch(batch)

        if self.teacher_model is None:
            return batch

        imgs = batch['img']     # [B, 3, H, W], float32, already normalised to [0,1]

        # Train mode → raw logit output from Detect head
        self.teacher_model.train()
        # Freeze BatchNorm running statistics (do NOT accumulate batch stats for teacher)
        # Uses cached list from _setup_kd() — avoids iterating all modules every batch
        for m in self._teacher_bn_layers:
            m.eval()

        with torch.no_grad():
            t_out = self.teacher_model(imgs)

        # Validate output format
        # In train mode, DetectionModel returns a list of 3 tensors [B, C, H, W]
        if not isinstance(t_out, (list, tuple)) or len(t_out) < 3:
            LOGGER.warning(
                f"[KD] Unexpected teacher output type: {type(t_out)} (len={len(t_out) if hasattr(t_out, '__len__') else 'N/A'}). "
                f"Expected list of 3 tensors from Detect head in train mode. "
                f"KD losses will be skipped for this batch."
            )
            self._teacher_logits     = None
            self._teacher_neck_feats = None
            return batch

        # Cache teacher Detect head outputs (raw logits, not post-processed)
        self._teacher_logits = [t.detach() for t in t_out[:3]]

        # Cache teacher neck features captured by persistent hooks
        if self._teacher_extractor is not None:
            self._teacher_neck_feats = [f.detach() for f in self._teacher_extractor.get_feats()]

        return batch

    def criterion(self, preds, batch):
        """
        Computes the combined KD training loss.

        Called by Ultralytics INSIDE the AMP autocast context (if amp=True).
        The student forward (which fires student neck hooks) has already run
        before this method is called.

        Args:
            preds : student Detect head output.
                    Either a list of 3 tensors [B, 4*reg_max+nc, H, W]
                    or a 2-tuple (y, feats) depending on Ultralytics version.
            batch : training batch dict

        Returns:
            (total_loss, loss_items) — Ultralytics expects exactly this format.
            loss_items is the unmodified detection sub-loss breakdown for logging.
        """
        # ── Step 1: Standard YOLO detection loss (box IoU + classification + DFL) ──
        loss_det, loss_items = super().criterion(preds, batch)

        if self.teacher_model is None or self._teacher_logits is None:
            return loss_det, loss_items

        # Unwrap preds: newer Ultralytics versions may return (y, feats) tuple
        if isinstance(preds, tuple):
            feats = preds[1]
        elif isinstance(preds, list):
            feats = preds
        else:
            LOGGER.warning(
                f"[KD] Unexpected preds type: {type(preds)}. "
                f"Expected list or tuple. KD losses skipped for this batch."
            )
            return loss_det, loss_items

        # Validate we have exactly 3 FPN scale outputs for KD
        if not isinstance(feats, list) or len(feats) < 3:
            LOGGER.warning(
                f"[KD] Expected feats to be a list of ≥3 tensors, got "
                f"{type(feats)} (len={len(feats) if hasattr(feats, '__len__') else 'N/A'}). "
                f"KD losses skipped for this batch."
            )
            return loss_det, loss_items

        kd_loss = torch.tensor(0.0, device=loss_det.device, dtype=loss_det.dtype)

        # ── Step 2: Logit KD loss (KL divergence on class channels only) ─────────
        if self.kd_beta > 0:
            # Detect head output: [B, 4*reg_max + nc, H, W]
            # Split into box regression and class score channels
            student_detect = _unwrap_model(self.model).model[-1]
            reg_max = student_detect.reg_max                    # 16 for all YOLOv8 variants
            nc      = student_detect.nc
            box_ch  = 4 * reg_max

            teacher_detect  = self.teacher_model.model[-1]
            teacher_reg_max = teacher_detect.reg_max
            teacher_nc      = teacher_detect.nc                 # same as nc (validated above)
            teacher_box_ch  = 4 * teacher_reg_max

            # Class-channel-only tensors: [B, nc, H, W] per scale
            student_cls = [p.split([box_ch, nc], dim=1)[1] for p in feats]
            teacher_cls = [
                t.split([teacher_box_ch, teacher_nc], dim=1)[1]
                for t in self._teacher_logits
            ]

            l_logit  = logit_kd_loss(student_cls, teacher_cls, self.kd_temperature)
            kd_loss  = kd_loss + self.kd_beta * l_logit

        # ── Step 3: Feature KD loss (MSE on L2-normalised neck features) ─────────
        if (self.kd_gamma > 0
                and self.feat_adapter       is not None
                and self._student_extractor is not None
                and self._teacher_neck_feats is not None):

            # Student neck features captured by persistent hooks during the student
            # forward pass that just ran (before this criterion call)
            student_neck_feats = self._student_extractor.get_feats()   # has grad — correct
            teacher_neck_feats = self._teacher_neck_feats               # detached — correct

            if len(student_neck_feats) == 3 and len(teacher_neck_feats) == 3:
                # Project student channels to teacher channel space
                student_adapted = self.feat_adapter(student_neck_feats)

                # Align spatial size if needed (same imgsz → same dims; check for safety)
                aligned_teacher = []
                for s_a, t_f in zip(student_adapted, teacher_neck_feats):
                    if s_a.shape[2:] != t_f.shape[2:]:
                        t_f = F.interpolate(
                            t_f, size=s_a.shape[2:], mode='bilinear', align_corners=False
                        )
                    aligned_teacher.append(t_f)

                l_feat  = feature_kd_loss(student_adapted, aligned_teacher)
                kd_loss = kd_loss + self.kd_gamma * l_feat

        # ── Step 4: Combine detection loss and KD losses ──────────────────────────
        # Ultralytics loss_det is intrinsically scaled by batch_size (sum over batch).
        # Our kd_loss is computed as a mean over the batch.
        # We MUST multiply kd_loss by batch_size so the alpha:beta:gamma ratio 
        # is preserved correctly regardless of the user's batch size setting.
        batch_size = feats[0].shape[0] if isinstance(feats, list) else 1
        kd_loss = kd_loss * batch_size
        
        total_loss = self.kd_alpha * loss_det + kd_loss

        return total_loss, loss_items

    # -------------------------------------------------------------------------
    # Checkpoint saving — detach hooks to avoid pickle errors
    # -------------------------------------------------------------------------

    def save_model(self):
        """
        Detach forward hooks before serialization to prevent pickle errors
        (closures are not picklable), then re-attach them immediately after.
        """
        # Remove hooks so deepcopy/pickle inside super().save_model() succeeds
        if self._student_extractor is not None:
            self._student_extractor.remove()
        if self._teacher_extractor is not None:
            self._teacher_extractor.remove()

        super().save_model()

        # Re-attach hooks for continued training
        if self._student_extractor is not None:
            self._student_extractor.register()
        if self._teacher_extractor is not None:
            self._teacher_extractor.register()

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def final_eval(self):
        """
        Called by Ultralytics at the end of training (before on_train_end callbacks).
        Removes persistent forward hooks so they don't interfere with final evaluation.
        """
        # Remove hooks before the final validation pass
        if self._student_extractor is not None:
            self._student_extractor.remove()
            LOGGER.info("Student neck hooks removed.")
        if self._teacher_extractor is not None:
            self._teacher_extractor.remove()
            LOGGER.info("Teacher neck hooks removed.")

        return super().final_eval()