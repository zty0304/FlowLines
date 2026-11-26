import torch
import torch.nn as nn
import torch.nn.functional as F


class HairStructureLoss(nn.Module):
    def __init__(
        self,
        lambda_center=1.0,
        lambda_orient=0.5,
        lambda_bg=0.2,
        lambda_smooth=0.05,
        lambda_dice=1.0,
    ):
        super().__init__()
        self.lambda_center = lambda_center
        self.lambda_orient = lambda_orient
        self.lambda_bg = lambda_bg
        self.lambda_smooth = lambda_smooth
        self.lambda_dice = lambda_dice

        self.bce = nn.BCELoss(reduction='none')

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        """
        pred, target: [B,1,H,W], in [0,1]
        weight: [B,1,H,W] or None
        """
        pred = pred.contiguous()
        target = target.contiguous()

        if weight is not None:
            pred = pred * weight
            target = target * weight

        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        return 1.0 - dice.mean()

    def forward(self, pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        center_pred = pred['center']                # [B,1,H,W], sigmoid output
        orient_pred = pred['orient']                # [B,2,H,W]
        center_gt = batch['centerline_gt']          # [B,1,H,W]
        orient_gt = batch['orientation_gt']         # [B,2,H,W]
        mask = batch['mask']                        # [B,1,H,W]

        # -----------------------------
        # 1) center loss = masked BCE + Dice
        # -----------------------------
        bce_map = self.bce(center_pred, center_gt)
        center_bce = (bce_map * mask).sum() / (mask.sum() + 1e-6)

        center_dice = self.dice_loss(center_pred, center_gt, weight=mask)
        center_loss = center_bce + self.lambda_dice * center_dice

        # -----------------------------
        # 2) background suppression
        # -----------------------------
        bg_loss = (center_pred * (1.0 - mask)).mean()

        # -----------------------------
        # 3) orientation loss
        # only supervise where center GT exists
        # normalize pred/gt before cosine similarity
        # -----------------------------
        orient_weight = ((center_gt > 0.05).float() * mask)   # [B,1,H,W]

        orient_pred_norm = F.normalize(orient_pred, dim=1, eps=1e-6)
        orient_gt_norm = F.normalize(orient_gt, dim=1, eps=1e-6)

        cos_sim = (orient_pred_norm * orient_gt_norm).sum(dim=1, keepdim=True)  # [B,1,H,W]
        orient_loss_map = (1.0 - cos_sim) * orient_weight
        orient_loss = orient_loss_map.sum() / (orient_weight.sum() + 1e-6)

        # -----------------------------
        # 4) smoothness loss
        # only enforce inside hair mask
        # -----------------------------
        dx = orient_pred_norm[:, :, :, 1:] - orient_pred_norm[:, :, :, :-1]
        dy = orient_pred_norm[:, :, 1:, :] - orient_pred_norm[:, :, :-1, :]

        mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]

        smooth_x = (dx.abs().mean(dim=1, keepdim=True) * mask_x).sum() / (mask_x.sum() + 1e-6)
        smooth_y = (dy.abs().mean(dim=1, keepdim=True) * mask_y).sum() / (mask_y.sum() + 1e-6)
        smooth_loss = smooth_x + smooth_y

        # -----------------------------
        # total
        # -----------------------------
        total = (
            self.lambda_center * center_loss
            + self.lambda_orient * orient_loss
            + self.lambda_bg * bg_loss
            + self.lambda_smooth * smooth_loss
        )

        stats = {
            'loss': float(total.detach().cpu()),
            'center_loss': float(center_loss.detach().cpu()),
            'center_bce': float(center_bce.detach().cpu()),
            'center_dice': float(center_dice.detach().cpu()),
            'orient_loss': float(orient_loss.detach().cpu()),
            'bg_loss': float(bg_loss.detach().cpu()),
            'smooth_loss': float(smooth_loss.detach().cpu()),
        }
        return total, stats