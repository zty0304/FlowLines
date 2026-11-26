import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from datasets.hair_dataset import HairStructureDataset
from models.unet import HairStructUNet
from models.losses import HairStructureLoss
from utils.common import ensure_dir, load_yaml, save_checkpoint, set_seed
from utils.visualization import orientation_to_rgb, save_mask, save_rgb


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_center = 0.0
    total_orient = 0.0
    total_bg = 0.0
    total_smooth = 0.0
    n = 0

    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            pred = model(batch['input'])
            loss, stats = criterion(pred, batch)

            total_loss += float(loss.detach().cpu())
            total_center += float(stats.get("center_loss", 0.0))
            total_orient += float(stats.get("orient_loss", 0.0))
            total_bg += float(stats.get("bg_loss", 0.0))
            total_smooth += float(stats.get("smooth_loss", 0.0))
            n += 1

    n = max(n, 1)
    return {
        "loss": total_loss / n,
        "center_loss": total_center / n,
        "orient_loss": total_orient / n,
        "bg_loss": total_bg / n,
        "smooth_loss": total_smooth / n,
    }


def save_preview(model, batch, output_dir, step_name):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        pred = model(batch['input'])

    image = batch['image'][0].permute(1, 2, 0).detach().cpu().numpy()                 # H,W,3
    mask = batch['mask'][0, 0].detach().cpu().numpy()                                  # H,W
    center_gt = batch['centerline_gt'][0, 0].detach().cpu().numpy()                    # H,W
    center_pred = pred['center'][0, 0].detach().cpu().numpy()                          # H,W
    orient_gt = batch['orientation_gt'][0].permute(1, 2, 0).detach().cpu().numpy()     # H,W,2
    orient_pred = pred['orient'][0].permute(1, 2, 0).detach().cpu().numpy()            # H,W,2

    center_pred = np.clip(center_pred, 0.0, 1.0)

    # 1) 原图 / mask / GT / 概率图
    save_rgb((image * 255).astype(np.uint8), os.path.join(output_dir, f'{step_name}_image.png'))
    save_mask(mask, os.path.join(output_dir, f'{step_name}_mask.png'))
    save_mask(center_gt, os.path.join(output_dir, f'{step_name}_center_gt.png'))
    save_mask(center_pred, os.path.join(output_dir, f'{step_name}_center_pred_prob.png'))

    # 2) 白底黑线版本（更接近你的 line 视觉习惯）
    center_gt_inv = (1.0 - np.clip(center_gt, 0.0, 1.0)) * 255.0
    center_pred_inv = (1.0 - center_pred) * 255.0

    save_rgb(
        np.stack([center_gt_inv, center_gt_inv, center_gt_inv], axis=-1).astype(np.uint8),
        os.path.join(output_dir, f'{step_name}_center_gt_whitebg.png')
    )
    save_rgb(
        np.stack([center_pred_inv, center_pred_inv, center_pred_inv], axis=-1).astype(np.uint8),
        os.path.join(output_dir, f'{step_name}_center_pred_whitebg.png')
    )

    # 3) 方向场可视化
    save_rgb(
        orientation_to_rgb(orient_gt, center_gt),
        os.path.join(output_dir, f'{step_name}_orient_gt.png')
    )
    save_rgb(
        orientation_to_rgb(orient_pred, center_pred),
        os.path.join(output_dir, f'{step_name}_orient_pred.png')
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg['seed'])

    device = cfg['device'] if (cfg['device'] == 'cpu' or torch.cuda.is_available()) else 'cpu'
    use_cuda = device.startswith('cuda')

    ensure_dir(cfg['output_dir'])
    ensure_dir(os.path.join(cfg['output_dir'], 'checkpoints'))
    ensure_dir(os.path.join(cfg['output_dir'], 'previews'))

    train_ds = HairStructureDataset(cfg['train_dir'], augment=cfg.get('augment', False))
    val_ds = HairStructureDataset(cfg['val_dir'], augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=use_cuda,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg['num_workers'],
        pin_memory=use_cuda,
        drop_last=False,
    )

    model = HairStructUNet(in_channels=4, base_ch=32, out_channels=3).to(device)

    criterion = HairStructureLoss(
        lambda_center=cfg['lambda_center'],
        lambda_orient=cfg['lambda_orient'],
        lambda_bg=cfg['lambda_bg'],
        lambda_smooth=cfg['lambda_smooth'],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['epochs']
    )

    start_epoch = 0
    best_val = float('inf')

    if cfg.get('resume'):
        print(f"Resuming from: {cfg['resume']}")
        ckpt = torch.load(cfg['resume'], map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_val = ckpt.get('best_val', best_val)

    for epoch in range(start_epoch, cfg['epochs']):
        model.train()
        running_loss = 0.0
        running_center = 0.0
        running_orient = 0.0
        running_bg = 0.0
        running_smooth = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{cfg["epochs"]}')
        for batch in pbar:
            batch = to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(batch['input'])
            loss, stats = criterion(pred, batch)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu())
            running_center += float(stats.get("center_loss", 0.0))
            running_orient += float(stats.get("orient_loss", 0.0))
            running_bg += float(stats.get("bg_loss", 0.0))
            running_smooth += float(stats.get("smooth_loss", 0.0))

            pbar.set_postfix(
                loss=f'{stats["loss"]:.4f}',
                center=f'{stats["center_loss"]:.4f}',
                orient=f'{stats["orient_loss"]:.4f}'
            )

        scheduler.step()

        train_n = max(len(train_loader), 1)
        train_stats = {
            "loss": running_loss / train_n,
            "center_loss": running_center / train_n,
            "orient_loss": running_orient / train_n,
            "bg_loss": running_bg / train_n,
            "smooth_loss": running_smooth / train_n,
        }

        val_stats = validate(model, val_loader, criterion, device)

        print(
            f'Epoch {epoch + 1}: '
            f'train={train_stats["loss"]:.4f}, '
            f'val={val_stats["loss"]:.4f} | '
            f'train_center={train_stats["center_loss"]:.4f}, '
            f'val_center={val_stats["center_loss"]:.4f} | '
            f'train_orient={train_stats["orient_loss"]:.4f}, '
            f'val_orient={val_stats["orient_loss"]:.4f}'
        )

        # 保存 preview
        preview_batch = next(iter(val_loader))
        preview_batch = to_device(preview_batch, device)
        save_preview(
            model,
            preview_batch,
            os.path.join(cfg['output_dir'], 'previews'),
            f'epoch_{epoch + 1:03d}'
        )

        # 保存 last
        last_ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val': best_val,
            'config': cfg,
        }
        save_checkpoint(last_ckpt, os.path.join(cfg['output_dir'], 'checkpoints', 'last.pt'))

        # 定期保存
        if (epoch + 1) % cfg['save_every'] == 0:
            save_checkpoint(
                last_ckpt,
                os.path.join(cfg['output_dir'], 'checkpoints', f'epoch_{epoch + 1:03d}.pt')
            )

        # 保存 best
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            best_ckpt = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val': best_val,
                'config': cfg,
            }
            save_checkpoint(best_ckpt, os.path.join(cfg['output_dir'], 'checkpoints', 'best.pt'))
            print(f'Saved new best model: {best_val:.4f}')


if __name__ == '__main__':
    main()