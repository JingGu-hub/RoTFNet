import torch
import torch.nn.functional as F
import numpy as np

from utils.utils import get_clean_loss_tensor_mask, get_clean_mask

def compute_delay_loss(args, out, y, k_losses, inds, epoch):
    loss = F.cross_entropy(out, y, reduction='none').cuda()

    k_losses[inds, epoch % args.delay_loss_k] = loss.detach().cpu().numpy()
    if epoch >= args.start_delay_loss:
        t = torch.tensor(data=np.sum(k_losses, axis=1) / k_losses.shape[1], dtype=torch.float32, requires_grad=True, device=loss.device)
        loss = 0.01 * loss + t[inds]

    return loss, k_losses

def delay_loss(args, outs, y, k_losses, epoch, loss_all, inds, update_inds):
    """Compute the sum of loss for each k in k_list."""
    out1, out2, out3 = outs[0], outs[1], outs[2]
    k_loss1, k_loss2, k_loss3 = k_losses[0], k_losses[1], k_losses[2]

    delay_loss1, k_loss1 = compute_delay_loss(args, out1, y, k_loss1, inds, epoch)
    delay_loss2, k_loss2 = compute_delay_loss(args, out2, y, k_loss2, inds, epoch)
    delay_loss3, k_loss3 = compute_delay_loss(args, out3, y, k_loss3, inds, epoch)

    losses_total = delay_loss1 + delay_loss2 + delay_loss3
    loss_all[inds] = losses_total.detach().cpu().numpy()

    if epoch >= args.start_mask_epoch:
        remember_rate = 1 - args.label_noise_rate
        if epoch >= args.start_refurb:
            final_mask = get_clean_mask(losses_total, inds, update_inds, remember_rate=remember_rate)
        else:
            final_mask = get_clean_loss_tensor_mask(losses_total, remember_rate=remember_rate)
    else:
        final_mask = torch.ones_like(losses_total)

    final_loss1 = torch.sum(final_mask * delay_loss1) / max(torch.count_nonzero(final_mask).item(), 1)
    final_loss2 = torch.sum(final_mask * delay_loss2) / max(torch.count_nonzero(final_mask).item(), 1)
    final_loss3 = torch.sum(final_mask * delay_loss3) / max(torch.count_nonzero(final_mask).item(), 1)

    return final_loss1, final_loss2, final_loss3, k_loss1, k_loss2, k_loss3, loss_all

def freq_aug_loss(mse_criterion, data, freq_aug_data):
    """Compute the frequency augmentation loss."""
    mse_loss = [mse_criterion(data, freq_aug_data[i]) for i in range(len(freq_aug_data))]
    scale_freq_loss = torch.stack(mse_loss)
    scale_freq_loss = torch.mean(scale_freq_loss)

    return scale_freq_loss