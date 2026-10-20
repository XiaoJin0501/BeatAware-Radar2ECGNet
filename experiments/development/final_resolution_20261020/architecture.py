"""Last bounded design: v2 arm1, optional native-resolution fusion/soft-min loss."""
import bridge
import torch
from torch import nn
from torch.nn import functional as F


class FinalREM(nn.Module):
    def __init__(self, variant, seed=42, in_bins=50):
        super().__init__()
        if variant not in ('A', 'B', 'C', 'D'):
            raise ValueError(variant)
        self.variant = variant
        base = bridge.old_model.initialize_arm(1, seed)
        for name, child in base.named_children():
            setattr(self, name, child)
        if in_bins != 50:
            # CR_RVS support; not part of the MMECG screening queue.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+100003)
                self.stem = bridge.old_model.WSSTStem(in_bins=in_bins, width=128, dropout=.1)
        if variant in ('C', 'D'):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+800024)
                layers = []
                widths = (in_bins, 32, 32, 32, 32, 128)
                for i, dilation in enumerate((1, 2, 4, 8, 16)):
                    layers.extend((nn.Conv1d(widths[i], widths[i+1], 7,
                        padding=3*dilation, dilation=dilation, bias=False),
                        nn.BatchNorm1d(widths[i+1]), nn.GELU()))
                self.time_path = nn.Sequential(*layers)
                self.time_projection = nn.Conv1d(128, 128, 1, bias=False)
                self.fusion_gate = nn.Conv1d(128, 128, 1)

    def forward(self, wsst, td=None):
        h_tf = self.backbone(self.stem(wsst))
        h_tf = F.interpolate(h_tf, size=1600, mode='linear', align_corners=False)
        logits = self.anchor_head(h_tf)  # The existing event pathway is unchanged.
        h = h_tf
        if self.variant in ('C', 'D'):
            if td is None or td.shape != (len(wsst), wsst.shape[1], 1600):
                raise ValueError('Strictly aligned native RCG required')
            h_td = self.time_path(td)
            h = h_tf + torch.sigmoid(self.fusion_gate(h_tf))*self.time_projection(h_td)
        return dict(wave=self.global_path(h), anchor_logits=logits)


def per_sample_loss(pred, target):
    a, b = pred-pred.mean(-1, keepdim=True), target-target.mean(-1, keepdim=True)
    pcc = (a*b).sum(-1)/(a.norm(dim=-1)*b.norm(dim=-1)).clamp_min(1e-8)
    l1 = (pred-target).abs().mean(dim=(-1, -2))
    corr = (1-pcc).mean(-1)
    deriv = (torch.diff(pred, dim=-1)-torch.diff(target, dim=-1)).abs().mean(dim=(-1, -2))
    return l1+.5*corr+deriv


def waveform_loss(pred, target, tolerant):
    if not tolerant:
        return per_sample_loss(pred, target).mean()
    # All 41 shifts share 1560 samples. No padding, wrapping or variable support.
    # Positive tau selects a later reference sample; event labels are NEVER shifted.
    reference_shifts = target.unfold(-1, 1560, 1).transpose(1, 2)
    p = pred[..., 20:1580].unsqueeze(1).expand(-1, 41, -1, -1)
    losses = per_sample_loss(p.flatten(0, 1), reference_shifts.flatten(0, 1)).reshape(len(pred), 41)
    return (-.1*torch.logsumexp(-losses/.1, dim=1)).mean()


def objective(output, target, events, variant):
    if variant in ('A', 'C'):
        # Exact existing ordinary objective, including numerical reductions.
        return bridge.old_model.objective(output, target, events)
    wave = waveform_loss(output['wave'], target, tolerant=True)
    anchor = F.binary_cross_entropy_with_logits(output['anchor_logits'], events)
    total = wave+.3*anchor
    return total, dict(loss=total, waveform=wave, anchor=anchor)


def counts(model):
    shared = sum(p.numel() for n, p in model.named_parameters() if n.startswith(('stem.', 'backbone.')))
    added = sum(p.numel() for n, p in model.named_parameters() if n.startswith(('time_path.', 'time_projection.', 'fusion_gate.')))
    return dict(total=sum(p.numel() for p in model.parameters()), backbone=shared,
                added=added, added_fraction_of_backbone=added/shared)
