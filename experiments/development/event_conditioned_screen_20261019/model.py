"""Four controlled continuous reconstruction arms. No Shape-200 head."""
import environment
import torch
from torch import nn
from torch.nn import functional as F
from src.models.continuous.ce_net import WSSTStem, _TCN


def regional_masks(probabilities):
    """QRS=max(Q,R,S), T=P_T; dilation then normalized Gaussian smoothing.

    At 200 Hz: symmetric QRS +/-50ms, T +/-100ms; Gaussian sigma10ms,
    support +/-30ms. Zero padding, no normalization by per-record maxima.
    """
    if probabilities.ndim != 3 or probabilities.shape[1] != 4:
        raise ValueError('Expected [B,4,T] probabilities')
    qrs = F.max_pool1d(probabilities[:, :3].amax(1, keepdim=True), 21, 1, 10)
    twave = F.max_pool1d(probabilities[:, 3:4], 41, 1, 20)
    x = torch.arange(-6, 7, device=probabilities.device, dtype=probabilities.dtype)
    kernel = torch.exp(-.5 * (x / 2).square())
    kernel = (kernel / kernel.sum()).reshape(1, 1, -1).repeat(2, 1, 1)
    return F.conv1d(torch.cat((qrs, twave), 1), kernel, padding=6, groups=2)


class Path(nn.Sequential):
    def __init__(self, width, kernel):
        super().__init__(nn.Conv1d(width, 64, kernel, padding=kernel//2, bias=False),
                         nn.BatchNorm1d(64), nn.GELU(),
                         nn.Conv1d(64, 1, kernel, padding=kernel//2))


class EventConditionedREM(nn.Module):
    def __init__(self, arm, width=128):
        super().__init__()
        if arm not in (1, 2, 3, 4):
            raise ValueError('arm must be 1..4')
        self.arm = arm
        self.stem = WSSTStem(in_bins=50, width=width, dropout=.1)
        self.backbone = _TCN(width, [1, 2, 4, 8, 16], dropout=.1)
        self.anchor_head = nn.Sequential(
            nn.Conv1d(width, 64, 5, padding=2, bias=False), nn.BatchNorm1d(64),
            nn.GELU(), nn.Conv1d(64, 4, 5, padding=2))
        self.global_path = Path(width, 5)
        if arm != 1:
            self.qrs_path = Path(width, 5)
            self.t_path = Path(width, 15)
        if arm == 2:
            self.feature_gate = nn.Conv1d(width, 4, 1)

    def decode(self, h, logits, oracle_events=None):
        if self.arm != 4 and oracle_events is not None:
            raise PermissionError('Deployable arms cannot accept GT conditioning')
        if self.arm == 4 and oracle_events is None:
            raise PermissionError('Oracle arm requires explicitly supplied GT events')
        wave = self.global_path(h)
        gates = None
        if self.arm != 1:
            if self.arm == 2:
                source = torch.sigmoid(self.feature_gate(h))
            elif self.arm == 3:
                source = torch.sigmoid(logits).detach()
            else:
                source = oracle_events.detach()
                if source.shape != logits.shape:
                    raise ValueError('Oracle events must be [B,4,1600]')
            gates = regional_masks(source)
            wave = wave + gates[:, :1] * self.qrs_path(h) + gates[:, 1:] * self.t_path(h)
        return {'wave': wave, 'anchor_logits': logits, 'gates': gates}

    def forward(self, wsst, oracle_events=None):
        if wsst.ndim != 4 or wsst.shape[1] != 50:
            raise ValueError('Expected MMECG WSST [B,50,F,T]')
        h = self.backbone(self.stem(wsst))
        h = F.interpolate(h, size=1600, mode='linear', align_corners=False)
        return self.decode(h, self.anchor_head(h), oracle_events)


def initialize_arm(arm, seed):
    """Stable per-module seeds ensure bitwise equal common initial tensors.

    Added branches never consume RNG before common modules are initialized.
    Training RNG is reset independently by the caller after initialization.
    """
    model = EventConditionedREM(arm)
    for name, module in model.named_children():
        offsets = {'stem': 1, 'backbone': 2, 'anchor_head': 3, 'global_path': 4,
                   'qrs_path': 5, 't_path': 6, 'feature_gate': 7}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + offsets[name] * 100003)
            for child in module.modules():
                if hasattr(child, 'reset_parameters'):
                    child.reset_parameters()
    return model


def parameter_counts(model):
    return {'total': sum(p.numel() for p in model.parameters()),
            'shared': sum(p.numel() for name, p in model.named_parameters()
                          if name.startswith(('stem.', 'backbone.', 'anchor_head.'))),
            'decoder': sum(p.numel() for name, p in model.named_parameters()
                           if not name.startswith(('stem.', 'backbone.', 'anchor_head.')))}


def objective(output, target, events):
    pred = output['wave']
    l1 = F.l1_loss(pred, target)
    a, b = pred-pred.mean(-1, keepdim=True), target-target.mean(-1, keepdim=True)
    pcc = (a*b).sum(-1) / (a.norm(dim=-1)*b.norm(dim=-1)).clamp_min(1e-8)
    correlation = 1-pcc.mean()
    # Native adjacent differences, NOT derivatives multiplied by fs.
    derivative = F.l1_loss(torch.diff(pred, dim=-1), torch.diff(target, dim=-1))
    anchor = F.binary_cross_entropy_with_logits(output['anchor_logits'], events)
    total = l1 + .5*correlation + derivative + .3*anchor
    return total, dict(loss=total, l1=l1, correlation=correlation,
                       derivative=derivative, anchor=anchor)


