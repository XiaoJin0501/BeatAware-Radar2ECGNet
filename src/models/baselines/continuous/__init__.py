"""Matched-protocol continuous-ECG baselines for the `loso_v2` benchmark.

Every model here reproduces a published radar-to-ECG method's defining
mechanism and is adapted -- never re-invented -- to the frozen continuous
contract of `docs/CONTINUOUS_ECG_TASK_CONTRACT.md`:

    forward(wsst, rcg=None) -> (waveform [B, 1, 1600], event [B, 2, 1600] | None)

The waveform is a LINEAR output in the train-fold robust-scaled target space, so
`scripts/train_continuous_ecg_pilot.py`'s evaluator can invert it to raw ECG
units exactly as it does for REM-Net. `event` is `None` for every baseline whose
paper has no event head; the training script then drops the event BCE term and
the metric table must report `event_f1_*` as not applicable.

Each adaptation is recorded in `ADAPTATIONS` and mirrored in the registered
experiment spec. Adaptations preserve the baseline's mechanism; they never
"improve" it, and they never give a baseline information REM-Net does not have.

See `docs/BASELINE_REPRODUCTION_POLICY.md`: these are the common-protocol
(`adapted`) lane, not original-protocol reproductions.
"""

from __future__ import annotations

from src.models.baselines.continuous.attention_dcgan import AttentionDCGANGenerator, DCGANDiscriminator
from src.models.baselines.continuous.cross_modal_conformer import CrossModalConformer
from src.models.baselines.continuous.mmecg_tcn import MMECGBaseline
from src.models.baselines.continuous.radarode_mtl import RadarODEMTLContinuous
from src.models.baselines.continuous.wavegru_net import WaveGRUNet

WIN = 1600
FS = 200

#: name -> (class, input key, paper)
BASELINES = {
    "mmecg_tcn": (MMECGBaseline, "rcg",
                  "Chen et al., Contactless ECG Monitoring With Millimeter Wave Radar, IEEE TMC 2024"),
    "radarode_mtl": (RadarODEMTLContinuous, "wsst",
                     "Zhang et al., radarODE-MTL, IEEE TIM 2025"),
    "wavegru_net": (WaveGRUNet, "rcg",
                    "Xu et al., WaveGRU-Net, Signal Processing 2025"),
    "attention_dcgan": (AttentionDCGANGenerator, "rcg",
                        "Jin et al., Attention-DCGAN, IEEE Sensors Journal 2025"),
    "cross_modal_conformer": (CrossModalConformer, "rcg",
                              "Liu et al., Cross-Modal Autoencoder, IEEE Sensors Journal 2024"),
}

#: Documented deviations from each paper, reproduced verbatim in the spec files.
ADAPTATIONS = {
    "mmecg_tcn": [
        "D1 input: the paper's 50 K-means voxel signals at length 640 are replaced by this "
        "dataset's 50 range-bin RCG at length 1600 (the same tensor every other baseline sees).",
        "D2 decoding: the paper's autoregressive 256-way mu-law softmax over previous ECG "
        "samples is replaced by a parallel continuous head. Autoregressive decoding of 1600 "
        "steps per window is computationally prohibitive at 11-fold scale, and the frozen "
        "contract scores continuous raw-ECG regression, not mu-law classes. The causal "
        "dilated TCN stack, the CNN-Transformer spatio-temporal encoder and the receptive "
        "field are kept.",
        "D3 position embedding: the paper embeds each voxel's 3D coordinates; range-bin index "
        "embeddings are used instead because the processed dataset exposes bins, not voxels.",
    ],
    "radarode_mtl": [
        "D1 input: WSST (50, 71, 240) covering the full 8 s window instead of the 4 s "
        "(50, 71, 120) tensor, matching the support REM-Net receives.",
        "D2 output: the published ECG_shape (200) / PPI (260) / anchor (800) heads cannot "
        "produce a continuous 1600-sample stream without the long-term assembly this project "
        "could not faithfully reproduce (claim ledger F2). The official DCNResNet + squeeze "
        "backbone is kept verbatim from the authors' repository and an adapted continuous "
        "decoder built from the authors' own DeconvBlock/ConvBlock stack emits 1x1600.",
        "D3 multitask: EGA gradient alignment across ECG_shape/PPI/anchor is not exercised, "
        "because only the continuous task is scored. Label the row 'adapted (continuous head)'.",
        "D4 decoder resolution: a third transposed stage takes the decoder to ~1600 samples "
        "natively. Two stages reach 549 and the remaining 3x would have to be interpolated, "
        "band-limiting the output below the QRS and handicapping the baseline.",
    ],
    "wavegru_net": [
        "D1 input: MIMO beamforming to one phase channel is replaced by this dataset's 50 "
        "range-bin RCG; MODWT is applied per bin.",
        "D2 transform: MODWT is realised as a fixed a-trous (undecimated) sym4 filter bank "
        "with 5 levels (6 bands) running inside the model on GPU, checked against pywt.swt. "
        "The paper's basis, level count and shift invariance are preserved.",
        "D3 length: the paper's 1x1024 slices become the contract's 1x1600 windows.",
        "D4 width: 50 bins x 6 bands would make the published 64-tap convolutions "
        "disproportionately wide, so a 1x1 band-mixing layer reduces 300 channels to 32 "
        "before the published 3-block CNN. The CNN, transposed-convolution module, Bi-GRU "
        "and dropout head keep the published structure.",
    ],
    "attention_dcgan": [
        "D1 segmentation: the paper feeds 200-sample fragments centred on ECG R peaks. Under "
        "the strict continuous contract that segmentation is a ground-truth oracle, so the "
        "generator maps the whole 8 s window instead.",
        "D2 output activation: the paper's Tanh output is kept and followed by a fixed gain "
        "fitted on the TRAINING fold only, because the contract's target lives in "
        "train-fold robust-scaled units rather than the paper's [-1, 1] range.",
        "D3 optimiser: the paper's Adam lr 3e-4 / batch 32 is replaced by the protocol budget "
        "(AdamW lr 1e-4, batch 16, 100 epochs, patience 15), identical to every other arm.",
        "D4 objective: an L1 reconstruction term (weight 10) accompanies the published purely "
        "adversarial generator loss, which carries no paired signal and does not converge on "
        "this task.",
        "D5 encoder: the paper's dense layer mixes a whole 200-sample beat before the "
        "deconvolutions. On continuous windows a stride-1 stack sees ~16 samples of context "
        "and cannot form a beat (measured: PCC ~0, variance ratio drifting past 1), so a "
        "four-stage strided encoder feeds the four published deconvolution layers, which now "
        "upsample back to 1600.",
    ],
    "cross_modal_conformer": [
        "D1 input: the paper's 1x1000 single-channel mmWave slice becomes the 50-bin RCG at "
        "length 1600.",
        "D2 pretraining: the ECG-to-ECG autoencoder and the cardiac-event predictor are "
        "pretrained per fold on TRAINING-fold ECG only (never validation or test), preserving "
        "the paper's deep-feature loss. Set --pretrain_epochs 0 to disable, which drops the "
        "deep-feature term and must then be reported as a removed component.",
        "D3 loss: waveform MSE + STFT-magnitude MSE, as published.",
        "D4 term scaling: the spectrogram and deep-feature terms are divided by their own "
        "target energy. Unnormalised, the spectrogram MSE dominates in robust-scaled units "
        "and, being sign-blind, drives the waveform to an inverted solution (measured: "
        "F1@100 0.92 with PCC -0.16). The published composition is unchanged.",
        "D5 decoder resolution: the convolutional subsampling divides time by 8, so the "
        "decoder mirrors it with transposed convolutions. Synthesising the ECG by "
        "interpolating the 25 Hz latent would band-limit the output below the QRS.",
    ],
}


def build_baseline(name: str, **kwargs):
    """Instantiate a registered baseline by name."""
    if name not in BASELINES:
        raise ValueError(f"unknown baseline {name!r}; registered: {sorted(BASELINES)}")
    cls, _, _ = BASELINES[name]
    return cls(**kwargs)


def input_key(name: str) -> str:
    """Which fold tensor the baseline consumes: 'rcg' (50, 1600) or 'wsst' (50, 71, 240)."""
    if name not in BASELINES:
        raise ValueError(f"unknown baseline {name!r}")
    return BASELINES[name][1]


__all__ = [
    "BASELINES", "ADAPTATIONS", "build_baseline", "input_key",
    "MMECGBaseline", "RadarODEMTLContinuous", "WaveGRUNet",
    "AttentionDCGANGenerator", "DCGANDiscriminator", "CrossModalConformer",
]
