"""
mmecg_dataset.py — MMECG H5-based Dataset Loader

直接读取 /home/qhh2237/Datasets/MMECG/processed/ 下的预构建 H5 文件，
支持 LOSO（11 折）和 Samplewise 两种协议，返回真正的三路 DataLoader。

Usage:
    from src.data.mmecg_dataset import build_loso_loaders_h5, build_samplewise_loaders_h5

    # LOSO（fold_idx 1-based，Fold 1~11）
    train_loader, val_loader, test_loader = build_loso_loaders_h5(
        fold_idx=1,
        loso_dir="/home/qhh2237/Datasets/MMECG/processed/loso",
    )

    # Samplewise
    train_loader, val_loader, test_loader = build_samplewise_loaders_h5(
        sw_dir="/home/qhh2237/Datasets/MMECG/processed/samplewise",
    )

H5 文件每个 split 包含的 dataset keys：
    rcg               float32 [N, 50, 1600]  per-channel z-score
    ecg               float32 [N,  1, 1600]  z-score（loader 内再 min-max → [0,1]）
    rpeak_indices     vlen int32              R 峰位置 → Gaussian mask
    q_indices         vlen int32  (-1=missing)
    s_indices         vlen int32
    tpeak_indices     vlen int32
    delineation_valid uint8 [N]
    subject_id        int32 [N]
    physistatus       bytes [N]   b"NB"/b"IB"/b"SP"/b"PE"
"""

from pathlib import Path

import h5py
import numpy as np
import scipy.signal
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, WeightedRandomSampler

# ── 상수 ──────────────────────────────────────────────────────────────────────
PHYSISTATUS_MAP: dict[str, int] = {"NB": 0, "IB": 1, "SP": 2, "PE": 3}
RPEAK_SIGMA_DEFAULT = 5


# ── 내부 도구 함수 ──────────────────────────────────────────────────────────────

def _gaussian_mask(indices: np.ndarray, length: int, sigma: float = RPEAK_SIGMA_DEFAULT) -> np.ndarray:
    """vlen rpeak_indices (int32) → Gaussian soft label [length] float32."""
    mask = np.zeros(length, dtype=np.float32)
    for idx in indices:
        idx = int(idx)
        if idx < 0:
            continue
        lo = max(0, idx - 3 * int(sigma))
        hi = min(length, idx + 3 * int(sigma) + 1)
        t = np.arange(lo, hi)
        mask[lo:hi] += np.exp(-0.5 * ((t - idx) / sigma) ** 2)
    return np.clip(mask, 0.0, 1.0)


def _minmax_ecg(ecg_win: np.ndarray) -> np.ndarray:
    """Per-window min-max normalization → [0, 1]. Handles flat signal."""
    lo, hi = float(ecg_win.min()), float(ecg_win.max())
    if hi - lo < 1e-8:
        return np.zeros_like(ecg_win, dtype=np.float32)
    return ((ecg_win - lo) / (hi - lo)).astype(np.float32)


def _decode_physistatus(raw) -> str:
    if isinstance(raw, bytes):
        return raw.decode()
    return str(raw)


def _band_energy_per_bin(
    rcg_group: np.ndarray, fs: float, lo: float, hi: float, order: int = 4,
) -> np.ndarray:
    """rcg_group: (M, R, L) → (R,) mean band-energy per range bin.

    Filter each (M, R, L) tensor in the heart band, square, mean over time and
    over the M windows of one (subject, scene) recording group.
    """
    b, a = scipy.signal.butter(order, [lo, hi], btype="band", fs=fs)
    y = scipy.signal.filtfilt(b, a, rcg_group, axis=-1)
    return (y ** 2).mean(axis=(0, -1)).astype(np.float32)   # (R,)


def _oracle_corr_per_bin(
    rcg_group: np.ndarray, ecg_group: np.ndarray,
) -> np.ndarray:
    """rcg_group:(M, R, L), ecg_group:(M, 1, L) → (R,) mean |Pearson(bin, ECG)|.

    Per-window |corr|, averaged over the M windows of one (subject, scene) group.
    Uses ECG GT — ONLY use as deployable upper-bound, NOT for LOSO-test inference.
    """
    rc = rcg_group - rcg_group.mean(axis=-1, keepdims=True)             # (M, R, L)
    ec = ecg_group[:, 0, :] - ecg_group[:, 0, :].mean(axis=-1, keepdims=True)  # (M, L)
    num = (rc * ec[:, None, :]).sum(axis=-1)                            # (M, R)
    den = np.sqrt(
        (rc ** 2).sum(axis=-1) * (ec ** 2).sum(axis=-1)[:, None] + 1e-12
    )
    corr = np.abs(num / den)                                            # (M, R)
    return corr.mean(axis=0).astype(np.float32)                          # (R,)


# ── 커스텀 collate（vlen peak 인덱스용）──────────────────────────────────────────

def _collate_with_peaks(batch: list[dict]) -> dict:
    """
    test DataLoader 전용 collate_fn.
    Tensor 배열은 stack, int 스칼라는 LongTensor, vlen peak 인덱스는 list 유지.
    """
    keys_tensor = {"radar", "ecg", "rpeak", "delin_valid"}
    keys_int    = {"subject", "state"}
    keys_list   = {"r_idx", "q_idx", "s_idx", "t_idx"}

    result: dict = {}
    for k in keys_tensor:
        if k in batch[0]:
            result[k] = torch.stack([b[k] for b in batch])
    for k in keys_int:
        if k in batch[0]:
            result[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)
    for k in keys_list:
        if k in batch[0]:
            result[k] = [b[k] for b in batch]
    return result


# ── Dataset ──────────────────────────────────────────────────────────────────

class MMECGWindowedH5Dataset(Dataset):
    """
    단일 H5 파일에서 윈도우 데이터를 로드하는 Dataset.

    Parameters
    ----------
    h5_path : str | Path
        H5 파일 경로（train.h5 / val.h5 / test.h5）
    include_peak_indices : bool
        True = q/s/t/r 인덱스 및 delineation_valid 도 포함（test set용）
    rpeak_sigma : float
        Gaussian 소프트 레이블의 sigma（samples）
    """

    def __init__(
        self,
        h5_path: str | Path,
        include_peak_indices: bool = False,
        rpeak_sigma: float = RPEAK_SIGMA_DEFAULT,
        narrow_bandpass: bool = False,
        bp_lo: float = 0.8,
        bp_hi: float = 3.5,
        fs: float = 200.0,
        topk_bins: int | None = None,
        target_norm: str = "minmax",
        topk_method: str = "energy",
    ):
        super().__init__()
        h5_path = Path(h5_path)
        if not h5_path.exists():
            raise FileNotFoundError(f"H5 file not found: {h5_path}")
        if topk_method not in ("energy", "corr"):
            raise ValueError(f"Unknown topk_method: {topk_method!r}")

        self.include_peak_indices = include_peak_indices
        self.topk_bins = topk_bins
        self.target_norm = target_norm
        self.topk_method = topk_method

        with h5py.File(h5_path, "r") as hf:
            N = hf["rcg"].shape[0]

            # ── 고정 크기 배열 ──────────────────────────────────────────
            rcg_raw = hf["rcg"][:]                          # [N, 50, 1600] float32，z-score 완료
            ecg_raw = hf["ecg"][:]                          # [N,  1, 1600] float32，z-score 완료
            subj    = hf["subject_id"][:].astype(np.int32)  # [N]
            phys_b  = hf["physistatus"][:]                  # [N] bytes

            # ── 심박 협대역 필터（0.8-3.5 Hz）──────────────────────────
            if narrow_bandpass:
                b, a = scipy.signal.butter(4, [bp_lo, bp_hi], btype="band", fs=fs)
                rcg_raw = np.stack(
                    [scipy.signal.filtfilt(b, a, rcg_raw[i], axis=-1) for i in range(N)]
                )
                # re-normalize channel-wise per window
                mu  = rcg_raw.mean(axis=-1, keepdims=True)
                std = rcg_raw.std( axis=-1, keepdims=True) + 1e-8
                rcg_raw = ((rcg_raw - mu) / std).astype(np.float32)

            # ── physistatus → int (used by topk grouping below) ──────
            state_int = np.array(
                [PHYSISTATUS_MAP.get(_decode_physistatus(p), 0) for p in phys_b],
                dtype=np.int32,
            )

            # ── per-(subject, scene) top-K range-bin selection ──────
            #   selection criterion: 0.8-3.5 Hz heart-band energy per bin,
            #   averaged over all windows of each (subject, scene) group.
            #   stays deployable: uses only input RCG, no ECG GT.
            self.topk_indices: dict[tuple[int, int], np.ndarray] = {}
            if topk_bins is not None and topk_bins < rcg_raw.shape[1]:
                groups = sorted(set(zip(subj.tolist(), state_int.tolist())))
                rcg_sliced = np.empty((N, topk_bins, rcg_raw.shape[2]), dtype=np.float32)
                for s_id, st in groups:
                    mask = (subj == s_id) & (state_int == st)
                    if topk_method == "energy":
                        scores = _band_energy_per_bin(
                            rcg_raw[mask], fs=fs, lo=bp_lo, hi=bp_hi,
                        )
                    else:  # "corr"
                        scores = _oracle_corr_per_bin(
                            rcg_raw[mask], ecg_raw[mask],
                        )
                    top_idx = np.argsort(scores)[-topk_bins:][::-1].astype(np.int32)
                    self.topk_indices[(int(s_id), int(st))] = top_idx
                    rcg_sliced[mask] = rcg_raw[mask][:, top_idx, :]
                rcg_raw = rcg_sliced

            # ── ECG 정규화（target_norm='minmax' → [0,1]; 'zscore' → 그대로）
            if target_norm == "minmax":
                ecg_norm = np.stack(
                    [_minmax_ecg(ecg_raw[i]) for i in range(N)], axis=0
                )  # [N, 1, 1600]
            elif target_norm == "zscore":
                ecg_norm = ecg_raw.astype(np.float32)
            else:
                raise ValueError(f"Unknown target_norm: {target_norm}")

            # ── R 피크 → Gaussian 마스크 ──────────────────────────────
            rpeak_ds = hf["rpeak_indices"]
            rpeak = np.stack(
                [_gaussian_mask(rpeak_ds[i], 1600, sigma=rpeak_sigma)[np.newaxis, :]
                 for i in range(N)],
                axis=0,
            )  # [N, 1, 1600]

            # state already computed above for topk grouping
            state = state_int

            # ── 선택적：peak 인덱스（test set용）─────────────────────
            if include_peak_indices:
                r_ds = hf["rpeak_indices"]
                q_ds = hf["q_indices"]
                s_ds = hf["s_indices"]
                t_ds = hf["tpeak_indices"]
                self._r_idx = [r_ds[i].astype(np.int32) for i in range(N)]
                self._q_idx = [q_ds[i].astype(np.int32) for i in range(N)]
                self._s_idx = [s_ds[i].astype(np.int32) for i in range(N)]
                self._t_idx = [t_ds[i].astype(np.int32) for i in range(N)]
                self._delin = hf["delineation_valid"][:].astype(np.uint8)

        self._rcg   = rcg_raw
        self._ecg   = ecg_norm
        self._rpeak = rpeak
        self._subj  = subj
        self._state = state
        self._N     = N

    def __len__(self) -> int:
        return self._N

    def __getitem__(self, idx: int) -> dict:
        item = {
            "radar":   torch.from_numpy(self._rcg[idx]),    # [50, 1600]
            "ecg":     torch.from_numpy(self._ecg[idx]),    # [ 1, 1600]
            "rpeak":   torch.from_numpy(self._rpeak[idx]),  # [ 1, 1600]
            "subject": int(self._subj[idx]),
            "state":   int(self._state[idx]),
        }
        if self.include_peak_indices:
            item["r_idx"]      = self._r_idx[idx]
            item["q_idx"]      = self._q_idx[idx]
            item["s_idx"]      = self._s_idx[idx]
            item["t_idx"]      = self._t_idx[idx]
            item["delin_valid"] = torch.tensor(bool(self._delin[idx]))
        return item

    def class_weights(self) -> torch.Tensor:
        """
        physistatus 클래스 역빈도 가중치 → WeightedRandomSampler용.
        samplewise split에서 클래스 균형을 보장.
        """
        unique, counts = np.unique(self._state, return_counts=True)
        inv = 1.0 / counts.astype(np.float32)
        cls2w = dict(zip(unique.tolist(), inv.tolist()))
        weights = np.array([cls2w[s] for s in self._state], dtype=np.float32)
        return torch.from_numpy(weights)

    def subject_weights(self) -> torch.Tensor:
        """
        수험자 역빈도 가중치 → WeightedRandomSampler용.
        Sub 1/2처럼 훈련 샘플이 극단적으로 적은 수험자의 학습을 보장.
        """
        unique, counts = np.unique(self._subj, return_counts=True)
        inv = 1.0 / counts.astype(np.float32)
        sub2w = dict(zip(unique.tolist(), inv.tolist()))
        weights = np.array([sub2w[s] for s in self._subj], dtype=np.float32)
        return torch.from_numpy(weights)


# ── 팩토리 함수 ──────────────────────────────────────────────────────────────

def _subset_values(ds: Dataset, attr: str) -> np.ndarray:
    """Return dataset metadata values, supporting Subset/ConcatDataset wrappers."""
    if isinstance(ds, MMECGWindowedH5Dataset):
        return getattr(ds, attr)
    if isinstance(ds, Subset):
        base = _subset_values(ds.dataset, attr)
        return base[np.asarray(ds.indices, dtype=np.int64)]
    if isinstance(ds, ConcatDataset):
        parts = [_subset_values(child, attr) for child in ds.datasets]
        return np.concatenate(parts, axis=0)
    raise TypeError(f"Unsupported dataset type for sampler metadata: {type(ds)!r}")


def _weights_from_values(values: np.ndarray) -> torch.Tensor:
    unique, counts = np.unique(values, return_counts=True)
    inv = 1.0 / counts.astype(np.float32)
    val2w = dict(zip(unique.tolist(), inv.tolist()))
    return torch.from_numpy(np.array([val2w[v] for v in values], dtype=np.float32))


def _make_sampler(
    ds: Dataset,
    balanced_sampling: bool,
    balance_by: str,
) -> WeightedRandomSampler | None:
    """Return a WeightedRandomSampler or None (sequential)."""
    if not balanced_sampling:
        return None
    attr = "_subj" if balance_by == "subject" else "_state"
    weights = _weights_from_values(_subset_values(ds, attr))
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def _split_calibration_indices(
    states: np.ndarray,
    calib_ratio: float,
    calib_val_ratio: float,
    seed: int,
    calib_n_train: int | None = None,
    calib_n_val: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val/test split over physistatus for calibration.

    By default, calib_ratio controls labeled target-subject windows added to
    training, and calib_val_ratio controls labeled target-subject windows used
    for early stopping/model selection.

    If calib_n_train and/or calib_n_val are provided, fixed segment counts take
    precedence over ratios. Counts are allocated approximately stratified by
    physistatus. The remaining windows are held out for testing.
    """
    rng = np.random.default_rng(seed)
    calib_idx: list[int] = []
    calib_val_idx: list[int] = []
    eval_idx: list[int] = []

    unique_states = sorted(np.unique(states).tolist())
    state_indices: dict[int, np.ndarray] = {}
    for st in unique_states:
        idx = np.flatnonzero(states == st)
        rng.shuffle(idx)
        state_indices[int(st)] = idx

    total_n = int(len(states))
    fixed_count_mode = calib_n_train is not None or calib_n_val is not None

    if fixed_count_mode:
        if calib_n_train is None:
            calib_n_train = int(round(total_n * calib_ratio))
        if calib_n_val is None:
            calib_n_val = int(round(total_n * calib_val_ratio))
        calib_n_train = int(calib_n_train)
        calib_n_val = int(calib_n_val)
        if calib_n_train <= 0:
            raise ValueError(f"calib_n_train must be > 0, got {calib_n_train}")
        if calib_n_val < 0:
            raise ValueError(f"calib_n_val must be >= 0, got {calib_n_val}")
        if calib_n_train + calib_n_val >= total_n:
            raise ValueError(
                "calib_n_train + calib_n_val must leave at least one eval segment, "
                f"got {calib_n_train}+{calib_n_val} for {total_n} segments"
            )

        # Global stratified sampling in two stages: train, then validation from
        # the remaining pool. This preserves approximate scene proportions and
        # works for folds with a single scene as well.
        all_idx = np.arange(total_n)
        train_pick: list[int] = []
        val_pick: list[int] = []
        remaining_by_state: dict[int, np.ndarray] = {}

        for st, idx in state_indices.items():
            n_st_train = int(np.floor(len(idx) / total_n * calib_n_train))
            train_pick.extend(idx[:n_st_train].tolist())
            remaining_by_state[st] = idx[n_st_train:]
        short = calib_n_train - len(train_pick)
        if short > 0:
            leftovers = np.concatenate(list(remaining_by_state.values()))
            rng.shuffle(leftovers)
            extra = leftovers[:short]
            train_pick.extend(extra.tolist())
            train_set = set(train_pick)
            for st, rem in remaining_by_state.items():
                remaining_by_state[st] = np.asarray(
                    [i for i in rem.tolist() if int(i) not in train_set],
                    dtype=np.int64,
                )

        remaining_total = total_n - len(train_pick)
        for st, rem in remaining_by_state.items():
            n_st_val = int(np.floor(len(rem) / max(remaining_total, 1) * calib_n_val))
            val_pick.extend(rem[:n_st_val].tolist())
            remaining_by_state[st] = rem[n_st_val:]
        short = calib_n_val - len(val_pick)
        if short > 0:
            leftovers = np.concatenate(list(remaining_by_state.values()))
            rng.shuffle(leftovers)
            val_pick.extend(leftovers[:short].tolist())

        used = set(train_pick) | set(val_pick)
        eval_pick = [int(i) for i in all_idx.tolist() if int(i) not in used]
        calib_idx.extend(train_pick)
        calib_val_idx.extend(val_pick)
        eval_idx.extend(eval_pick)
    else:
        if not 0.0 < calib_ratio < 1.0:
            raise ValueError(f"calib_ratio must be in (0, 1), got {calib_ratio}")
        if not 0.0 <= calib_val_ratio < 1.0:
            raise ValueError(f"calib_val_ratio must be in [0, 1), got {calib_val_ratio}")
        if calib_ratio + calib_val_ratio >= 1.0:
            raise ValueError(
                "calib_ratio + calib_val_ratio must be < 1.0, "
                f"got {calib_ratio + calib_val_ratio}"
            )
        for st in unique_states:
            idx = state_indices[int(st)]
            n_calib = int(round(len(idx) * calib_ratio))
            n_calib_val = int(round(len(idx) * calib_val_ratio))
            if calib_val_ratio > 0.0 and len(idx) >= 3:
                n_calib = max(n_calib, 1)
                n_calib_val = max(n_calib_val, 1)
                if n_calib + n_calib_val >= len(idx):
                    n_calib_val = max(1, len(idx) - n_calib - 1)
                if n_calib + n_calib_val >= len(idx):
                    n_calib = max(1, len(idx) - n_calib_val - 1)
            elif len(idx) > 1:
                n_calib = min(max(n_calib, 1), len(idx) - 1)
                n_calib_val = 0
            else:
                n_calib = 0
                n_calib_val = 0
            calib_idx.extend(idx[:n_calib].tolist())
            calib_val_idx.extend(idx[n_calib:n_calib + n_calib_val].tolist())
            eval_idx.extend(idx[n_calib + n_calib_val:].tolist())
    return (
        np.asarray(sorted(calib_idx), dtype=np.int64),
        np.asarray(sorted(calib_val_idx), dtype=np.int64),
        np.asarray(sorted(eval_idx), dtype=np.int64),
    )


def build_loso_loaders_h5(
    fold_idx: int,
    loso_dir: str | Path,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    balanced_sampling: bool = True,
    balance_by: str = "subject",
    narrow_bandpass: bool = False,
    topk_bins: int | None = None,
    target_norm: str = "minmax",
    topk_method: str = "energy",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    LOSO 프로토콜：fold_XX/{train,val,test}.h5 를 로드하여
    (train_loader, val_loader, test_loader) 를 반환.

    Parameters
    ----------
    fold_idx : int
        1-based fold 번호（1~11）. 03B_create_loso_splits.py 와 동일한 번호 체계.
    balanced_sampling : bool
        True → WeightedRandomSampler로 균형 학습.
    balance_by : str
        "subject" (기본) → 수험자 역빈도 가중치.
        "class"         → physistatus 역빈도 가중치.
    narrow_bandpass : bool
        True → RCG에 0.8-3.5 Hz 심박 협대역 필터 적용.
    """
    fold_dir = Path(loso_dir) / f"fold_{fold_idx:02d}"
    if not fold_dir.exists():
        raise FileNotFoundError(f"Fold directory not found: {fold_dir}")

    ds_kwargs = {
        "narrow_bandpass": narrow_bandpass,
        "topk_bins": topk_bins,
        "target_norm": target_norm,
        "topk_method": topk_method,
    }
    train_ds = MMECGWindowedH5Dataset(fold_dir / "train.h5", include_peak_indices=False, **ds_kwargs)
    val_ds   = MMECGWindowedH5Dataset(fold_dir / "val.h5",   include_peak_indices=False, **ds_kwargs)
    test_ds  = MMECGWindowedH5Dataset(fold_dir / "test.h5",  include_peak_indices=True,  **ds_kwargs)

    sampler = _make_sampler(train_ds, balanced_sampling, balance_by)
    if sampler is not None:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=_collate_with_peaks,
    )
    return train_loader, val_loader, test_loader


def build_loso_calibration_loaders_h5(
    fold_idx: int,
    loso_dir: str | Path,
    calib_ratio: float = 0.4,
    calib_val_ratio: float = 0.1,
    calib_n_train: int | None = None,
    calib_n_val: int | None = None,
    calib_seed: int = 42,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    balanced_sampling: bool = True,
    balance_by: str = "subject",
    narrow_bandpass: bool = False,
    topk_bins: int | None = None,
    target_norm: str = "minmax",
    topk_method: str = "energy",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """LOSO with supervised subject calibration.

    A stratified fraction of the held-out subject's test windows is added to
    training as labeled calibration data, another fraction is used as
    target-subject validation for early stopping, and the remaining windows are
    held out for testing. Report this protocol as "LOSO + subject calibration",
    not as strict LOSO.
    """
    fold_dir = Path(loso_dir) / f"fold_{fold_idx:02d}"
    if not fold_dir.exists():
        raise FileNotFoundError(f"Fold directory not found: {fold_dir}")

    ds_kwargs = {
        "narrow_bandpass": narrow_bandpass,
        "topk_bins": topk_bins,
        "target_norm": target_norm,
        "topk_method": topk_method,
    }
    train_base = MMECGWindowedH5Dataset(fold_dir / "train.h5", include_peak_indices=False, **ds_kwargs)
    test_plain = MMECGWindowedH5Dataset(fold_dir / "test.h5", include_peak_indices=False, **ds_kwargs)
    test_full = MMECGWindowedH5Dataset(fold_dir / "test.h5", include_peak_indices=True, **ds_kwargs)

    calib_idx, calib_val_idx, eval_idx = _split_calibration_indices(
        test_plain._state,
        calib_ratio=calib_ratio,
        calib_val_ratio=calib_val_ratio,
        calib_n_train=calib_n_train,
        calib_n_val=calib_n_val,
        seed=calib_seed + fold_idx,
    )
    if len(calib_idx) == 0 or len(calib_val_idx) == 0 or len(eval_idx) == 0:
        raise RuntimeError(
            f"Invalid calibration split for fold {fold_idx}: "
            f"calib_train={len(calib_idx)} calib_val={len(calib_val_idx)} "
            f"eval={len(eval_idx)}"
        )

    train_ds = ConcatDataset([train_base, Subset(test_plain, calib_idx.tolist())])
    val_ds = Subset(test_plain, calib_val_idx.tolist())
    test_ds = Subset(test_full, eval_idx.tolist())

    sampler = _make_sampler(train_ds, balanced_sampling, balance_by)
    if sampler is not None:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=_collate_with_peaks,
    )
    return train_loader, val_loader, test_loader


def build_samplewise_loaders_h5(
    sw_dir: str | Path,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    balanced_sampling: bool = True,
    balance_by: str = "subject",
    narrow_bandpass: bool = False,
    topk_bins: int | None = None,
    target_norm: str = "minmax",
    topk_method: str = "energy",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Samplewise 프로토콜：samplewise/{train,val,test}.h5 를 로드하여
    (train_loader, val_loader, test_loader) 를 반환.

    Parameters
    ----------
    balance_by : str
        "subject" (기본) → 수험자 역빈도 가중치.
        "class"         → physistatus 역빈도 가중치.
    narrow_bandpass : bool
        True → RCG에 0.8-3.5 Hz 심박 협대역 필터 적용.
    """
    sw_dir = Path(sw_dir)

    ds_kwargs = {
        "narrow_bandpass": narrow_bandpass,
        "topk_bins": topk_bins,
        "target_norm": target_norm,
        "topk_method": topk_method,
    }
    train_ds = MMECGWindowedH5Dataset(sw_dir / "train.h5", include_peak_indices=False, **ds_kwargs)
    val_ds   = MMECGWindowedH5Dataset(sw_dir / "val.h5",   include_peak_indices=False, **ds_kwargs)
    test_ds  = MMECGWindowedH5Dataset(sw_dir / "test.h5",  include_peak_indices=True,  **ds_kwargs)

    sampler = _make_sampler(train_ds, balanced_sampling, balance_by)
    if sampler is not None:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=_collate_with_peaks,
    )
    return train_loader, val_loader, test_loader
