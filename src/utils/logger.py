"""logger.py — 实验日志封装（Python logging + TensorBoard）"""

import logging
import sys
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    """
    统一的实验日志接口。

    同时写入：
      - 控制台 (stdout)
      - 文件  <log_dir>/train.log
      - TensorBoard <log_dir>/

    Parameters
    ----------
    log_dir : Path
        实验日志目录，如 experiments/ExpA/fold_0/logs/
    name : str
        logger 名称
    """

    def __init__(self, log_dir: Path, name: str = "train") -> None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # ── Python logging ────────────────────────────────────────────────
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.INFO)
        self._logger.handlers.clear()

        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")

        # 控制台
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        self._logger.addHandler(ch)

        # 文件
        fh = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
        fh.setFormatter(fmt)
        self._logger.addHandler(fh)

        # ── TensorBoard ───────────────────────────────────────────────────
        self.writer = SummaryWriter(log_dir=str(log_dir))

    # ── logging 代理 ──────────────────────────────────────────────────────

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        self._logger.warning(msg)

    def error(self, msg: str) -> None:
        self._logger.error(msg)

    # ── TensorBoard 写入 ──────────────────────────────────────────────────

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)

    def log_dict(self, d: dict, step: int, prefix: str = "") -> None:
        """批量写入标量字典，key 自动加 prefix/。"""
        for k, v in d.items():
            tag = f"{prefix}/{k}" if prefix else k
            self.writer.add_scalar(tag, float(v), step)

    def close(self) -> None:
        self.writer.close()
