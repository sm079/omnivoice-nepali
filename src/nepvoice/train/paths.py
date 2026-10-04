"""Where a training run's artefacts live.

::

    <run>/                         what a finished run keeps
      config.toml                  resolved configuration (data prep + training)
      adapter_config.json          LoRA adapter layout, shared by every checkpoint
      adapter_model.safetensors    the final adapter (``train.final_checkpoint``)
      checkpoint-<step>.safetensors  the other saved steps
      eval.json                    dev loss per checkpoint (when evaluated)
      scratch/                     while training; deleted by the package step
        manifests/  tokens/  data_config.json  exp/checkpoint-<step>/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ADAPTER = "adapter_model.safetensors"
ADAPTER_CONFIG = "adapter_config.json"


@dataclass(frozen=True)
class TrainPaths:
    root: Path

    @classmethod
    def of(cls, run: Path) -> "TrainPaths":
        return cls(Path(run))

    # ---- what a finished run keeps ----
    @property
    def config(self) -> Path:
        return self.root / "config.toml"

    @property
    def adapter(self) -> Path:
        return self.root / ADAPTER

    @property
    def adapter_config(self) -> Path:
        return self.root / ADAPTER_CONFIG

    def checkpoint_file(self, step: int) -> Path:
        return self.root / f"checkpoint-{step}.safetensors"

    @property
    def eval_file(self) -> Path:
        return self.root / "eval.json"

    @property
    def packaged(self) -> bool:
        """Finished: adapters written and scratch deleted (an interrupted package step is redone)."""
        return self.adapter.exists() and self.adapter_config.exists() and not self.scratch.exists()

    # ---- while training ----
    @property
    def scratch(self) -> Path:
        return self.root / "scratch"

    @property
    def manifests(self) -> Path:
        return self.scratch / "manifests"

    def manifest(self, split: str) -> Path:
        return self.manifests / f"{split}.jsonl"

    @property
    def tokens(self) -> Path:
        return self.scratch / "tokens"

    def token_list(self, split: str) -> Path:
        return self.tokens / split / "data.lst"

    @property
    def data_config(self) -> Path:
        return self.scratch / "data_config.json"

    @property
    def exp(self) -> Path:
        """OmniVoice output dir: ``checkpoint-<step>/`` folders (with optimizer state), logs, the run config."""
        return self.scratch / "exp"

    @property
    def eval_results(self) -> Path:
        return self.exp / "eval.json"

    def checkpoints(self) -> list[Path]:
        """``checkpoint-<step>`` folders, oldest first."""
        if not self.exp.exists():
            return []
        found = [p for p in self.exp.glob("checkpoint-*") if p.is_dir() and p.name.split("-")[-1].isdigit()]
        return sorted(found, key=lambda p: int(p.name.split("-")[-1]))
