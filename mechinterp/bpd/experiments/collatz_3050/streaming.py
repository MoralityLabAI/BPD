"""Bounded-memory execution of the pinned native Nanbeige recurrence.

Only one physical decoder layer is resident at a time. The original model
forward controls recurrence, normalization and attention. Embeddings are read
by row, and the output head is read in bounded row chunks. The production
constructor accepts a verified local snapshot; it never downloads weights.

The memory limits are runtime admission checks. A CUDA allocator cap and
sampled process RSS do not establish a measured RTX 3050 result by themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Sequence


MODEL_ID = "Nanbeige/Nanbeige4.2-3B"
MODEL_REVISION = "451ed48c3273ecef7ea8faaa43c31ce529763bb1"
SOURCE_SHA256 = {
    "config.json": "f6cb15b22847664f3a6049dc4b58fdd10f1650d112ac99a1da3d051f17c2ca19",
    "configuration_nanbeige.py": "98abb9faa303717b234b3d381a9650e8cb656ccf287a12f6ad6215a3bbf8be10",
    "modeling_nanbeige.py": "547737f989f5cb741c1a568acaf85f83521fa67a8f7268e19f9a37e60127c0d5",
    "model.safetensors.index.json": "30d8da0fa8b97abc6d9eddfd017a0cb7a649bcbd58caa57804c56c767db5c0f1",
    "tokenizer_config.json": "82316b70213c65e5f820c03d5a11561ccda4055bda984cb770ed0d51fe497c56",
    "tokenizer.json": "1d858a0fc007f22af6ae18bfa1ae52d30e398aa9cd1ea06e7777176869346a3f",
    "tokenizer.model": "fb41d04798b714520a9b075727b0226538b7330254299062742c50ec8374bc36",
    "added_tokens.json": "9e3b127a27647df2c353cc1e5500826f7cdbe8bd15a458e368bba8422e9719cf",
    "special_tokens_map.json": "b718fce2b7a8940ffeddc1e67f3b092cc0d13ac885c63a021528786f8c4cf6c0",
}
WEIGHT_FILES = {
    "model-00001-of-00002.safetensors": {
        "bytes": 4973547960,
        "sha256": "09d265d5ec837bc64462796b7f8c110be9a135a55ed7a6eb5d07e0e90c976a94",
    },
    "model-00002-of-00002.safetensors": {
        "bytes": 3366076760,
        "sha256": "31019e7870a044f44bc3f7e981f8c5ecd42d341e5ca6cfdbfd07fb95d95be389",
    },
}
GIB = 1024 ** 3


class MemoryBudgetExceeded(RuntimeError):
    """The run has crossed a declared memory admission limit."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_snapshot(snapshot: str | Path, *, hash_weights: bool = False) -> dict:
    """Check the official source pin and local shard inventory without Torch.

    With hash_weights=True, every weight byte is checked against the official
    LFS SHA-256 using a bounded one-MiB buffer. Production execution requests
    that check once when constructing its backend.
    """
    root = Path(snapshot).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Local Nanbeige snapshot directory is missing: {root}")
    hashes = {}
    for name, expected in SOURCE_SHA256.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Pinned snapshot file is missing: {name}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Pinned source hash mismatch for {name}")
        hashes[name] = actual
    shard_receipts = {}
    for name, expected in WEIGHT_FILES.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Weight shard is missing: {name}")
        size = path.stat().st_size
        if size != expected["bytes"]:
            raise ValueError(f"Weight shard size mismatch for {name}")
        actual = sha256_file(path) if hash_weights else None
        if hash_weights and actual != expected["sha256"]:
            raise ValueError(f"Official LFS weight hash mismatch for {name}")
        shard_receipts[name] = {"bytes": size, "sha256": actual,
                                "expected_sha256": expected["sha256"]}
    config = json.loads((root / "config.json").read_text())
    return {
        "model_id": MODEL_ID, "revision": MODEL_REVISION,
        "source_sha256": hashes, "weight_files": shard_receipts,
        "weights_sha256_verified": hash_weights,
        "config": {key: config.get(key) for key in (
            "hidden_size", "intermediate_size", "num_hidden_layers", "num_loops",
            "vocab_size", "num_attention_heads", "num_key_value_heads", "head_dim")},
    }


class TensorStore:
    """Short-lived safetensors mappings with copied, bounded return values."""

    def __init__(self, snapshot: str | Path):
        self.root = Path(snapshot).resolve()
        index = self.root / "model.safetensors.index.json"
        if index.exists():
            self.weight_map = json.loads(index.read_text())["weight_map"]
        else:
            # Tiny native-model fixtures save a single standard safetensors file.
            from safetensors import safe_open
            path = self.root / "model.safetensors"
            with safe_open(path, framework="pt", device="cpu") as handle:
                self.weight_map = {name: path.name for name in handle.keys()}
        if not isinstance(self.weight_map, dict) or not self.weight_map:
            raise ValueError("Empty or invalid checkpoint weight map")
        for filename in set(self.weight_map.values()):
            if Path(filename).name != filename or not (self.root / filename).is_file():
                raise ValueError("Checkpoint shards must be existing files inside the snapshot")
        self.read_bytes = 0

    def tensor(self, name: str, *, device: Any, dtype: Any) -> Any:
        from safetensors import safe_open
        with safe_open(self.root / self.weight_map[name], framework="pt", device="cpu") as handle:
            source = handle.get_tensor(name)
            # copy=True also prevents a CPU fixture from retaining mmap storage.
            result = source.to(device=device, dtype=dtype, copy=True)
            self.read_bytes += source.numel() * source.element_size()
            del source
        return result

    def rows(self, name: str, indices: Sequence[int], *, device: Any, dtype: Any) -> Any:
        import torch
        from safetensors import safe_open
        if not indices:
            raise ValueError("At least one tensor row is required")
        with safe_open(self.root / self.weight_map[name], framework="pt", device="cpu") as handle:
            view = handle.get_slice(name)
            shape = view.get_shape()
            if len(shape) != 2 or any(type(i) is not int or i < 0 or i >= shape[0] for i in indices):
                raise ValueError("Invalid matrix row selection")
            copied = [view[i:i + 1].clone() for i in indices]
            rows = torch.cat(copied, dim=0)
            self.read_bytes += rows.numel() * rows.element_size()
            del copied, view
        return rows.to(device=device, dtype=dtype, copy=True)

    def row_range(self, name: str, start: int, stop: int, *, device: Any, dtype: Any) -> Any:
        from safetensors import safe_open
        with safe_open(self.root / self.weight_map[name], framework="pt", device="cpu") as handle:
            view = handle.get_slice(name)
            shape = view.get_shape()
            if len(shape) != 2 or not 0 <= start < stop <= shape[0]:
                raise ValueError("Invalid matrix row range")
            source = view[start:stop]
            result = source.to(device=device, dtype=dtype, copy=True)
            self.read_bytes += source.numel() * source.element_size()
            del source, view
        return result


class MemoryGuard:
    """Sample process RSS and enforce the CUDA allocator budget."""

    def __init__(self, device: Any, cuda_budget_gib: float, host_budget_gib: float,
                 baseline_rss: int):
        import psutil
        import torch
        if not 0 < cuda_budget_gib < 2 or not 0 < host_budget_gib < 3:
            raise ValueError("This profile requires CUDA budget <2 GiB and added RSS budget <3 GiB")
        self.device = device
        self.cuda_limit = int(cuda_budget_gib * GIB)
        self.host_limit = int(host_budget_gib * GIB)
        self.baseline_rss = baseline_rss
        self.process = psutil.Process()
        self.peak_rss = baseline_rss
        self.peak_cuda_allocated = 0
        self.peak_cuda_reserved = 0
        self.samples = 0
        if device.type == "cuda":
            total = torch.cuda.get_device_properties(device).total_memory
            torch.cuda.set_per_process_memory_fraction(min(self.cuda_limit / total, 1), device)
            torch.cuda.reset_peak_memory_stats(device)
        self.check("initialization")

    def check(self, stage: str) -> None:
        import torch
        rss = self.process.memory_info().rss
        self.peak_rss = max(self.peak_rss, rss)
        self.samples += 1
        if rss - self.baseline_rss > self.host_limit:
            raise MemoryBudgetExceeded(f"Added process RSS exceeded its cap during {stage}")
        if self.device.type == "cuda":
            allocated = torch.cuda.max_memory_allocated(self.device)
            reserved = torch.cuda.max_memory_reserved(self.device)
            self.peak_cuda_allocated = max(self.peak_cuda_allocated, allocated)
            self.peak_cuda_reserved = max(self.peak_cuda_reserved, reserved)
            if max(allocated, reserved) > self.cuda_limit:
                raise MemoryBudgetExceeded(f"CUDA allocator exceeded its cap during {stage}")

    def receipt(self) -> dict:
        return {
            "baseline_rss_bytes": self.baseline_rss,
            "peak_rss_bytes": self.peak_rss,
            "peak_added_rss_bytes": max(0, self.peak_rss - self.baseline_rss),
            "peak_cuda_allocated_bytes": self.peak_cuda_allocated,
            "peak_cuda_reserved_bytes": self.peak_cuda_reserved,
            "cuda_budget_bytes": self.cuda_limit, "added_rss_budget_bytes": self.host_limit,
            "memory_samples": self.samples,
            "memory_scope": "sampled process RSS and Torch CUDA allocator; driver/display memory excluded",
        }


@dataclass
class RunResult:
    boundary_states_cpu: tuple[Any, ...]
    boundary_last_cpu: tuple[Any, ...]
    final_hidden_cpu: Any
    receipt: dict


class Backend:
    """Run the pinned native model with streamed weights and native norm hooks."""

    def __init__(self, snapshot: str | Path, *, device: str = "cuda:0",
                 dtype: str = "bfloat16", loops: int = 2, max_tokens: int = 128,
                 cuda_budget_gib: float = 1.75, host_budget_gib: float = 2.75,
                 head_chunk_rows: int = 256, baseline_rss_bytes: int | None = None):
        import psutil
        if type(loops) is not int or loops != 2:
            raise ValueError("Production COL-02 uses the checkpoint's trained two-loop schedule")
        baseline = psutil.Process().memory_info().rss if baseline_rss_bytes is None else baseline_rss_bytes
        self.provenance = inspect_snapshot(snapshot, hash_weights=True)
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
        execution_device = torch.device(device)
        if execution_device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Production streaming requires CUDA; run the CPU fixture tests for source validation")
        if dtype != "bfloat16" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("The registered profile requires native CUDA BF16 support")
        config = AutoConfig.from_pretrained(str(snapshot), trust_remote_code=True, local_files_only=True)
        self._validate_config(config)
        config.num_loops = loops
        config.use_cache = False
        config._attn_implementation = "sdpa"
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True,
                                                   torch_dtype=torch.bfloat16,
                                                   attn_implementation="sdpa")
        self._initialize(model, TensorStore(snapshot), execution_device, torch.bfloat16,
                         loops, max_tokens, cuda_budget_gib, host_budget_gib,
                         head_chunk_rows, baseline)

    @staticmethod
    def _validate_config(config: Any) -> None:
        expected = {"hidden_size": 3072, "num_hidden_layers": 22, "vocab_size": 166144,
                    "num_attention_heads": 48, "num_key_value_heads": 8, "head_dim": 128,
                    "num_loops": 2, "intermediate_size": 10752}
        if any(getattr(config, key, None) != value for key, value in expected.items()):
            raise ValueError("The local configuration differs from the pinned Nanbeige architecture")
        flags = ("skip_loop_final_norm", "enable_double_loop_split", "loop_share_kv",
                 "enable_hyper_connection", "enable_mhc", "enable_depth_attention")
        if any(getattr(config, flag, False) for flag in flags) or getattr(config, "emb_neighbor_num", None):
            raise ValueError("Unsupported alternate recurrence or embedding path")
        if getattr(config, "loop_loss_weights", None):
            raise ValueError("Loop-loss weights would override the registered loop count")

    @classmethod
    def _from_fixture(cls, model: Any, snapshot: str | Path, *, loops: int = 2,
                      max_tokens: int = 128, dtype: str = "float32") -> "Backend":
        """CPU-only test factory for an official-code tiny model saved locally."""
        import psutil
        import torch
        if dtype not in ("float32", "bfloat16"):
            raise ValueError("CPU native fixtures support float32 and bfloat16")
        fixture_dtype = getattr(torch, dtype)
        instance = cls.__new__(cls)
        instance.provenance = {"evidence": "tiny native architecture CPU fixture; not pretrained weights"}
        model.config.num_loops = loops
        model.config.use_cache = False
        model.eval().to("meta")
        instance._initialize(model, TensorStore(snapshot), torch.device("cpu"), fixture_dtype,
                             loops, max_tokens, 1.75, 2.75, 7,
                             psutil.Process().memory_info().rss)
        return instance

    def _initialize(self, model: Any, store: TensorStore, device: Any, dtype: Any,
                    loops: int, max_tokens: int, cuda_budget_gib: float,
                    host_budget_gib: float, head_chunk_rows: int, baseline_rss: int) -> None:
        import torch
        if type(loops) is not int or not 1 <= loops <= 4:
            raise ValueError("Loop count must be an integer from one through four")
        if type(max_tokens) is not int or not 2 <= max_tokens <= 128:
            raise ValueError("The 3050 profile accepts at most 128 tokens")
        if type(head_chunk_rows) is not int or not 1 <= head_chunk_rows <= 1024:
            raise ValueError("Head chunk size must be from one through 1024 rows")
        self.model, self.store, self.device, self.dtype = model.eval(), store, device, dtype
        self.loops, self.max_tokens, self.head_chunk_rows = loops, max_tokens, head_chunk_rows
        self.guard = MemoryGuard(device, cuda_budget_gib, host_budget_gib, baseline_rss)
        self._hooks = []
        self._running = False
        self._closed = False
        self._active_layer = None
        self._layer_visits = []
        self._boundaries = []
        self._last_vectors = []
        self._intervention = None
        self._capture_prefix = True
        self._protected_checks = []
        self._specs = []
        expected_keys = set(dict(model.named_parameters()))
        if expected_keys != set(store.weight_map):
            raise ValueError("Checkpoint parameter inventory differs from the native model")
        if not all(p.device.type == "meta" for p in model.parameters()):
            raise ValueError("The streaming model must begin with all parameters on meta")
        self.largest_layer_parameter_bytes = 0
        for index, layer in enumerate(model.model.layers):
            specs = [(name, tuple(parameter.shape)) for name, parameter in layer.named_parameters()]
            self._specs.append(specs)
            count = sum(parameter.numel() for parameter in layer.parameters())
            self.largest_layer_parameter_bytes = max(self.largest_layer_parameter_bytes,
                                                     count * torch.empty((), dtype=dtype).element_size())
            # The pinned RoPE buffer is nonpersistent and absent from the checkpoint.
            rotary = layer.self_attn.rotary_emb
            rotary.inv_freq = 1.0 / (rotary.base ** (
                torch.arange(0, rotary.dim, 2, dtype=torch.int64, device=device).float() / rotary.dim))
            self._hooks.append(layer.register_forward_pre_hook(self._pre_hook(index), with_kwargs=True))
            self._hooks.append(layer.register_forward_hook(self._post_hook(index), always_call=True))
        self._set_parameter(model.model.norm, "weight", store.tensor("model.norm.weight", device=device, dtype=dtype))
        self._hooks.append(model.model.norm.register_forward_hook(self._norm_hook))
        self.guard.check("model setup")

    @staticmethod
    def _set_parameter(module: Any, name: str, value: Any) -> None:
        import torch
        parent_name, _, leaf = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        parent._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)

    def _pre_hook(self, index: int) -> Callable:
        def load(layer: Any, args: tuple, kwargs: dict) -> None:
            if self._active_layer is not None:
                raise RuntimeError("More than one physical decoder layer became resident")
            self._active_layer = index
            expected_loop = len(self._layer_visits) // len(self._specs)
            expected_layer = len(self._layer_visits) % len(self._specs)
            if expected_layer != index or kwargs.get("loop_idx", 0) != expected_loop:
                raise RuntimeError("Native recurrent layer order changed")
            for name, shape in self._specs[index]:
                value = self.store.tensor(f"model.layers.{index}.{name}", device=self.device, dtype=self.dtype)
                if tuple(value.shape) != shape:
                    raise ValueError("Checkpoint tensor shape disagrees with native parameter")
                self._set_parameter(layer, name, value)
                del value
                self.guard.check(f"layer {index} load")
            self._layer_visits.append([expected_loop, index])
        return load

    def _release_layer(self, index: int) -> None:
        import torch
        layer = self.model.model.layers[index]
        for name, shape in self._specs[index]:
            self._set_parameter(layer, name, torch.empty(shape, dtype=self.dtype, device="meta"))
        self._active_layer = None

    def _post_hook(self, index: int) -> Callable:
        def release(layer: Any, args: tuple, output: Any) -> None:
            try:
                self.guard.check(f"layer {index} forward")
            finally:
                self._release_layer(index)
        return release

    def _norm_hook(self, module: Any, args: tuple, output: Any) -> Any:
        import torch
        visit = len(self._last_vectors)
        if visit >= self.loops:
            raise RuntimeError("Unexpected extra native loop normalization")
        before = output[:, -1, :].clone()
        changed = self._intervention(output, visit) if self._intervention is not None else output
        if not isinstance(changed, torch.Tensor) or changed.shape != output.shape:
            raise ValueError("Intervention changed the native boundary tensor shape")
        if changed.device != output.device or changed.dtype != output.dtype:
            raise ValueError("Intervention changed native device or dtype")
        if not torch.isfinite(changed).all().item():
            raise ValueError("Intervention produced a nonfinite boundary state")
        protected = torch.equal(before, changed[:, -1, :])
        if not protected:
            raise ValueError("Intervention changed the protected final-token native head input")
        self._protected_checks.append(protected)
        self._last_vectors.append(changed[0, -1, :].detach().to("cpu", copy=True))
        if self._capture_prefix:
            self._boundaries.append(changed.detach().to("cpu", copy=True))
        self.guard.check(f"native loop boundary {visit}")
        return changed

    def run(self, input_ids: Any, intervention: Callable | None = None,
            capture_prefix: bool = True) -> RunResult:
        import torch
        if self._closed or self._running:
            raise RuntimeError("Backend is closed or already running")
        ids = torch.as_tensor(input_ids, device="cpu")
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Input token IDs must be integers")
        ids = ids.to(dtype=torch.long)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.ndim != 2 or ids.shape[0] != 1 or not 2 <= ids.shape[1] <= self.max_tokens:
            raise ValueError("Use one unpadded sequence with length 2..max_tokens")
        if ids.min().item() < 0 or ids.max().item() >= self.model.config.vocab_size:
            raise ValueError("Input token outside the model vocabulary")
        self._running = True
        self._layer_visits, self._boundaries, self._last_vectors = [], [], []
        self._protected_checks = []
        self._intervention, self._capture_prefix = intervention, bool(capture_prefix)
        bytes_before = self.store.read_bytes
        started = time.monotonic()
        try:
            with torch.inference_mode():
                embeddings = self.store.rows("model.embed_tokens.weight", ids[0].tolist(),
                                             device=self.device, dtype=self.dtype).unsqueeze(0)
                self.guard.check("embedding rows")
                output = self.model.model(inputs_embeds=embeddings, use_cache=False,
                                          output_attentions=False, output_hidden_states=False,
                                          return_dict=True)
                final = output.last_hidden_state[0, -1, :].detach().to("cpu", copy=True)
                del output, embeddings
            expected = [[loop, index] for loop in range(self.loops) for index in range(len(self._specs))]
            if self._layer_visits != expected or len(self._last_vectors) != self.loops:
                raise RuntimeError("Native recurrence did not execute the declared complete schedule")
            if not torch.equal(final, self._last_vectors[-1]):
                raise RuntimeError("Native final state disagrees with its last loop boundary")
            if not self.layers_released():
                raise RuntimeError("A decoder parameter remained resident after its visit")
            self.guard.check("completed native continuation")
            receipt = self.memory_receipt()
            receipt.update({
                "loop_count": self.loops, "physical_layers": len(self._specs),
                "native_layer_visits": self._layer_visits,
                "native_layer_visit_columns": ["zero_based_loop", "physical_layer"],
                "input_tokens": ids.shape[1], "use_cache": False,
                "streamed_checkpoint_bytes": self.store.read_bytes - bytes_before,
                "elapsed_seconds": time.monotonic() - started,
                "protected_last_token_equal_each_boundary": self._protected_checks,
                "decoder_parameters_released_to_meta": True,
            })
            return RunResult(tuple(self._boundaries), tuple(self._last_vectors), final, receipt)
        finally:
            if self._active_layer is not None:
                self._release_layer(self._active_layer)
            self._running = False
            self._intervention = None

    def layers_released(self) -> bool:
        return all(parameter.device.type == "meta" for layer in self.model.model.layers
                   for parameter in layer.parameters())

    def score_hidden(self, hidden_cpu: Any, token_ids: Sequence[int] | None = None) -> Any:
        """Return CPU float32 logits, including every vocabulary row if requested."""
        import torch
        import torch.nn.functional as functional
        if self._closed or self._running:
            raise RuntimeError("Backend is closed or already running")
        hidden = torch.as_tensor(hidden_cpu)
        if hidden.ndim == 1:
            hidden = hidden.unsqueeze(0)
        if hidden.ndim != 2 or hidden.shape[1] != self.model.config.hidden_size or hidden.shape[0] > 32:
            raise ValueError("Head scoring expects at most 32 vectors with native hidden width")
        if not torch.isfinite(hidden).all().item():
            raise ValueError("Nonfinite head input")
        selected = None if token_ids is None else list(token_ids)
        if selected is not None and (not selected or any(type(i) is not int or i < 0 or
                                                       i >= self.model.config.vocab_size for i in selected)):
            raise ValueError("Invalid answer token IDs")
        size = self.model.config.vocab_size if selected is None else len(selected)
        scores = torch.empty((hidden.shape[0], size), dtype=torch.float32, device="cpu")
        with torch.inference_mode():
            vector = hidden.to(device=self.device, dtype=self.dtype)
            for start in range(0, size, self.head_chunk_rows):
                stop = min(size, start + self.head_chunk_rows)
                if selected is None:
                    weight = self.store.row_range("lm_head.weight", start, stop,
                                                  device=self.device, dtype=self.dtype)
                else:
                    weight = self.store.rows("lm_head.weight", selected[start:stop],
                                             device=self.device, dtype=self.dtype)
                logits = functional.linear(vector, weight)
                scores[:, start:stop] = logits.float().to("cpu")
                self.guard.check("streamed output-head chunk")
                del logits, weight
            del vector
        self.guard.check("output-head completion")
        return scores

    def memory_receipt(self) -> dict:
        return {
            **self.guard.receipt(), "device": str(self.device), "dtype": str(self.dtype),
            "largest_physical_layer_parameter_bytes": self.largest_layer_parameter_bytes,
            "weight_residency": "one physical decoder layer; embeddings/head read by row",
            "provenance": self.provenance,
        }

    def close(self) -> None:
        import torch
        if self._running:
            raise RuntimeError("Cannot close a running backend")
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        self.model.to("meta")
        self._closed = True
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def __enter__(self) -> "Backend":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
