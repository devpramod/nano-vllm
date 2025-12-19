import logging
import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.platforms import get_platform
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model

logger = logging.getLogger(__name__)


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        # Get platform for device-agnostic operations
        self.platform = get_platform(config.device_type)
        self.device = self.platform.device_name
        # HPU doesn't benefit from pinned memory (unified memory architecture)
        self.pin_memory = self.device == "cuda"

        # Initialize distributed with platform-specific backend (nccl/hccl)
        dist.init_process_group(
            self.platform.get_distributed_backend(),
            "tcp://localhost:2333",
            world_size=self.world_size,
            rank=rank,
        )
        self.platform.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device(self.device)
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.bucketing = None  # Initialized later for HPU

        # Initial warmup for memory measurement (before HPU graph wrapping)
        self.warmup_model()
        self.allocate_kv_cache()

        # Platform-specific inference optimization
        if not self.enforce_eager:
            # HPU: Initialize bucketing, wrap model, then warmup all bucket shapes
            if self.device == "hpu":
                self._init_hpu_bucketing()
                self.model = self.platform.wrap_model_for_inference(self.model)
                self.warmup_buckets()
            # CUDA: Wrap (no-op) then capture explicit CUDA graphs per batch size
            elif self.platform.supports_cuda_graphs:
                self.model = self.platform.wrap_model_for_inference(self.model)
                self.capture_cudagraph()

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        # Only delete graphs if they were captured (CUDA only, not HPU)
        if not self.enforce_eager and self.platform.supports_cuda_graphs:
            del self.graphs, self.graph_pool
        self.platform.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        self.platform.empty_cache()
        self.platform.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        self.platform.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = self.platform.get_memory_info()
        used = total - free
        peak = self.platform.get_peak_memory()
        current = self.platform.get_current_memory()
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0, (
            f"Not enough memory for KV cache. Free: {free}, Peak: {peak}, "
            f"Block size: {block_bytes} bytes. Try reducing max_model_len or batch size."
        )
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def _init_hpu_bucketing(self):
        """Initialize HPU bucketing manager after KV cache allocation."""
        from nanovllm.platforms.hpu.bucketing import LinearBucketing

        self.bucketing = LinearBucketing(self.config)
        self.bucketing.initialize(self.config.num_kvcache_blocks)
        logger.info(f"HPU bucketing initialized with {self.bucketing.total_buckets} total buckets")

    @torch.inference_mode()
    def warmup_buckets(self):
        """
        Warmup all bucket shapes to pre-compile HPU graphs.

        Each forward pass with a unique shape compiles a new HPU graph.
        By warming up all bucket shapes, we avoid runtime compilation.
        """
        if self.bucketing is None:
            logger.warning("Bucketing not initialized, skipping bucket warmup")
            return

        logger.info(f"Starting HPU bucket warmup: {self.bucketing.total_buckets} buckets")
        self.platform.empty_cache()

        # Warmup prefill buckets
        prefill_buckets = self.bucketing.get_prefill_buckets()
        for i, (bs, seq_len) in enumerate(prefill_buckets):
            logger.debug(f"Warmup prefill bucket {i+1}/{len(prefill_buckets)}: bs={bs}, seq_len={seq_len}")

            # Create dummy inputs matching bucket shape
            input_ids = torch.zeros(bs * seq_len, dtype=torch.int64, device=self.device)
            positions = torch.zeros(bs * seq_len, dtype=torch.int64, device=self.device)

            # Run forward pass to compile HPU graph for this shape
            self.model(input_ids, positions)

            # Critical: mark_step() flushes the compiled graph
            if hasattr(self.platform, "mark_step"):
                self.platform.mark_step()
            self.platform.synchronize()

        # Warmup decode buckets
        decode_buckets = self.bucketing.get_decode_buckets()
        for i, (bs, num_blocks) in enumerate(decode_buckets):
            logger.debug(f"Warmup decode bucket {i+1}/{len(decode_buckets)}: bs={bs}, num_blocks={num_blocks}")

            # Decode: input is always 1 token per sequence
            input_ids = torch.zeros(bs, dtype=torch.int64, device=self.device)
            positions = torch.zeros(bs, dtype=torch.int64, device=self.device)

            # Run forward pass
            self.model(input_ids, positions)

            # Flush compiled graph
            if hasattr(self.platform, "mark_step"):
                self.platform.mark_step()
            self.platform.synchronize()

        self.platform.empty_cache()
        logger.info("HPU bucket warmup complete")

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(
            input_ids, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        positions = torch.tensor(
            positions, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(
            input_ids, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        positions = torch.tensor(
            positions, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        # Use eager execution for: prefill, enforce_eager, large batch, or HPU (uses lazy mode)
        use_eager = (
            is_prefill
            or self.enforce_eager
            or input_ids.size(0) > 512
            or not self.platform.supports_cuda_graphs
        )
        if use_eager:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

        # HPU lazy mode: mark_step() before model execution to flush tensor transfers
        if hasattr(self.platform, "mark_step"):
            self.platform.mark_step()

        logits = self.run_model(input_ids, positions, is_prefill)

        # HPU lazy mode: mark_step() before sampling to ensure logits are computed
        if hasattr(self.platform, "mark_step"):
            self.platform.mark_step()

        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
