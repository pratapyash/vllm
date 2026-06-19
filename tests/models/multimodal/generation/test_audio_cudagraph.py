# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder CUDA graph capture/replay for AUDIO encoders.

Audio counterpart of ``test_vit_cudagraph.py``: validates that a model whose
audio encoder implements ``SupportsEncoderCudaGraph`` boots, captures the
encoder budget graphs, and replays them to produce output (i.e. the
``cudagraph_mm_encoder`` path functions end-to-end). It checks that
capture/replay works, not output quality.
"""

from dataclasses import dataclass, field

import pytest

from vllm.platforms import current_platform


@dataclass
class AudioCudagraphTestConfig:
    model: str
    audio_prompt: str
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    max_tokens: int = 64
    max_num_seqs: int = 2
    vllm_runner_kwargs: dict = field(default_factory=dict)
    compilation_config_overrides: dict = field(default_factory=dict)
    marks: list = field(default_factory=list)
    skip: bool = False


def params_with_marks(
    configs: dict[str, AudioCudagraphTestConfig],
) -> list[pytest.param]:
    return [
        pytest.param(model_id, marks=cfg.marks) for model_id, cfg in configs.items()
    ]


def qwen_omni_audio_template(content: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|>"
        f"{content}<|im_end|>\n<|im_start|>assistant\n"
    )


MODEL_CONFIGS: dict[str, AudioCudagraphTestConfig] = {
    "qwen2_5_omni": AudioCudagraphTestConfig(
        model="Qwen/Qwen2.5-Omni-3B",
        audio_prompt=qwen_omni_audio_template("Transcribe the audio."),
        vllm_runner_kwargs={"trust_remote_code": True},
        marks=[pytest.mark.core_model],
    ),
}


def get_compilation_config(config: AudioCudagraphTestConfig):
    return {
        "cudagraph_mm_encoder": True,
        # Up to 2 audio clips per captured graph: co-scheduled clips exercise the
        # multi-item budget-packing path; otherwise each replays its own graph.
        "encoder_cudagraph_max_vision_items_per_batch": 2,
        **config.compilation_config_overrides,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _encoder_cudagraph_graph_hits(worker):
    """Run on each worker via collective_rpc: number of encoder items served by a
    captured CUDA graph (vs eager fallback). None if no manager was built."""
    mgr = getattr(worker.model_runner, "encoder_cudagraph_manager", None)
    return None if mgr is None else mgr.graph_hits


@pytest.mark.parametrize("model_id", params_with_marks(MODEL_CONFIGS))
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_audio_cudagraph(model_id, vllm_runner, audio_assets):
    config = MODEL_CONFIGS[model_id]

    if config.skip:
        pytest.skip(f"{model_id} is marked to be skipped.")

    # One prompt per audio asset (different durations) so replays exercise the
    # variable-length cu_seqlens; with max_vision_items_per_batch=2, co-scheduled
    # clips also exercise the multi-item budget-packing path.
    prompts = [config.audio_prompt for _ in audio_assets]
    audios = [[asset.audio_and_sample_rate] for asset in audio_assets]

    with vllm_runner(
        config.model,
        dtype=config.dtype,
        max_model_len=config.max_model_len,
        max_num_seqs=config.max_num_seqs,
        limit_mm_per_prompt={"audio": 1},
        compilation_config=get_compilation_config(config),
        **config.vllm_runner_kwargs,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(
            prompts, config.max_tokens, audios=audios
        )

        # Basic validation that we got a response for each audio.
        assert len(outputs) == len(audio_assets)
        for output_ids, output_text in outputs:
            assert len(output_ids) > 0
            assert len(output_text) > 0
            assert isinstance(output_text, str)

        # The encoder cudagraph path must have actually replayed a captured graph --
        # otherwise this test would still pass if every replay silently fell back to
        # eager (the manager built but never hit), defeating the point of the test.
        hits = vllm_model.collective_rpc(_encoder_cudagraph_graph_hits)
        assert hits and hits[0] is not None and hits[0] > 0, (
            f"audio encoder cudagraph never replayed (graph_hits={hits}); the "
            "cudagraph_mm_encoder path fell back to eager"
        )
