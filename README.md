# jero-arch-gemma

<p align="center">
  <a href="https://hook.lma1980.duckdns.org/jobs">
    <img src="https://hook.lma1980.duckdns.org/cards/LMA1980/jero-arch-gemma/card.svg" alt="jero-arch-gemma Factory Scorecard" width="520"/>
  </a>
</p>

<p align="center">
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-arch-gemma/status.svg" alt="CI Status"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-arch-gemma/score.svg" alt="Quality Score"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-arch-gemma/coverage.svg" alt="Coverage"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-arch-gemma/benchmark.svg" alt="Benchmark"/></a>
</p>

______________________________________________________________________

Google Gemma 2, Gemma 3, and Gemma 4 decoder and VLM architecture recipes for the SpacemiT K3 platform.

## Capabilities & Architecture Contract

- **Recipes & Sparse MoE / PLE Helpers**: Exports `GemmaRecipe` (Gemma 2 / Gemma 3 / FunctionGemma), `Gemma4Recipe` (Edge E2B/E4B Per-Layer Embeddings and 26B-A4B Sparse MoE), `ExpertWeights`, `apply_gemma4_ple_residual`, `dispatch_sparse_moe_layer`, `select_top_k_experts`, `write_and_slice_sliding_kv`, and `pub fn matvec_any` (parallelized via `rayon`).
- **ADR-045 Fail-Closed Execution**: `prefill_step`, `decode_step`, and `validate_model` require real resident weights via `ModelWeights::tensor(...)`, returning `K3Error::TensorBoundsError` or `K3Error::CapabilityMismatch` on missing, dimension-mismatched, or unsupported tensors without synthetic fallbacks.
- **Dual Format Support (`GGUF` & `SafeTensors`)**: Supports `GgufFile` (`blk.{l}.*`) and HuggingFace `SafeTensorsFile` (`language_model.model.layers.{l}.*` and `model.layers.{l}.*`) with bit-identical forward-pass math across both weight layouts.
- **VLM Layer 0 Soft-Prompt Conditioning**: Injects `ExecutionContext::soft_prompt_embeds()` / `soft_prompt_dims()` at Layer 0 ahead of discrete prompt tokens so multimodal embeddings from `jero-arch-vision` populate the KV cache seamlessly.

## Ecosystem & Documentation

- **Public Documentation Portal**: [`lma1980.github.io/model-runner`](https://github.com/LMA1980/lma1980.github.io/tree/main/model-runner) ([Reference Manual](https://github.com/LMA1980/lma1980.github.io/blob/main/model-runner/manual.md), [Crate Docs](https://github.com/LMA1980/lma1980.github.io/tree/main/model-runner/docs/crates))
- **Upstream Dependencies**: [`jero-core`](https://github.com/LMA1980/jero-core), [`jero-kernels-rvv`](https://github.com/LMA1980/jero-kernels-rvv), and [`rayon`](https://crates.io/crates/rayon) (flat peer crate with no dependency on `jero-engine`)
- **Downstream & Companion Crates**: [`jero-engine`](https://github.com/LMA1980/jero-engine), [`jero-arch-vision`](https://github.com/LMA1980/jero-arch-vision), and [`jero-kv-cache`](https://github.com/LMA1980/jero-kv-cache)

## License

Licensed under the GNU Lesser General Public License v3.0 with the LGPL-3.0 Linking Exception (`LGPL-3.0-only WITH LGPL-3.0-linking-exception`). See [LICENSE](LICENSE) for details.
