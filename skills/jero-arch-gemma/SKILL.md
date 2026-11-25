---
name: jero-arch-gemma
description: Google Gemma 2, Gemma 3, and Gemma 4 decoder/VLM architecture recipes (`GemmaRecipe`, `Gemma4Recipe`) with `SafeTensors` + `GGUF` hydration, ADR-045 fail-closed execution, and Layer 0 `soft_prompt_embeds` support.
---

# `jero-arch-gemma` Architecture Crate

Provides standalone Gemma 2, Gemma 3, FunctionGemma, and Gemma 4 decoder/VLM recipes extracted into a flat peer crate over `jero-core` and `jero-kernels-rvv`.

## Exported Surface

- **Recipes**: `GemmaRecipe` (`new_gemma2`, `new_gemma3`, `new_functiongemma`, `from_metadata`, `from_gguf`, `from_safetensors`) and `Gemma4Recipe` (`new`, `new_edge_ple`, `new_moe_26b_a4b`, `from_metadata`, `from_gguf`, `from_safetensors`).
- **Sparse MoE & Edge PLE**: `ExpertWeights`, `select_top_k_experts`, `dispatch_sparse_moe_layer`, and `apply_gemma4_ple_residual`.
- **KV & Math Helpers**: `write_and_slice_sliding_kv` (self-contained sliding-window KV ring writer) and `pub fn matvec_any` (parallelized across Rayon worker pools via `rayon::current_num_threads() > 1`).

## Key Architectural Invariants

1. **Flat Peer Crate Boundary**: Never add a dependency on `jero-engine`. Compute parallelism uses `rayon` directly inside `matvec_any` and `ExecutionContext::compute_logits`.
2. **ADR-045 Anti-Stub Fail-Closed Execution**: Never reintroduce synthetic slice projections or zero-filled fallback tensors. Missing, out-of-bounds, non-finite, or dimension-mismatched tensors (`token_embd.weight`, `blk.{l}.attn_{q,k,v,output}.weight`, `blk.{l}.ffn_{gate,up,down}.weight`, `output_norm.weight`, or PLE/MoE tensors when configured) must return `K3Error::TensorBoundsError` or `K3Error::CapabilityMismatch`.
3. **Dual Format Weight Resolution**: Lookup resolves both GGUF (`blk.{l}.*`, `token_embd.weight`, `output_norm.weight`, `output.weight`) and HuggingFace SafeTensors (`language_model.model.layers.{l}.*`, `model.layers.{l}.*`, `lm_head.weight`), preserving bit-exact output parity across `[in_dim, out_dim]` and `[out_dim, in_dim]` layouts.
4. **Layer 0 VLM Soft-Prompt Conditioning**: `prefill_step` reads `ExecutionContext::soft_prompt_embeds()` / `soft_prompt_dims()` before discrete `prompt_tokens`, validates `hidden_dim` and `n_ctx` headroom (`K3Error::ContextLengthExceeded`), and injects continuous multimodal prefix embeddings directly into Layer 0.
5. **License & Privacy Boundary**: `LGPL-3.0-only WITH LGPL-3.0-linking-exception`. Public code, docs, and tests must never reference private lab hostnames or LAN IPs.

## Local Verification Gates

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo check --target riscv64gc-unknown-linux-gnu --all-targets
cargo test --lib --tests
cargo test --no-default-features --lib --tests
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
MIRIFLAGS="-Zmiri-disable-isolation" cargo +nightly miri test --lib
cargo agents plugin validate Symposium.toml --no-check-crates
```
