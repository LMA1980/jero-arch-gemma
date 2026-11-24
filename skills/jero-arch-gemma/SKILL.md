---
name: jero-arch-gemma
description: Google Gemma 2, Gemma 3, and Gemma 4 decoder/VLM architecture recipes (`GemmaRecipe`, `Gemma4Recipe`) with `soft_prompt_embeds` Layer 0 support for the SpacemiT K3 platform.
---

# `jero-arch-gemma` Architecture Crate

Provides Gemma 2 / Gemma 3 / Gemma 4 architecture recipes (`GemmaRecipe`, `Gemma4Recipe`), logit soft-capping, interleaved SWA/global attention, and VLM `soft_prompt_embeds` Layer 0 injection for K3 inference.

## Crate Contract & Dependencies

- **License**: `LGPL-3.0-only WITH LGPL-3.0-linking-exception`
- **Upstream Dependencies**: `jero-core` (zero-copy `ModelWeights`, `ExecutionContext`, `ArchRecipe`, `K3Error`) and optional `jero-kernels-rvv` (enabled by default via `with-kernels`).
- **Safety & Lints**: `#![deny(unsafe_op_in_unsafe_fn)]`, `clippy::pedantic`, and `clippy::undocumented_unsafe_blocks`.

## Local Verification Gates

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --lib --tests
cargo test --no-default-features --lib --tests
MIRIFLAGS="-Zmiri-disable-isolation" cargo +nightly miri test --lib
```
