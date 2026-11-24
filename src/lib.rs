//! `jero-arch-gemma`: Google Gemma 2, Gemma 3, `FunctionGemma`, and Gemma 4 architecture recipes
//! for the `SpacemiT` K3 platform.
//!
//! Implements [`ArchRecipe`] for Gemma 2, Gemma 3, `FunctionGemma`, and Gemma 4 topologies,
//! integrating:
//! - Unit-offset `(1.0 + w)` `RMSNorm` and per-head QK-normalization,
//! - `GeGLU` FFN activations and attention/final logit softcapping (`gemma2`),
//! - Interleaved local sliding-window cyclic KV rings and global full-context attention
//!   with dual `RoPE` frequencies (`10_000.0` local vs `1_000_000.0` global),
//! - Zero-copy quantized Gemma 4 Edge (`E2B`/`E4B`) Per-Layer Embeddings (`PLE`),
//! - Gemma-4-26B-A4B top-`k` Sparse `MoE` deterministic softmax routing,
//! - Strict fail-closed `ModelWeights` lookup across `GGUF` and `SafeTensors` naming (ADR-045), and
//! - Layer 0 `ExecutionContext::soft_prompt_embeds()` prefix conditioning for VLM pipelines.

#![deny(unsafe_op_in_unsafe_fn)]

mod weights;

#[cfg(test)]
mod tests;

use jero_core::{
    ArchRecipe, ExecutionContext, GgufFile, K3Error, ModelMetadata, ModelWeights, QuantType,
    SafeTensorsFile, TensorDescriptor, TensorView,
};
use jero_kernels_rvv::{
    apply_rope_inplace, dequantize_token_row, geglu_f32, gqa_causal_attention_f32,
    logit_softcap_tanh_inplace, qk_norm_inplace, rmsnorm_unit_offset,
};

pub use weights::matvec_any;
use weights::{
    OUTPUT_HEAD_ALIASES, OUTPUT_NORM_ALIASES, PLE_EMBD_ALIASES, TOKEN_EMBD_ALIASES, capability_err,
    decode_1d_f32_strict, embed_token_hidden, is_supported_gemma_quant, lookup_layer_tensor,
    lookup_tensor_aliases, missing_tensor_err, require_1d_f32, require_layer_1d_f32,
    require_layer_2d_matvec, tensor_bounds_err, try_lookup_layer_tensor, try_lookup_tensor_aliases,
    validate_2d_matvec_view,
};

/// Canonical crate identifier for this architecture recipe package.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");

/// Canonical SPDX license identifier for this architecture recipe package.
pub const CRATE_LICENSE: &str = "LGPL-3.0-only WITH LGPL-3.0-linking-exception";

/// Returns the crate description configured in `Cargo.toml`.
#[must_use]
pub const fn crate_description() -> &'static str {
    env!("CARGO_PKG_DESCRIPTION")
}

/// Expert weight views for a single Gemma 4 sparse Mixture-of-Experts (`MoE`) expert block.
#[derive(Debug, Clone)]
pub struct ExpertWeights<'a> {
    /// Gate projection weights (`ffn_gate_exps`).
    pub gate: TensorView<'a>,
    /// Up projection weights (`ffn_up_exps`).
    pub up: TensorView<'a>,
    /// Down projection weights (`ffn_down_exps`).
    pub down: TensorView<'a>,
}

/// Writes `(k_in, v_in)` at cyclic ring slot `pos % window_size` into a local sliding-window
/// layer buffer `[k_ring (window_size * kv_dim), v_ring (window_size * kv_dim)]` and returns
/// `(k_slice, v_slice, active_len)` bounded to `min(pos + 1, window_size)`.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `window_size == 0`, `kv_dim == 0`,
/// `k_in.len() != kv_dim`, `v_in.len() != kv_dim`, buffer dimensions overflow `usize`,
/// or `layer_kv.len() < 2 * window_size * kv_dim`.
pub fn write_and_slice_sliding_kv<'a>(
    layer_kv: &'a mut [f32],
    pos: usize,
    window_size: usize,
    kv_dim: usize,
    k_in: &[f32],
    v_in: &[f32],
) -> Result<(&'a [f32], &'a [f32], usize), K3Error> {
    if window_size == 0 || kv_dim == 0 {
        return Err(missing_tensor_err(
            "kv_cache",
            format!("sliding window ({window_size}) and kv_dim ({kv_dim}) must be > 0"),
        ));
    }
    if k_in.len() != kv_dim || v_in.len() != kv_dim {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "kv_cache".to_owned(),
            offset: 0,
            byte_len: size_of_val(k_in),
            buffer_len: size_of_val(v_in),
            reason: format!(
                "k_in len ({}) and v_in len ({}) must equal kv_dim ({kv_dim})",
                k_in.len(),
                v_in.len()
            ),
        });
    }
    let ring_elems = window_size.checked_mul(kv_dim).ok_or_else(|| {
        missing_tensor_err("kv_cache", "sliding-window ring elements overflowed usize")
    })?;
    let required_elems = ring_elems.checked_mul(2).ok_or_else(|| {
        missing_tensor_err(
            "kv_cache",
            "sliding-window K/V ring length overflowed usize",
        )
    })?;
    if layer_kv.len() < required_elems {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "kv_cache".to_owned(),
            offset: 0,
            byte_len: required_elems.saturating_mul(size_of::<f32>()),
            buffer_len: size_of_val(layer_kv),
            reason: format!(
                "layer_kv len {} < required {required_elems}",
                layer_kv.len()
            ),
        });
    }

    let slot = pos % window_size;
    let active_len = pos.saturating_add(1).min(window_size);
    let slot_start = slot * kv_dim;
    let slot_end = slot_start + kv_dim;

    let (k_ring, rest) = layer_kv.split_at_mut(ring_elems);
    let v_ring = &mut rest[..ring_elems];

    k_ring[slot_start..slot_end].copy_from_slice(k_in);
    v_ring[slot_start..slot_end].copy_from_slice(v_in);

    let active_elems = active_len * kv_dim;
    Ok((&k_ring[..active_elems], &v_ring[..active_elems], active_len))
}

fn infer_ple_row_width(
    ple_embd: &TensorView<'_>,
    ple_dim: usize,
    layer_idx: usize,
    token_idx: usize,
) -> Result<usize, K3Error> {
    let min_cols = layer_idx
        .checked_add(1)
        .and_then(|n| n.checked_mul(ple_dim))
        .ok_or_else(|| tensor_bounds_err(ple_embd, "PLE layer column offset overflowed usize"))?;
    let blk = ple_embd.quant().block_elements().max(1);
    let shape = ple_embd.shape();

    if shape.len() == 2 {
        let (d0, d1) = (shape[0], shape[1]);
        let d0_valid = d0 >= min_cols && d0 % ple_dim == 0 && d0 % blk == 0 && token_idx < d1;
        let d1_valid = d1 >= min_cols && d1 % ple_dim == 0 && d1 % blk == 0 && token_idx < d0;
        if d0_valid {
            return Ok(d0);
        }
        if d1_valid {
            return Ok(d1);
        }
    }

    let total_elems = ple_embd.num_elements();
    let max_rows = token_idx + 1;
    if total_elems >= min_cols && min_cols % blk == 0 && total_elems % min_cols == 0 {
        let rows = total_elems / min_cols;
        if token_idx < rows {
            return Ok(min_cols);
        }
    }
    if max_rows > 0 && total_elems % max_rows == 0 {
        let cols = total_elems / max_rows;
        if cols >= min_cols && cols % ple_dim == 0 && cols % blk == 0 {
            return Ok(cols);
        }
    }

    Err(tensor_bounds_err(
        ple_embd,
        format!(
            "cannot infer block-aligned PLE row width for shape {shape:?}, ple_dim={ple_dim}, layer={layer_idx}, token={token_idx}",
        ),
    ))
}

/// Applies a Gemma 4 Edge (`E2B`/`E4B`) Per-Layer Embedding (`PLE`) quantized token lookup,
/// `GeGLU` gated projection, and unit-offset `RMSNorm` residual addition in-place onto `hidden`.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if tensor shapes, buffer bounds, norm weights, or slice
/// dimensions fail validation, or [`K3Error::CapabilityMismatch`] if an unsupported quantization
/// format is used.
#[expect(
    clippy::too_many_arguments,
    reason = "contract matches explicit PLE tensor inputs across jero-engine callers"
)]
pub fn apply_gemma4_ple_residual(
    hidden: &mut [f32],
    token_id: u32,
    layer_idx: usize,
    ple_embd: &TensorView<'_>,
    ple_inp_gate: &TensorView<'_>,
    ple_proj: &TensorView<'_>,
    ple_norm_weight: &[f32],
    eps: f32,
) -> Result<(), K3Error> {
    let hidden_dim = hidden.len();
    if hidden_dim == 0 || ple_norm_weight.len() != hidden_dim {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "per_layer_norm".to_owned(),
            offset: 0,
            byte_len: size_of_val(ple_norm_weight),
            buffer_len: size_of_val(hidden),
            reason: format!(
                "PLE hidden_dim ({hidden_dim}) must be > 0 and match ple_norm_weight len ({})",
                ple_norm_weight.len()
            ),
        });
    }
    if !eps.is_finite() || eps < 0.0 {
        return Err(missing_tensor_err(
            "per_layer_norm",
            format!("invalid PLE RMSNorm epsilon: {eps}"),
        ));
    }
    for (idx, &w) in ple_norm_weight.iter().enumerate() {
        if !w.is_finite() {
            return Err(missing_tensor_err(
                "per_layer_norm",
                format!("non-finite PLE norm weight ({w}) at index {idx}"),
            ));
        }
    }
    if !is_supported_gemma_quant(ple_embd.quant()) {
        return Err(K3Error::unsupported_tensor_quant(
            "gemma4",
            ple_embd.name(),
            ple_embd.quant(),
        ));
    }

    let gate_elems = ple_inp_gate.num_elements();
    if gate_elems == 0 || gate_elems % hidden_dim != 0 {
        return Err(tensor_bounds_err(
            ple_inp_gate,
            format!(
                "ple_inp_gate elements ({gate_elems}) must be a positive multiple of hidden_dim ({hidden_dim})"
            ),
        ));
    }
    let ple_dim = gate_elems / hidden_dim;
    let _ = validate_2d_matvec_view(*ple_inp_gate, "gemma4", hidden_dim, ple_dim)?;
    let _ = validate_2d_matvec_view(*ple_proj, "gemma4", ple_dim, hidden_dim)?;

    let token_idx = usize::try_from(token_id)
        .map_err(|_| tensor_bounds_err(ple_embd, "token_id exceeds usize range"))?;
    let row_width = infer_ple_row_width(ple_embd, ple_dim, layer_idx, token_idx)?;
    let mut full_row = vec![0.0_f32; row_width];
    dequantize_token_row(ple_embd, token_idx, &mut full_row)?;

    let layer_start = layer_idx * ple_dim;
    let layer_ple = &full_row[layer_start..layer_start + ple_dim];

    let mut gate_out = vec![0.0_f32; ple_dim];
    matvec_any(&mut gate_out, ple_inp_gate, hidden)?;

    let mut gated = vec![0.0_f32; ple_dim];
    geglu_f32(&mut gated, &gate_out, layer_ple)?;

    let mut proj_out = vec![0.0_f32; hidden_dim];
    matvec_any(&mut proj_out, ple_proj, &gated)?;

    let mut normed = vec![0.0_f32; hidden_dim];
    rmsnorm_unit_offset(&mut normed, &proj_out, ple_norm_weight, eps)?;

    for (h, &delta) in hidden.iter_mut().zip(normed.iter()) {
        *h += delta;
    }
    Ok(())
}

/// Selects the top-`k` experts deterministically from `router_logits` with ascending index
/// tie-breaking, returning `(expert_idx, gating_weight)` pairs.
///
/// When `normalize` is `true`, the selected top-`k` weights are renormalized to sum to `1.0`.
/// When `normalize` is `false`, raw full-router softmax probabilities are returned.
#[must_use]
pub fn select_top_k_experts(
    router_logits: &[f32],
    top_k: usize,
    normalize: bool,
) -> Vec<(usize, f32)> {
    if router_logits.is_empty() || top_k == 0 {
        return Vec::new();
    }

    let max_logit = router_logits
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);

    let mut probs = Vec::with_capacity(router_logits.len());
    let mut sum_exp = 0.0_f32;
    for &logit in router_logits {
        let e = if logit.is_finite() && max_logit.is_finite() {
            (logit - max_logit).exp()
        } else {
            0.0_f32
        };
        probs.push(e);
        sum_exp += e;
    }

    if sum_exp > 0.0_f32 && sum_exp.is_finite() {
        for p in &mut probs {
            *p /= sum_exp;
        }
    } else {
        #[expect(
            clippy::cast_precision_loss,
            reason = "expert count fits comfortably in f32"
        )]
        let uniform = 1.0_f32 / (router_logits.len() as f32);
        probs.fill(uniform);
    }

    let mut ranked: Vec<(usize, f32, f32)> = router_logits
        .iter()
        .copied()
        .zip(probs)
        .enumerate()
        .map(|(idx, (logit, prob))| {
            let finite_logit = if logit.is_finite() {
                logit
            } else {
                f32::NEG_INFINITY
            };
            (idx, finite_logit, prob)
        })
        .collect();

    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let k = top_k.min(ranked.len());
    let mut selected: Vec<(usize, f32)> = ranked
        .into_iter()
        .take(k)
        .map(|(idx, _, prob)| (idx, prob))
        .collect();

    if normalize {
        let selected_sum: f32 = selected.iter().map(|&(_, w)| w).sum();
        if selected_sum > 0.0_f32 && selected_sum.is_finite() {
            for (_, w) in &mut selected {
                *w /= selected_sum;
            }
        } else {
            #[expect(clippy::cast_precision_loss, reason = "top_k fits comfortably in f32")]
            let uniform = 1.0_f32 / (k as f32);
            for (_, w) in &mut selected {
                *w = uniform;
            }
        }
    }

    selected
}

/// Evaluates a Gemma-4 sparse `MoE` layer by computing router logits, selecting top-`k`
/// experts deterministically with normalized softmax weights, and accumulating weighted
/// `GeGLU` expert outputs only for active experts.
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if `experts` is empty, `top_k == 0`, an unsupported
/// quantization is used, or router logits are non-finite. Returns [`K3Error::TensorBoundsError`]
/// on dimension or buffer errors.
pub fn dispatch_sparse_moe_layer(
    out: &mut [f32],
    hidden: &[f32],
    router_weight: &TensorView<'_>,
    experts: &[ExpertWeights<'_>],
    top_k: usize,
) -> Result<(), K3Error> {
    let hidden_dim = hidden.len();
    if hidden_dim == 0 || out.len() != hidden_dim {
        return Err(K3Error::TensorBoundsError {
            tensor_name: router_weight.name().to_owned(),
            offset: router_weight.offset(),
            byte_len: size_of_val(out),
            buffer_len: size_of_val(hidden),
            reason: format!(
                "MoE hidden ({hidden_dim}) must be > 0 and match out len ({})",
                out.len()
            ),
        });
    }
    if experts.is_empty() || top_k == 0 || top_k > experts.len() {
        return Err(capability_err(
            "gemma4",
            format!(
                "invalid sparse MoE config: experts={}, top_k={top_k}",
                experts.len()
            ),
        ));
    }
    let _ = validate_2d_matvec_view(*router_weight, "gemma4", hidden_dim, experts.len())?;

    let mut router_logits = vec![0.0_f32; experts.len()];
    matvec_any(&mut router_logits, router_weight, hidden)?;
    for (idx, &val) in router_logits.iter().enumerate() {
        if !val.is_finite() {
            return Err(capability_err(
                "gemma4",
                format!("non-finite MoE router logit {val} at expert index {idx}"),
            ));
        }
    }

    let selected = select_top_k_experts(&router_logits, top_k, true);
    out.fill(0.0);

    for (expert_idx, gate_weight) in selected {
        let exp = &experts[expert_idx];
        let gate_elems = exp.gate.num_elements();
        if gate_elems == 0 || gate_elems % hidden_dim != 0 {
            return Err(tensor_bounds_err(
                &exp.gate,
                format!(
                    "expert gate elements ({gate_elems}) not divisible by hidden_dim ({hidden_dim})"
                ),
            ));
        }
        let ffn_dim = gate_elems / hidden_dim;
        let _ = validate_2d_matvec_view(exp.gate, "gemma4", hidden_dim, ffn_dim)?;
        let _ = validate_2d_matvec_view(exp.up, "gemma4", hidden_dim, ffn_dim)?;
        let _ = validate_2d_matvec_view(exp.down, "gemma4", ffn_dim, hidden_dim)?;

        let mut gate_buf = vec![0.0_f32; ffn_dim];
        let mut up_buf = vec![0.0_f32; ffn_dim];
        matvec_any(&mut gate_buf, &exp.gate, hidden)?;
        matvec_any(&mut up_buf, &exp.up, hidden)?;

        let mut act_buf = vec![0.0_f32; ffn_dim];
        geglu_f32(&mut act_buf, &gate_buf, &up_buf)?;

        let mut expert_out = vec![0.0_f32; hidden_dim];
        matvec_any(&mut expert_out, &exp.down, &act_buf)?;

        for (dst, &src) in out.iter_mut().zip(expert_out.iter()) {
            *dst += gate_weight * src;
        }
    }

    Ok(())
}

/// Google Gemma 2, Gemma 3, and `FunctionGemma` architecture recipe implementation.
#[derive(Debug, Clone)]
pub struct GemmaRecipe {
    /// Exact architecture identifier (`"gemma2"`, `"gemma3"`, or `"functiongemma"`).
    pub architecture: &'static str,
    /// Transformer block layer count.
    pub num_layers: usize,
    /// Model embedding and hidden dimension.
    pub hidden_dim: usize,
    /// Feed-forward intermediate dimension.
    pub ffn_dim: usize,
    /// Query attention head count.
    pub n_heads_q: usize,
    /// Key/value head count (GQA/MQA).
    pub n_heads_kv: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Unit-offset `RMSNorm` epsilon.
    pub rms_eps: f32,
    /// Local sliding-window attention `RoPE` base frequency (`10_000.0`).
    pub rope_freq_base_local: f32,
    /// Global full-context attention `RoPE` base frequency (`1_000_000.0` on Gemma 3).
    pub rope_freq_base_global: f32,
    /// Local attention sliding-window capacity in tokens (`4096` on Gemma 2, `1024` on Gemma 3).
    pub sliding_window: usize,
    /// Optional attention score softcapping (`50.0` on Gemma 2).
    pub attn_logit_softcap: Option<f32>,
    /// Optional final LM-head logit softcapping (`30.0` on Gemma 2).
    pub final_logit_softcap: Option<f32>,
    /// Enables per-head Q/K unit-offset normalization (`true` on Gemma 3).
    pub use_qk_norm: bool,
}

impl Default for GemmaRecipe {
    fn default() -> Self {
        Self::new_gemma3()
    }
}

impl GemmaRecipe {
    /// Creates a default Gemma 2 recipe (1:1 local/global attention, `4096` SWA, softcapping).
    #[must_use]
    pub const fn new_gemma2() -> Self {
        Self {
            architecture: "gemma2",
            num_layers: 26,
            hidden_dim: 2304,
            ffn_dim: 9216,
            n_heads_q: 8,
            n_heads_kv: 4,
            head_dim: 256,
            vocab_size: 256_000,
            rms_eps: 1e-6,
            rope_freq_base_local: 10_000.0,
            rope_freq_base_global: 10_000.0,
            sliding_window: 4096,
            attn_logit_softcap: Some(50.0),
            final_logit_softcap: Some(30.0),
            use_qk_norm: false,
        }
    }

    /// Creates a default Gemma 3 recipe (5:1 local/global attention, `1024` SWA, QK-norm, dual `RoPE`).
    #[must_use]
    pub const fn new_gemma3() -> Self {
        Self {
            architecture: "gemma3",
            num_layers: 34,
            hidden_dim: 2560,
            ffn_dim: 10_240,
            n_heads_q: 8,
            n_heads_kv: 4,
            head_dim: 256,
            vocab_size: 262_144,
            rms_eps: 1e-6,
            rope_freq_base_local: 10_000.0,
            rope_freq_base_global: 1_000_000.0,
            sliding_window: 1024,
            attn_logit_softcap: None,
            final_logit_softcap: None,
            use_qk_norm: true,
        }
    }

    /// Creates a default `FunctionGemma` recipe (Gemma 3 270M/2B specialized functional action SLM).
    #[must_use]
    pub const fn new_functiongemma() -> Self {
        Self {
            architecture: "functiongemma",
            num_layers: 18,
            hidden_dim: 640,
            ffn_dim: 2048,
            n_heads_q: 4,
            n_heads_kv: 1,
            head_dim: 256,
            vocab_size: 262_144,
            rms_eps: 1e-6,
            rope_freq_base_local: 10_000.0,
            rope_freq_base_global: 1_000_000.0,
            sliding_window: 512,
            attn_logit_softcap: None,
            final_logit_softcap: None,
            use_qk_norm: true,
        }
    }

    /// Hydrates a [`GemmaRecipe`] from parsed [`ModelMetadata`] across both `GGUF` and
    /// `SafeTensors` naming conventions.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] if `metadata.architecture()` is not `"gemma2"`,
    /// `"gemma3"`, `"gemma3_text"`, or `"functiongemma"`.
    pub fn from_metadata(metadata: &ModelMetadata) -> Result<Self, K3Error> {
        let mut recipe = match metadata.architecture() {
            "gemma2" => Self::new_gemma2(),
            "gemma3" | "gemma3_text" => Self::new_gemma3(),
            "functiongemma" => Self::new_functiongemma(),
            other => {
                return Err(capability_err(
                    other,
                    format!(
                        "GemmaRecipe supports 'gemma2', 'gemma3', and 'functiongemma', got '{other}'"
                    ),
                ));
            }
        };

        let hydrated = HydratedCommon::from_metadata(metadata);
        if hydrated.num_layers > 0 {
            recipe.num_layers = hydrated.num_layers;
        }
        if hydrated.hidden_dim > 0 {
            recipe.hidden_dim = hydrated.hidden_dim;
        }
        if hydrated.ffn_dim > 0 {
            recipe.ffn_dim = hydrated.ffn_dim;
        }
        if hydrated.n_heads_q > 0 {
            recipe.n_heads_q = hydrated.n_heads_q;
        }
        if hydrated.n_heads_kv > 0 {
            recipe.n_heads_kv = hydrated.n_heads_kv;
        }
        if hydrated.head_dim >= 2 {
            recipe.head_dim = hydrated.head_dim;
        } else if recipe.n_heads_q > 0 && recipe.hidden_dim >= recipe.n_heads_q {
            recipe.head_dim = recipe.hidden_dim / recipe.n_heads_q;
        }
        if hydrated.vocab_size > 0 {
            recipe.vocab_size = hydrated.vocab_size;
        }
        if let Some(eps) = hydrated.rms_eps {
            recipe.rms_eps = eps;
        }
        if let Some(freq) = hydrated.rope_freq_base {
            recipe.rope_freq_base_global = freq;
            if recipe.architecture == "gemma2" {
                recipe.rope_freq_base_local = freq;
            }
        }
        if let Some(sw) = hydrated.sliding_window {
            recipe.sliding_window = sw;
        }
        if let Some(cap) = hydrated.attn_logit_softcap {
            recipe.attn_logit_softcap = Some(cap);
        }
        if let Some(cap) = hydrated.final_logit_softcap {
            recipe.final_logit_softcap = Some(cap);
        }
        Ok(recipe)
    }

    /// Hydrates a [`GemmaRecipe`] from a parsed [`GgufFile`].
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] if the model architecture is unsupported.
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Result<Self, K3Error> {
        let mut recipe = Self::from_metadata(gguf.metadata())?;
        let local_key = format!("{}.rope.freq_base_local", gguf.architecture());
        if let Some(local_theta) = gguf
            .metadata_value(&local_key)
            .or_else(|| gguf.metadata_value("rope.freq_base_local"))
            .and_then(jero_core::GgufValue::as_f32)
            .filter(|v| v.is_finite() && *v > 0.0)
        {
            recipe.rope_freq_base_local = local_theta;
        }
        Ok(recipe)
    }

    /// Hydrates a [`GemmaRecipe`] from a parsed [`SafeTensorsFile`].
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] if the model architecture is unsupported.
    pub fn from_safetensors(st: &SafeTensorsFile<'_>) -> Result<Self, K3Error> {
        let mut recipe = Self::from_metadata(st.metadata())?;
        let arch_local_key = format!("{}.rope.freq_base_local", st.architecture());
        if let Some(local_theta) = parse_st_f32(
            st,
            &[
                "rope_local_base_freq",
                &arch_local_key,
                "rope.freq_base_local",
            ],
        ) {
            recipe.rope_freq_base_local = local_theta;
        }
        if let Some(cap) = parse_st_f32(st, &["attn_logit_softcapping"]) {
            recipe.attn_logit_softcap = Some(cap);
        }
        if let Some(cap) = parse_st_f32(st, &["final_logit_softcapping"]) {
            recipe.final_logit_softcap = Some(cap);
        }
        Ok(recipe)
    }

    /// Returns `true` when `layer_idx` uses global full-context attention.
    #[must_use]
    pub const fn is_global_layer(&self, layer_idx: usize) -> bool {
        if matches!(self.architecture.as_bytes(), b"gemma2") {
            layer_idx % 2 == 1
        } else {
            (layer_idx + 1) % 6 == 0
        }
    }

    /// Returns the layer-appropriate `RoPE` base frequency (`rope_freq_base_global` vs `rope_freq_base_local`).
    #[must_use]
    pub const fn layer_rope_theta(&self, layer_idx: usize) -> f32 {
        if self.is_global_layer(layer_idx) {
            self.rope_freq_base_global
        } else {
            self.rope_freq_base_local
        }
    }

    fn layer_config(&self) -> Result<GemmaLayerConfig, K3Error> {
        GemmaLayerConfig::validated(GemmaLayerConfig {
            architecture: self.architecture,
            num_layers: self.num_layers,
            hidden_dim: self.hidden_dim,
            ffn_dim: self.ffn_dim,
            n_heads_q: self.n_heads_q,
            n_heads_kv: self.n_heads_kv,
            head_dim: self.head_dim,
            rms_eps: self.rms_eps,
            rope_freq_base_local: self.rope_freq_base_local,
            rope_freq_base_global: self.rope_freq_base_global,
            sliding_window: self.sliding_window,
            attn_logit_softcap: self.attn_logit_softcap,
            final_logit_softcap: self.final_logit_softcap,
            use_qk_norm: self.use_qk_norm,
            ple_dim: None,
            num_experts: None,
            num_experts_per_tok: None,
        })
    }
}

/// Google Gemma 4 architecture recipe supporting dense+PLE (`E2B`/`E4B`) and Sparse `MoE` (`26B-A4B`).
#[derive(Debug, Clone)]
pub struct Gemma4Recipe {
    /// Exact architecture identifier (`"gemma4"`).
    pub architecture: &'static str,
    /// Transformer block layer count.
    pub num_layers: usize,
    /// Model embedding and hidden dimension.
    pub hidden_dim: usize,
    /// Feed-forward intermediate dimension.
    pub ffn_dim: usize,
    /// Query attention head count.
    pub n_heads_q: usize,
    /// Key/value head count.
    pub n_heads_kv: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Unit-offset `RMSNorm` epsilon.
    pub rms_eps: f32,
    /// Local sliding-window attention `RoPE` base frequency (`10_000.0`).
    pub rope_freq_base_local: f32,
    /// Global full-context attention `RoPE` base frequency (`1_000_000.0`).
    pub rope_freq_base_global: f32,
    /// Local sliding-window span in tokens (`1024`).
    pub sliding_window: usize,
    /// Optional Per-Layer Embedding (`PLE`) width for Edge (`E2B`/`E4B`) models.
    pub ple_dim: Option<usize>,
    /// Optional Sparse `MoE` total routed expert count (`128` for `Gemma-4-26B-A4B`).
    pub num_experts: Option<usize>,
    /// Optional Sparse `MoE` active expert count per token (`8` for `Gemma-4-26B-A4B`).
    pub num_experts_per_tok: Option<usize>,
}

impl Default for Gemma4Recipe {
    fn default() -> Self {
        Self::new()
    }
}

impl Gemma4Recipe {
    /// Creates a baseline `gemma4` architecture recipe with 5:1 hybrid local/global attention.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            architecture: "gemma4",
            num_layers: 36,
            hidden_dim: 2560,
            ffn_dim: 10_240,
            n_heads_q: 8,
            n_heads_kv: 4,
            head_dim: 256,
            vocab_size: 262_144,
            rms_eps: 1e-6,
            rope_freq_base_local: 10_000.0,
            rope_freq_base_global: 1_000_000.0,
            sliding_window: 1024,
            ple_dim: None,
            num_experts: None,
            num_experts_per_tok: None,
        }
    }

    /// Creates a `gemma4` Edge (`E2B`/`E4B`) recipe configured with Per-Layer Embeddings (`ple_dim`).
    #[must_use]
    pub const fn new_edge_ple(ple_dim: usize) -> Self {
        let mut r = Self::new();
        r.ple_dim = Some(ple_dim);
        r
    }

    /// Creates a `gemma4` `26B-A4B` Sparse `MoE` recipe (`128` experts, top-`8` active routing).
    #[must_use]
    pub const fn new_moe_26b_a4b() -> Self {
        let mut r = Self::new();
        r.hidden_dim = 3072;
        r.ffn_dim = 2048;
        r.num_experts = Some(128);
        r.num_experts_per_tok = Some(8);
        r
    }

    /// Hydrates a [`Gemma4Recipe`] from parsed [`ModelMetadata`] across both `GGUF` and
    /// `SafeTensors` naming conventions.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] if `metadata.architecture()` is not `"gemma4"` or
    /// `"gemma4_text"`.
    pub fn from_metadata(metadata: &ModelMetadata) -> Result<Self, K3Error> {
        if !matches!(metadata.architecture(), "gemma4" | "gemma4_text") {
            return Err(capability_err(
                metadata.architecture(),
                format!(
                    "Gemma4Recipe supports 'gemma4', got '{}'",
                    metadata.architecture()
                ),
            ));
        }

        let mut recipe = Self::new();
        let hydrated = HydratedCommon::from_metadata(metadata);
        if hydrated.num_layers > 0 {
            recipe.num_layers = hydrated.num_layers;
        }
        if hydrated.hidden_dim > 0 {
            recipe.hidden_dim = hydrated.hidden_dim;
        }
        if hydrated.ffn_dim > 0 {
            recipe.ffn_dim = hydrated.ffn_dim;
        }
        if hydrated.n_heads_q > 0 {
            recipe.n_heads_q = hydrated.n_heads_q;
        }
        if hydrated.n_heads_kv > 0 {
            recipe.n_heads_kv = hydrated.n_heads_kv;
        }
        if hydrated.head_dim >= 2 {
            recipe.head_dim = hydrated.head_dim;
        } else if recipe.n_heads_q > 0 && recipe.hidden_dim >= recipe.n_heads_q {
            recipe.head_dim = recipe.hidden_dim / recipe.n_heads_q;
        }
        if hydrated.vocab_size > 0 {
            recipe.vocab_size = hydrated.vocab_size;
        }
        if let Some(eps) = hydrated.rms_eps {
            recipe.rms_eps = eps;
        }
        if let Some(freq) = hydrated.rope_freq_base {
            recipe.rope_freq_base_global = freq;
        }
        if let Some(sw) = hydrated.sliding_window {
            recipe.sliding_window = sw;
        }
        if let Some(n_exp) = metadata.expert_count().filter(|&v| v > 0) {
            recipe.num_experts = Some(n_exp);
        }
        if let Some(top_k) = metadata.expert_used_count().filter(|&v| v > 0) {
            recipe.num_experts_per_tok = Some(top_k);
        }
        if let Some(ple_gate) = find_desc_layer0(
            metadata.tensors(),
            "per_layer_inp_gate.weight",
            &["per_layer_input_gate.weight"],
        ) && let Some(ple_dim) = infer_other_2d_dim(ple_gate, recipe.hidden_dim)
        {
            recipe.ple_dim = Some(ple_dim);
        }
        Ok(recipe)
    }

    /// Hydrates a [`Gemma4Recipe`] from a parsed [`GgufFile`].
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] if the model architecture is unsupported.
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Result<Self, K3Error> {
        let mut recipe = Self::from_metadata(gguf.metadata())?;
        let local_key = format!("{}.rope.freq_base_local", gguf.architecture());
        if let Some(local_theta) = gguf
            .metadata_value(&local_key)
            .or_else(|| gguf.metadata_value("rope.freq_base_local"))
            .and_then(jero_core::GgufValue::as_f32)
            .filter(|v| v.is_finite() && *v > 0.0)
        {
            recipe.rope_freq_base_local = local_theta;
        }
        if recipe.ple_dim.is_none()
            && let Some(ple_dim) = gguf
                .metadata_value("gemma4.per_layer_embedding_length")
                .or_else(|| gguf.metadata_value("gemma4.ple_dim"))
                .and_then(jero_core::GgufValue::as_usize)
                .filter(|&v| v > 0)
        {
            recipe.ple_dim = Some(ple_dim);
        }
        Ok(recipe)
    }

    /// Hydrates a [`Gemma4Recipe`] from a parsed [`SafeTensorsFile`].
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] if the model architecture is unsupported.
    pub fn from_safetensors(st: &SafeTensorsFile<'_>) -> Result<Self, K3Error> {
        let mut recipe = Self::from_metadata(st.metadata())?;
        let arch_local_key = format!("{}.rope.freq_base_local", st.architecture());
        if let Some(local_theta) = parse_st_f32(
            st,
            &[
                "rope_local_base_freq",
                &arch_local_key,
                "rope.freq_base_local",
            ],
        ) {
            recipe.rope_freq_base_local = local_theta;
        }
        if recipe.ple_dim.is_none()
            && let Some(ple_dim) = parse_st_usize(st, &["per_layer_embedding_length", "ple_dim"])
        {
            recipe.ple_dim = Some(ple_dim);
        }
        Ok(recipe)
    }

    /// Returns `true` when `layer_idx` uses global full-context attention (5:1 local-to-global ratio).
    #[must_use]
    pub const fn is_global_layer(&self, layer_idx: usize) -> bool {
        (layer_idx + 1) % 6 == 0
    }

    /// Returns the layer-appropriate `RoPE` base frequency (`1_000_000.0` global vs `10_000.0` local).
    #[must_use]
    pub const fn layer_rope_theta(&self, layer_idx: usize) -> f32 {
        if self.is_global_layer(layer_idx) {
            self.rope_freq_base_global
        } else {
            self.rope_freq_base_local
        }
    }

    fn layer_config(&self) -> Result<GemmaLayerConfig, K3Error> {
        GemmaLayerConfig::validated(GemmaLayerConfig {
            architecture: self.architecture,
            num_layers: self.num_layers,
            hidden_dim: self.hidden_dim,
            ffn_dim: self.ffn_dim,
            n_heads_q: self.n_heads_q,
            n_heads_kv: self.n_heads_kv,
            head_dim: self.head_dim,
            rms_eps: self.rms_eps,
            rope_freq_base_local: self.rope_freq_base_local,
            rope_freq_base_global: self.rope_freq_base_global,
            sliding_window: self.sliding_window,
            attn_logit_softcap: None,
            final_logit_softcap: None,
            use_qk_norm: true,
            ple_dim: self.ple_dim,
            num_experts: self.num_experts,
            num_experts_per_tok: self.num_experts_per_tok,
        })
    }
}

struct HydratedCommon {
    num_layers: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_eps: Option<f32>,
    rope_freq_base: Option<f32>,
    sliding_window: Option<usize>,
    attn_logit_softcap: Option<f32>,
    final_logit_softcap: Option<f32>,
}

impl HydratedCommon {
    fn from_metadata(metadata: &ModelMetadata) -> Self {
        let tensors = metadata.tensors();
        let num_layers = if metadata.layer_count() > 0 {
            metadata.layer_count()
        } else {
            infer_layer_count_from_tensors(tensors)
        };
        let hidden_dim = if metadata.embedding_dim() > 0 {
            metadata.embedding_dim()
        } else {
            infer_hidden_dim_from_tensors(tensors)
        };
        let ffn_dim = metadata
            .feed_forward_length()
            .filter(|&v| v > 0)
            .or_else(|| infer_ffn_dim_from_tensors(tensors, hidden_dim))
            .unwrap_or(0);
        let mut n_heads_q = metadata.head_count();
        let mut n_heads_kv = metadata.head_count_kv();

        let head_dim = metadata
            .key_length()
            .filter(|&kl| kl >= 2)
            .or_else(|| {
                find_desc_layer0(tensors, "attn_q_norm.weight", &["self_attn.q_norm.weight"])
                    .filter(|d| d.shape().len() == 1 && d.shape()[0] >= 2)
                    .map(|d| d.shape()[0])
            })
            .or_else(|| {
                if n_heads_kv > 0
                    && let Some(k_desc) =
                        find_desc_layer0(tensors, "attn_k.weight", &["self_attn.k_proj.weight"])
                    && let Some(k_out) = infer_other_2d_dim(k_desc, hidden_dim)
                    && k_out % n_heads_kv == 0
                {
                    return Some(k_out / n_heads_kv);
                }
                if n_heads_q > 0
                    && let Some(q_desc) =
                        find_desc_layer0(tensors, "attn_q.weight", &["self_attn.q_proj.weight"])
                    && let Some(q_out) = infer_other_2d_dim(q_desc, hidden_dim)
                    && q_out % n_heads_q == 0
                {
                    return Some(q_out / n_heads_q);
                }
                None
            })
            .unwrap_or(0);

        if n_heads_q == 0
            && head_dim > 0
            && let Some(q_desc) =
                find_desc_layer0(tensors, "attn_q.weight", &["self_attn.q_proj.weight"])
            && let Some(q_out) = infer_other_2d_dim(q_desc, hidden_dim)
            && q_out % head_dim == 0
        {
            n_heads_q = q_out / head_dim;
        }
        if n_heads_kv == 0
            && head_dim > 0
            && let Some(k_desc) =
                find_desc_layer0(tensors, "attn_k.weight", &["self_attn.k_proj.weight"])
            && let Some(k_out) = infer_other_2d_dim(k_desc, hidden_dim)
            && k_out % head_dim == 0
        {
            n_heads_kv = k_out / head_dim;
        }

        let vocab_size = metadata
            .vocab_size()
            .filter(|&v| v > 0)
            .or_else(|| infer_vocab_from_tensors(tensors, hidden_dim))
            .unwrap_or(0);

        Self {
            num_layers,
            hidden_dim,
            ffn_dim,
            n_heads_q,
            n_heads_kv,
            head_dim,
            vocab_size,
            rms_eps: metadata
                .rms_norm_eps()
                .filter(|v| v.is_finite() && *v > 0.0),
            rope_freq_base: metadata
                .rope_freq_base()
                .filter(|v| v.is_finite() && *v > 0.0),
            sliding_window: metadata.sliding_window().filter(|&v| v > 0),
            attn_logit_softcap: metadata
                .attn_logit_softcapping()
                .filter(|v| v.is_finite() && *v > 0.0),
            final_logit_softcap: metadata
                .final_logit_softcapping()
                .filter(|v| v.is_finite() && *v > 0.0),
        }
    }
}

fn parse_st_f32(st: &SafeTensorsFile<'_>, keys: &[&str]) -> Option<f32> {
    for &key in keys {
        if let Some(val) = st
            .metadata_value(key)
            .and_then(|s| s.trim().parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
        {
            return Some(val);
        }
    }
    None
}

fn parse_st_usize(st: &SafeTensorsFile<'_>, keys: &[&str]) -> Option<usize> {
    for &key in keys {
        if let Some(val) = st
            .metadata_value(key)
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&v| v > 0)
        {
            return Some(val);
        }
    }
    None
}

fn extract_layer_index(name: &str) -> Option<usize> {
    const PREFIXES: &[&str] = &[
        "blk.",
        "model.layers.",
        "language_model.model.layers.",
        "model.language_model.layers.",
        "layers.",
    ];
    for prefix in PREFIXES {
        if let Some(rest) = name.strip_prefix(prefix)
            && let Some((idx_str, _)) = rest.split_once('.')
            && let Ok(idx) = idx_str.parse::<usize>()
        {
            return Some(idx);
        }
    }
    None
}

fn infer_layer_count_from_tensors(tensors: &[TensorDescriptor]) -> usize {
    tensors
        .iter()
        .filter_map(|d| extract_layer_index(d.name()))
        .max()
        .map_or(0, |idx| idx.saturating_add(1))
}

fn find_desc<'a>(tensors: &'a [TensorDescriptor], names: &[&str]) -> Option<&'a TensorDescriptor> {
    for &name in names {
        if let Some(desc) = tensors.iter().find(|t| t.name() == name) {
            return Some(desc);
        }
    }
    None
}

fn find_desc_layer0<'a>(
    tensors: &'a [TensorDescriptor],
    gguf_suffix: &str,
    hf_suffixes: &[&str],
) -> Option<&'a TensorDescriptor> {
    let gguf_name = format!("blk.0.{gguf_suffix}");
    if let Some(desc) = tensors.iter().find(|t| t.name() == gguf_name) {
        return Some(desc);
    }
    for prefix in [
        "model.layers.0.",
        "language_model.model.layers.0.",
        "model.language_model.layers.0.",
        "layers.0.",
    ] {
        for hf_suffix in hf_suffixes {
            let candidate = format!("{prefix}{hf_suffix}");
            if let Some(desc) = tensors.iter().find(|t| t.name() == candidate) {
                return Some(desc);
            }
        }
    }
    None
}

fn infer_other_2d_dim(desc: &TensorDescriptor, known_dim: usize) -> Option<usize> {
    if known_dim == 0 || desc.shape().len() != 2 {
        return None;
    }
    let (d0, d1) = (desc.shape()[0], desc.shape()[1]);
    if d0 == known_dim && d1 > 0 {
        Some(d1)
    } else if d1 == known_dim && d0 > 0 {
        Some(d0)
    } else {
        None
    }
}

fn infer_hidden_dim_from_tensors(tensors: &[TensorDescriptor]) -> usize {
    if let Some(desc) = find_desc(tensors, OUTPUT_NORM_ALIASES)
        && desc.shape().len() == 1
        && desc.shape()[0] > 0
    {
        return desc.shape()[0];
    }
    if let Some(desc) = find_desc_layer0(tensors, "attn_norm.weight", &["input_layernorm.weight"])
        && desc.shape().len() == 1
        && desc.shape()[0] > 0
    {
        return desc.shape()[0];
    }
    if let Some(desc) = find_desc(tensors, TOKEN_EMBD_ALIASES)
        && desc.shape().len() == 2
    {
        return if desc.name() == "token_embd.weight" {
            desc.shape()[0]
        } else {
            desc.shape()[1]
        };
    }
    0
}

fn infer_ffn_dim_from_tensors(tensors: &[TensorDescriptor], hidden_dim: usize) -> Option<usize> {
    for (gguf, hf) in [
        ("ffn_gate.weight", "mlp.gate_proj.weight"),
        ("ffn_up.weight", "mlp.up_proj.weight"),
        ("ffn_down.weight", "mlp.down_proj.weight"),
    ] {
        if let Some(desc) = find_desc_layer0(tensors, gguf, &[hf])
            && let Some(dim) = infer_other_2d_dim(desc, hidden_dim)
        {
            return Some(dim);
        }
    }
    None
}

fn infer_vocab_from_tensors(tensors: &[TensorDescriptor], hidden_dim: usize) -> Option<usize> {
    if let Some(desc) = find_desc(tensors, OUTPUT_HEAD_ALIASES)
        && let Some(rows) = infer_other_2d_dim(desc, hidden_dim)
    {
        return Some(rows);
    }
    if let Some(desc) = find_desc(tensors, TOKEN_EMBD_ALIASES)
        && let Some(rows) = infer_other_2d_dim(desc, hidden_dim)
    {
        return Some(rows);
    }
    None
}

#[derive(Debug, Clone, Copy)]
struct GemmaLayerConfig {
    architecture: &'static str,
    num_layers: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    rms_eps: f32,
    rope_freq_base_local: f32,
    rope_freq_base_global: f32,
    sliding_window: usize,
    attn_logit_softcap: Option<f32>,
    final_logit_softcap: Option<f32>,
    use_qk_norm: bool,
    ple_dim: Option<usize>,
    num_experts: Option<usize>,
    num_experts_per_tok: Option<usize>,
}

impl GemmaLayerConfig {
    fn validated(self) -> Result<Self, K3Error> {
        let arch = self.architecture;
        if self.num_layers == 0 || self.hidden_dim == 0 || self.sliding_window == 0 {
            return Err(capability_err(
                arch,
                format!(
                    "num_layers ({}), hidden_dim ({}), and sliding_window ({}) must be > 0",
                    self.num_layers, self.hidden_dim, self.sliding_window
                ),
            ));
        }
        if self.num_experts.is_none() && self.ffn_dim == 0 {
            return Err(capability_err(arch, "dense Gemma ffn_dim must be > 0"));
        }
        if self.n_heads_q == 0 || self.n_heads_kv == 0 || self.n_heads_q % self.n_heads_kv != 0 {
            return Err(capability_err(
                arch,
                format!(
                    "invalid GQA head counts: n_heads_q={}, n_heads_kv={}",
                    self.n_heads_q, self.n_heads_kv
                ),
            ));
        }
        if self.head_dim < 2 || self.head_dim % 2 != 0 {
            return Err(capability_err(
                arch,
                format!("head_dim ({}) must be even and >= 2", self.head_dim),
            ));
        }
        if !self.rms_eps.is_finite()
            || self.rms_eps <= 0.0
            || !self.rope_freq_base_local.is_finite()
            || self.rope_freq_base_local <= 0.0
            || !self.rope_freq_base_global.is_finite()
            || self.rope_freq_base_global <= 0.0
        {
            return Err(capability_err(
                arch,
                "rms_eps and RoPE base frequencies must be finite and > 0",
            ));
        }
        if let Some(cap) = self.attn_logit_softcap
            && (!cap.is_finite() || cap <= 0.0)
        {
            return Err(capability_err(
                arch,
                format!("attn_logit_softcap ({cap}) must be finite and > 0"),
            ));
        }
        if let Some(cap) = self.final_logit_softcap
            && (!cap.is_finite() || cap <= 0.0)
        {
            return Err(capability_err(
                arch,
                format!("final_logit_softcap ({cap}) must be finite and > 0"),
            ));
        }
        if self.ple_dim == Some(0) {
            return Err(capability_err(arch, "ple_dim must be > 0 when enabled"));
        }
        match (self.num_experts, self.num_experts_per_tok) {
            (Some(n_exp), Some(top_k)) if n_exp > 0 && top_k > 0 && top_k <= n_exp => {}
            (None, None) => {}
            (n_exp, top_k) => {
                return Err(capability_err(
                    arch,
                    format!("invalid MoE expert config: num_experts={n_exp:?}, top_k={top_k:?}"),
                ));
            }
        }
        Ok(self)
    }

    const fn is_global_layer(self, layer: usize) -> bool {
        if matches!(self.architecture.as_bytes(), b"gemma2") {
            layer % 2 == 1
        } else {
            (layer + 1) % 6 == 0
        }
    }
}

#[derive(Clone, Copy)]
enum TokenInput<'a> {
    TokenId(u32),
    SoftEmbed(&'a [f32]),
}

#[derive(Clone, Copy)]
struct StepPosition {
    pos: usize,
    seq_len: usize,
}

fn validate_kv_capacity(
    cfg: GemmaLayerConfig,
    kv_cache_len: usize,
    end_pos: usize,
) -> Result<(usize, usize), K3Error> {
    let kv_dim = cfg.n_heads_kv * cfg.head_dim;
    let layer_stride = kv_cache_len / cfg.num_layers;
    let capacity_slots = layer_stride / (2 * kv_dim);
    if capacity_slots == 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "kv_cache".to_owned(),
            offset: 0,
            byte_len: cfg.num_layers * 2 * kv_dim * size_of::<f32>(),
            buffer_len: kv_cache_len * size_of::<f32>(),
            reason: format!(
                "kv_cache length {kv_cache_len} is too small for {} layers * 2 * kv_dim({kv_dim})",
                cfg.num_layers
            ),
        });
    }
    let has_global_layer = (0..cfg.num_layers).any(|l| cfg.is_global_layer(l));
    if has_global_layer && end_pos > capacity_slots {
        return Err(K3Error::ContextLengthExceeded {
            requested_tokens: end_pos,
            max_context_tokens: capacity_slots,
        });
    }
    Ok((layer_stride, capacity_slots))
}

struct AttentionViews<'a> {
    q: TensorView<'a>,
    k: TensorView<'a>,
    v: TensorView<'a>,
    o: TensorView<'a>,
}

fn load_attention_views(
    cfg: GemmaLayerConfig,
    layer: usize,
    weights: &dyn ModelWeights,
) -> Result<AttentionViews<'_>, K3Error> {
    let q_dim = cfg.n_heads_q * cfg.head_dim;
    let kv_dim = cfg.n_heads_kv * cfg.head_dim;
    Ok(AttentionViews {
        q: require_layer_2d_matvec(
            weights,
            cfg.architecture,
            layer,
            "attn_q.weight",
            &["self_attn.q_proj.weight"],
            cfg.hidden_dim,
            q_dim,
        )?,
        k: require_layer_2d_matvec(
            weights,
            cfg.architecture,
            layer,
            "attn_k.weight",
            &["self_attn.k_proj.weight"],
            cfg.hidden_dim,
            kv_dim,
        )?,
        v: require_layer_2d_matvec(
            weights,
            cfg.architecture,
            layer,
            "attn_v.weight",
            &["self_attn.v_proj.weight"],
            cfg.hidden_dim,
            kv_dim,
        )?,
        o: require_layer_2d_matvec(
            weights,
            cfg.architecture,
            layer,
            "attn_output.weight",
            &["self_attn.o_proj.weight"],
            q_dim,
            cfg.hidden_dim,
        )?,
    })
}

fn apply_optional_qk_norm(
    cfg: GemmaLayerConfig,
    layer: usize,
    q: &mut [f32],
    k: &mut [f32],
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    if !cfg.use_qk_norm {
        return Ok(());
    }
    let qk_norm_q = require_layer_1d_f32(
        weights,
        cfg.architecture,
        layer,
        "attn_q_norm.weight",
        &["self_attn.q_norm.weight"],
        cfg.head_dim,
    )?;
    let qk_norm_k = require_layer_1d_f32(
        weights,
        cfg.architecture,
        layer,
        "attn_k_norm.weight",
        &["self_attn.k_norm.weight"],
        cfg.head_dim,
    )?;
    qk_norm_inplace(q, cfg.head_dim, &qk_norm_q, cfg.rms_eps)?;
    qk_norm_inplace(k, cfg.head_dim, &qk_norm_k, cfg.rms_eps)
}

fn compute_layer_kv_attention(
    cfg: GemmaLayerConfig,
    layer: usize,
    step: StepPosition,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    kv_cache: &mut [f32],
) -> Result<Vec<f32>, K3Error> {
    let q_dim = cfg.n_heads_q * cfg.head_dim;
    let kv_dim = cfg.n_heads_kv * cfg.head_dim;
    #[expect(
        clippy::cast_precision_loss,
        reason = "head_dim fits comfortably in f32"
    )]
    let scale = 1.0_f32 / (cfg.head_dim as f32).sqrt();
    let mut attn_out = vec![0.0_f32; q_dim];

    let (layer_stride, capacity_slots) = validate_kv_capacity(cfg, kv_cache.len(), step.pos + 1)?;
    let layer_start = layer * layer_stride;
    if cfg.is_global_layer(layer) {
        let ring_elems = capacity_slots * kv_dim;
        let active_len = step.pos + 1;
        let (k_ring, v_ring) =
            kv_cache[layer_start..layer_start + 2 * ring_elems].split_at_mut(ring_elems);
        k_ring[step.pos * kv_dim..(step.pos + 1) * kv_dim].copy_from_slice(k);
        v_ring[step.pos * kv_dim..(step.pos + 1) * kv_dim].copy_from_slice(v);
        gqa_causal_attention_f32(
            &mut attn_out,
            q,
            &k_ring[..active_len * kv_dim],
            &v_ring[..active_len * kv_dim],
            cfg.n_heads_q,
            cfg.n_heads_kv,
            cfg.head_dim,
            active_len,
            scale,
        )?;
    } else {
        let eff_win = cfg.sliding_window.min(capacity_slots);
        let layer_kv = &mut kv_cache[layer_start..layer_start + 2 * eff_win * kv_dim];
        let (k_slice, v_slice, active_len) =
            write_and_slice_sliding_kv(layer_kv, step.pos, eff_win, kv_dim, k, v)?;
        gqa_causal_attention_f32(
            &mut attn_out,
            q,
            k_slice,
            v_slice,
            cfg.n_heads_q,
            cfg.n_heads_kv,
            cfg.head_dim,
            active_len,
            scale,
        )?;
    }

    if let Some(cap) = cfg.attn_logit_softcap {
        logit_softcap_tanh_inplace(&mut attn_out, cap);
    }
    Ok(attn_out)
}

fn run_attention_sublayer(
    cfg: GemmaLayerConfig,
    layer: usize,
    step: StepPosition,
    hidden: &mut [f32],
    kv_cache: &mut [f32],
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    let hidden_dim = cfg.hidden_dim;
    let attn_w = require_layer_1d_f32(
        weights,
        cfg.architecture,
        layer,
        "attn_norm.weight",
        &["input_layernorm.weight"],
        hidden_dim,
    )?;
    let mut norm_hidden = vec![0.0_f32; hidden_dim];
    rmsnorm_unit_offset(&mut norm_hidden, hidden, &attn_w, cfg.rms_eps)?;

    let views = load_attention_views(cfg, layer, weights)?;
    let mut q = vec![0.0_f32; cfg.n_heads_q * cfg.head_dim];
    let mut k = vec![0.0_f32; cfg.n_heads_kv * cfg.head_dim];
    let mut v = vec![0.0_f32; cfg.n_heads_kv * cfg.head_dim];
    matvec_any(&mut q, &views.q, &norm_hidden)?;
    matvec_any(&mut k, &views.k, &norm_hidden)?;
    matvec_any(&mut v, &views.v, &norm_hidden)?;

    apply_optional_qk_norm(cfg, layer, &mut q, &mut k, weights)?;

    let rope_theta = if cfg.is_global_layer(layer) {
        cfg.rope_freq_base_global
    } else {
        cfg.rope_freq_base_local
    };
    let rope_seq = step.seq_len.max(step.pos + 1);
    apply_rope_inplace(&mut q, cfg.head_dim, step.pos, rope_seq, rope_theta, None)?;
    apply_rope_inplace(&mut k, cfg.head_dim, step.pos, rope_seq, rope_theta, None)?;

    let attn_out = compute_layer_kv_attention(cfg, layer, step, &q, &k, &v, kv_cache)?;

    let mut proj_attn = vec![0.0_f32; hidden_dim];
    matvec_any(&mut proj_attn, &views.o, &attn_out)?;

    let residual_attn = if let Some(post_view) = try_lookup_layer_tensor(
        weights,
        layer,
        "post_attention_norm.weight",
        &["post_attention_layernorm.weight"],
    )? {
        let post_w = decode_1d_f32_strict(&post_view, cfg.architecture, hidden_dim)?;
        let mut normed = vec![0.0_f32; hidden_dim];
        rmsnorm_unit_offset(&mut normed, &proj_attn, &post_w, cfg.rms_eps)?;
        normed
    } else {
        proj_attn
    };

    for (h, &r) in hidden.iter_mut().zip(residual_attn.iter()) {
        *h += r;
    }
    Ok(())
}

fn load_layer_sparse_moe_experts(
    cfg: GemmaLayerConfig,
    layer: usize,
    n_exp: usize,
    weights: &dyn ModelWeights,
) -> Result<Vec<ExpertWeights<'_>>, K3Error> {
    let has_split_experts = try_lookup_layer_tensor(
        weights,
        layer,
        "ffn_gate_exps.0.weight",
        &["mlp.experts.0.gate_proj.weight"],
    )?
    .is_some();

    let mut experts = Vec::with_capacity(n_exp);
    if has_split_experts {
        for e in 0..n_exp {
            let gguf_g = format!("ffn_gate_exps.{e}.weight");
            let gguf_u = format!("ffn_up_exps.{e}.weight");
            let gguf_d = format!("ffn_down_exps.{e}.weight");
            let hf_g = format!("mlp.experts.{e}.gate_proj.weight");
            let hf_u = format!("mlp.experts.{e}.up_proj.weight");
            let hf_d = format!("mlp.experts.{e}.down_proj.weight");
            let gate = lookup_layer_tensor(weights, layer, &gguf_g, &[&hf_g])?;
            let up = lookup_layer_tensor(weights, layer, &gguf_u, &[&hf_u])?;
            let down = lookup_layer_tensor(weights, layer, &gguf_d, &[&hf_d])?;
            experts.push(ExpertWeights { gate, up, down });
        }
        return Ok(experts);
    }

    let stacked_gate = lookup_layer_tensor(
        weights,
        layer,
        "ffn_gate_exps.weight",
        &["mlp.experts.gate_proj.weight"],
    )?;
    let stacked_up = lookup_layer_tensor(
        weights,
        layer,
        "ffn_up_exps.weight",
        &["mlp.experts.up_proj.weight"],
    )?;
    let stacked_down = lookup_layer_tensor(
        weights,
        layer,
        "ffn_down_exps.weight",
        &["mlp.experts.down_proj.weight"],
    )?;
    for e in 0..n_exp {
        let gate = stacked_gate.slice_expert(e, n_exp)?;
        let up = stacked_up.slice_expert(e, n_exp)?;
        let down = stacked_down.slice_expert(e, n_exp)?;
        if !is_supported_gemma_quant(gate.quant()) {
            return Err(K3Error::unsupported_tensor_quant(
                cfg.architecture,
                gate.name(),
                gate.quant(),
            ));
        }
        experts.push(ExpertWeights { gate, up, down });
    }
    Ok(experts)
}

struct DenseFfnViews<'a> {
    gate: TensorView<'a>,
    up: TensorView<'a>,
    down: TensorView<'a>,
}

fn load_dense_ffn_views(
    cfg: GemmaLayerConfig,
    layer: usize,
    weights: &dyn ModelWeights,
) -> Result<DenseFfnViews<'_>, K3Error> {
    Ok(DenseFfnViews {
        gate: require_layer_2d_matvec(
            weights,
            cfg.architecture,
            layer,
            "ffn_gate.weight",
            &["mlp.gate_proj.weight"],
            cfg.hidden_dim,
            cfg.ffn_dim,
        )?,
        up: require_layer_2d_matvec(
            weights,
            cfg.architecture,
            layer,
            "ffn_up.weight",
            &["mlp.up_proj.weight"],
            cfg.hidden_dim,
            cfg.ffn_dim,
        )?,
        down: require_layer_2d_matvec(
            weights,
            cfg.architecture,
            layer,
            "ffn_down.weight",
            &["mlp.down_proj.weight"],
            cfg.ffn_dim,
            cfg.hidden_dim,
        )?,
    })
}

struct PleLayerWeights<'a> {
    embd: TensorView<'a>,
    inp_gate: TensorView<'a>,
    proj: TensorView<'a>,
    post_norm: Vec<f32>,
}

fn load_ple_layer_weights(
    cfg: GemmaLayerConfig,
    layer: usize,
    ple_dim: usize,
    weights: &dyn ModelWeights,
) -> Result<PleLayerWeights<'_>, K3Error> {
    Ok(PleLayerWeights {
        embd: lookup_tensor_aliases(weights, PLE_EMBD_ALIASES)?,
        inp_gate: require_layer_2d_matvec(
            weights,
            cfg.architecture,
            layer,
            "per_layer_inp_gate.weight",
            &["per_layer_input_gate.weight"],
            cfg.hidden_dim,
            ple_dim,
        )?,
        proj: require_layer_2d_matvec(
            weights,
            cfg.architecture,
            layer,
            "per_layer_proj.weight",
            &["per_layer_projection.weight"],
            ple_dim,
            cfg.hidden_dim,
        )?,
        post_norm: require_layer_1d_f32(
            weights,
            cfg.architecture,
            layer,
            "per_layer_post_norm.weight",
            &["post_per_layer_input_norm.weight"],
            cfg.hidden_dim,
        )?,
    })
}

fn run_ffn_and_ple_sublayers(
    cfg: GemmaLayerConfig,
    layer: usize,
    token_id: Option<u32>,
    hidden: &mut [f32],
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    let hidden_dim = cfg.hidden_dim;
    let ffn_norm_w = require_layer_1d_f32(
        weights,
        cfg.architecture,
        layer,
        "ffn_norm.weight",
        &["pre_feedforward_layernorm.weight"],
        hidden_dim,
    )?;
    let mut norm_ffn_in = vec![0.0_f32; hidden_dim];
    rmsnorm_unit_offset(&mut norm_ffn_in, hidden, &ffn_norm_w, cfg.rms_eps)?;

    let ffn_out = if let (Some(n_exp), Some(top_k)) = (cfg.num_experts, cfg.num_experts_per_tok) {
        let router_view = lookup_layer_tensor(
            weights,
            layer,
            "ffn_gate_inp.weight",
            &["mlp.gate.weight", "mlp.router.weight", "router.weight"],
        )?;
        let router_view =
            validate_2d_matvec_view(router_view, cfg.architecture, hidden_dim, n_exp)?;
        let experts = load_layer_sparse_moe_experts(cfg, layer, n_exp, weights)?;
        let mut moe_out = vec![0.0_f32; hidden_dim];
        dispatch_sparse_moe_layer(&mut moe_out, &norm_ffn_in, &router_view, &experts, top_k)?;
        moe_out
    } else {
        let views = load_dense_ffn_views(cfg, layer, weights)?;
        let mut gate = vec![0.0_f32; cfg.ffn_dim];
        let mut up = vec![0.0_f32; cfg.ffn_dim];
        matvec_any(&mut gate, &views.gate, &norm_ffn_in)?;
        matvec_any(&mut up, &views.up, &norm_ffn_in)?;
        let mut geglu = vec![0.0_f32; cfg.ffn_dim];
        geglu_f32(&mut geglu, &gate, &up)?;
        let mut down = vec![0.0_f32; hidden_dim];
        matvec_any(&mut down, &views.down, &geglu)?;
        down
    };

    let ffn_residual = if let Some(post_view) = try_lookup_layer_tensor(
        weights,
        layer,
        "post_ffw_norm.weight",
        &["post_feedforward_layernorm.weight"],
    )? {
        let w = decode_1d_f32_strict(&post_view, cfg.architecture, hidden_dim)?;
        let mut normed = vec![0.0_f32; hidden_dim];
        rmsnorm_unit_offset(&mut normed, &ffn_out, &w, cfg.rms_eps)?;
        normed
    } else {
        ffn_out
    };

    for (h, &r) in hidden.iter_mut().zip(ffn_residual.iter()) {
        *h += r;
    }

    if let Some(ple_dim) = cfg.ple_dim {
        let ple = load_ple_layer_weights(cfg, layer, ple_dim, weights)?;
        if let Some(tid) = token_id {
            apply_gemma4_ple_residual(
                hidden,
                tid,
                layer,
                &ple.embd,
                &ple.inp_gate,
                &ple.proj,
                &ple.post_norm,
                cfg.rms_eps,
            )?;
        }
    }

    Ok(())
}

fn project_lm_head(
    cfg: GemmaLayerConfig,
    normed_final: &[f32],
    logits: &mut [f32],
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    let hidden_dim = normed_final.len();
    let head = if let Some(explicit_head) = try_lookup_tensor_aliases(weights, OUTPUT_HEAD_ALIASES)?
    {
        explicit_head
    } else {
        lookup_tensor_aliases(weights, TOKEN_EMBD_ALIASES)?
    };
    let head = validate_2d_matvec_view(head, cfg.architecture, hidden_dim, logits.len())?;
    matvec_any(logits, &head, normed_final)?;
    if let Some(cap) = cfg.final_logit_softcap {
        logit_softcap_tanh_inplace(logits, cap);
    }
    Ok(())
}

fn execute_gemma_token(
    cfg: GemmaLayerConfig,
    input: TokenInput<'_>,
    step: StepPosition,
    kv_cache: &mut [f32],
    logits: Option<&mut [f32]>,
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    let (token_id, mut hidden) = match input {
        TokenInput::TokenId(tid) => {
            let h = embed_token_hidden(cfg.architecture, tid, cfg.hidden_dim, weights)?;
            (Some(tid), h)
        }
        TokenInput::SoftEmbed(embed) => {
            if embed.len() != cfg.hidden_dim {
                return Err(missing_tensor_err(
                    "soft_prompt_embeds",
                    format!(
                        "soft_prompt embed_dim ({}) does not match hidden_dim ({})",
                        embed.len(),
                        cfg.hidden_dim
                    ),
                ));
            }
            (None, embed.to_vec())
        }
    };

    for layer in 0..cfg.num_layers {
        run_attention_sublayer(cfg, layer, step, &mut hidden, kv_cache, weights)?;
        run_ffn_and_ple_sublayers(cfg, layer, token_id, &mut hidden, weights)?;
    }

    if let Some(logits_out) = logits {
        let out_norm_w = require_1d_f32(
            weights,
            cfg.architecture,
            OUTPUT_NORM_ALIASES,
            cfg.hidden_dim,
        )?;
        let mut normed_final = vec![0.0_f32; cfg.hidden_dim];
        rmsnorm_unit_offset(&mut normed_final, &hidden, &out_norm_w, cfg.rms_eps)?;
        if !logits_out.is_empty() {
            project_lm_head(cfg, &normed_final, logits_out, weights)?;
        }
    }

    Ok(())
}

fn validate_soft_prompt<'a>(
    cfg: GemmaLayerConfig,
    ctx: &ExecutionContext<'a>,
) -> Result<Option<(&'a [f32], usize)>, K3Error> {
    match (ctx.soft_prompt_embeds(), ctx.soft_prompt_dims()) {
        (None, None) => Ok(None),
        (Some(embeds), Some((num_soft, embed_dim))) => {
            if num_soft == 0 || embed_dim != cfg.hidden_dim {
                return Err(missing_tensor_err(
                    "soft_prompt_embeds",
                    format!(
                        "soft_prompt dims ({num_soft}, {embed_dim}) incompatible with hidden_dim={}",
                        cfg.hidden_dim
                    ),
                ));
            }
            let expected_len = num_soft.checked_mul(embed_dim).ok_or_else(|| {
                missing_tensor_err("soft_prompt_embeds", "soft_prompt dims overflowed usize")
            })?;
            if embeds.len() != expected_len {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: "soft_prompt_embeds".to_owned(),
                    offset: 0,
                    byte_len: expected_len.saturating_mul(size_of::<f32>()),
                    buffer_len: size_of_val(embeds),
                    reason: format!(
                        "soft_prompt_embeds length {} != {num_soft} * {embed_dim}",
                        embeds.len()
                    ),
                });
            }
            for (idx, &val) in embeds.iter().enumerate() {
                if !val.is_finite() {
                    return Err(K3Error::TensorBoundsError {
                        tensor_name: "soft_prompt_embeds".to_owned(),
                        offset: idx.saturating_mul(size_of::<f32>()),
                        byte_len: size_of::<f32>(),
                        buffer_len: size_of_val(embeds),
                        reason: format!("non-finite soft_prompt value ({val}) at index {idx}"),
                    });
                }
            }
            Ok(Some((embeds, num_soft)))
        }
        _ => Err(missing_tensor_err(
            "soft_prompt_embeds",
            "soft_prompt_embeds and soft_prompt_dims must both be present or both absent",
        )),
    }
}

fn execute_gemma_prefill(
    cfg: GemmaLayerConfig,
    ctx: &mut ExecutionContext<'_>,
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    let (discrete_tokens, soft_prompt) = if let Some(forced) = ctx.jump_forward_tokens() {
        (forced.to_vec(), None)
    } else {
        let soft = validate_soft_prompt(cfg, ctx)?;
        (ctx.prompt_tokens().to_vec(), soft)
    };

    let num_soft = soft_prompt.map_or(0, |(_, n)| n);
    let total_tokens =
        num_soft
            .checked_add(discrete_tokens.len())
            .ok_or(K3Error::ContextLengthExceeded {
                requested_tokens: usize::MAX,
                max_context_tokens: ctx.n_ctx(),
            })?;
    if total_tokens == 0 {
        return Err(capability_err(
            cfg.architecture,
            "prefill_step requires non-empty prompt_tokens or soft_prompt_embeds",
        ));
    }

    let start_pos = ctx.position();
    let end_pos = start_pos
        .checked_add(total_tokens)
        .ok_or(K3Error::ContextLengthExceeded {
            requested_tokens: usize::MAX,
            max_context_tokens: ctx.n_ctx(),
        })?;
    if end_pos > ctx.n_ctx() {
        return Err(K3Error::ContextLengthExceeded {
            requested_tokens: end_pos,
            max_context_tokens: ctx.n_ctx(),
        });
    }
    let _ = validate_kv_capacity(cfg, ctx.kv_cache().len(), end_pos)?;

    let compute_logits = ctx.compute_logits();
    let logits_len = ctx.logits().len();
    let mut final_logits = if compute_logits {
        vec![0.0_f32; logits_len]
    } else {
        Vec::new()
    };

    for step_idx in 0..total_tokens {
        let input = if let Some((embeds, n_soft)) = soft_prompt
            && step_idx < n_soft
        {
            let start = step_idx * cfg.hidden_dim;
            TokenInput::SoftEmbed(&embeds[start..start + cfg.hidden_dim])
        } else {
            TokenInput::TokenId(discrete_tokens[step_idx - num_soft])
        };

        let is_last = step_idx + 1 == total_tokens;
        let step = StepPosition {
            pos: start_pos + step_idx,
            seq_len: end_pos,
        };
        if is_last {
            execute_gemma_token(
                cfg,
                input,
                step,
                ctx.kv_cache_mut(),
                Some(&mut final_logits),
                weights,
            )?;
        } else {
            execute_gemma_token(cfg, input, step, ctx.kv_cache_mut(), None, weights)?;
        }
    }

    if compute_logits && !final_logits.is_empty() {
        ctx.logits_mut().copy_from_slice(&final_logits);
    }
    ctx.advance_position(total_tokens)?;
    Ok(())
}

fn execute_gemma_decode(
    cfg: GemmaLayerConfig,
    ctx: &mut ExecutionContext<'_>,
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    let tok = ctx.decode_token().ok_or_else(|| {
        capability_err(
            cfg.architecture,
            "decode_step called without a staged decode token",
        )
    })?;
    let pos = ctx.position();
    let end_pos = pos.checked_add(1).ok_or(K3Error::ContextLengthExceeded {
        requested_tokens: usize::MAX,
        max_context_tokens: ctx.n_ctx(),
    })?;
    if end_pos > ctx.n_ctx() {
        return Err(K3Error::ContextLengthExceeded {
            requested_tokens: end_pos,
            max_context_tokens: ctx.n_ctx(),
        });
    }
    let _ = validate_kv_capacity(cfg, ctx.kv_cache().len(), end_pos)?;

    let mut step_logits = vec![0.0_f32; ctx.logits().len()];
    let step = StepPosition {
        pos,
        seq_len: end_pos,
    };
    execute_gemma_token(
        cfg,
        TokenInput::TokenId(tok),
        step,
        ctx.kv_cache_mut(),
        Some(&mut step_logits),
        weights,
    )?;
    if ctx.compute_logits() && !step_logits.is_empty() {
        ctx.logits_mut().copy_from_slice(&step_logits);
    }
    ctx.advance_position(1)?;
    Ok(())
}

fn validate_gemma_layer_weights(
    cfg: GemmaLayerConfig,
    layer: usize,
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    let _ = require_layer_1d_f32(
        weights,
        cfg.architecture,
        layer,
        "attn_norm.weight",
        &["input_layernorm.weight"],
        cfg.hidden_dim,
    )?;
    let _ = load_attention_views(cfg, layer, weights)?;
    if cfg.use_qk_norm {
        let _ = require_layer_1d_f32(
            weights,
            cfg.architecture,
            layer,
            "attn_q_norm.weight",
            &["self_attn.q_norm.weight"],
            cfg.head_dim,
        )?;
        let _ = require_layer_1d_f32(
            weights,
            cfg.architecture,
            layer,
            "attn_k_norm.weight",
            &["self_attn.k_norm.weight"],
            cfg.head_dim,
        )?;
    }
    let _ = require_layer_1d_f32(
        weights,
        cfg.architecture,
        layer,
        "ffn_norm.weight",
        &["pre_feedforward_layernorm.weight"],
        cfg.hidden_dim,
    )?;
    if let Some(n_exp) = cfg.num_experts {
        let router = lookup_layer_tensor(
            weights,
            layer,
            "ffn_gate_inp.weight",
            &["mlp.gate.weight", "mlp.router.weight", "router.weight"],
        )?;
        let _ = validate_2d_matvec_view(router, cfg.architecture, cfg.hidden_dim, n_exp)?;
        let _ = load_layer_sparse_moe_experts(cfg, layer, n_exp, weights)?;
    } else {
        let _ = load_dense_ffn_views(cfg, layer, weights)?;
    }
    if let Some(ple_dim) = cfg.ple_dim {
        let _ = load_ple_layer_weights(cfg, layer, ple_dim, weights)?;
    }
    Ok(())
}

fn validate_gemma_model_weights(
    recipe: &dyn ArchRecipe,
    cfg: GemmaLayerConfig,
    vocab_size: usize,
    metadata: &ModelMetadata,
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    for desc in metadata.tensors() {
        if !recipe.supports_quant(desc.quant()) {
            return Err(K3Error::unsupported_tensor_quant(
                metadata.architecture(),
                desc.name(),
                desc.quant(),
            ));
        }
    }
    if weights.tensor_count() == 0 {
        return Ok(());
    }

    let embd = lookup_tensor_aliases(weights, TOKEN_EMBD_ALIASES)?;
    if !recipe.supports_quant(embd.quant()) {
        return Err(K3Error::unsupported_tensor_quant(
            cfg.architecture,
            embd.name(),
            embd.quant(),
        ));
    }
    let _ = validate_2d_matvec_view(embd, cfg.architecture, cfg.hidden_dim, vocab_size.max(1))?;
    let _ = require_1d_f32(
        weights,
        cfg.architecture,
        OUTPUT_NORM_ALIASES,
        cfg.hidden_dim,
    )?;
    if let Some(head) = try_lookup_tensor_aliases(weights, OUTPUT_HEAD_ALIASES)? {
        let _ = validate_2d_matvec_view(head, cfg.architecture, cfg.hidden_dim, vocab_size.max(1))?;
    }

    for layer in 0..cfg.num_layers {
        validate_gemma_layer_weights(cfg, layer, weights)?;
    }
    Ok(())
}

impl ArchRecipe for GemmaRecipe {
    fn architecture_id(&self) -> &'static str {
        self.architecture
    }

    fn target_layouts(&self) -> &'static [&'static str] {
        &["native", "ime_tile_8x16"]
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        is_supported_gemma_quant(quant)
    }

    fn validate_model(
        &self,
        metadata: &ModelMetadata,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let cfg = self.layer_config()?;
        validate_gemma_model_weights(self, cfg, self.vocab_size, metadata, weights)
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let cfg = self.layer_config()?;
        execute_gemma_prefill(cfg, ctx, weights)
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let cfg = self.layer_config()?;
        execute_gemma_decode(cfg, ctx, weights)
    }
}

impl ArchRecipe for Gemma4Recipe {
    fn architecture_id(&self) -> &'static str {
        self.architecture
    }

    fn target_layouts(&self) -> &'static [&'static str] {
        &["native", "ime_tile_8x16"]
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        is_supported_gemma_quant(quant)
    }

    fn validate_model(
        &self,
        metadata: &ModelMetadata,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let cfg = self.layer_config()?;
        validate_gemma_model_weights(self, cfg, self.vocab_size, metadata, weights)
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let cfg = self.layer_config()?;
        execute_gemma_prefill(cfg, ctx, weights)
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let cfg = self.layer_config()?;
        execute_gemma_decode(cfg, ctx, weights)
    }
}
