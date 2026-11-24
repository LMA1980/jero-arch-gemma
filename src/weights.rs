//! Strict zero-copy tensor lookup, geometry validation, and quantized matrix-vector
//! dispatch across `GGUF` and `SafeTensors` naming conventions (ADR-045).

use jero_core::{K3Error, ModelWeights, QuantType, TensorView};
use jero_kernels_rvv::{
    BaselineKernel, VectorKernel, dequantize_token_row, dot_product_f32_scalar, vec_dot_bf16,
    vec_dot_f16, vec_dot_i2_s, vec_dot_iq4_nl, vec_dot_q2_k, vec_dot_q3_k, vec_dot_q4_0,
    vec_dot_q4_1, vec_dot_q5_0, vec_dot_q5_1, vec_dot_q5_k, vec_dot_q6_k, vec_dot_q8_0,
};

pub(crate) const TOKEN_EMBD_ALIASES: &[&str] = &[
    "token_embd.weight",
    "model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
    "model.language_model.embed_tokens.weight",
    "embed_tokens.weight",
];

pub(crate) const OUTPUT_NORM_ALIASES: &[&str] = &[
    "output_norm.weight",
    "model.norm.weight",
    "language_model.model.norm.weight",
    "model.language_model.norm.weight",
    "norm.weight",
];

pub(crate) const OUTPUT_HEAD_ALIASES: &[&str] = &[
    "output.weight",
    "lm_head.weight",
    "language_model.lm_head.weight",
];

pub(crate) const PLE_EMBD_ALIASES: &[&str] = &[
    "per_layer_token_embd.weight",
    "model.per_layer_embed_tokens.weight",
    "model.embed_tokens_per_layer.weight",
    "per_layer_embed_tokens.weight",
];

const HF_LAYER_PREFIXES: &[&str] = &[
    "model.layers.",
    "language_model.model.layers.",
    "model.language_model.layers.",
    "layers.",
];

pub(crate) fn tensor_bounds_err(view: &TensorView<'_>, reason: impl Into<String>) -> K3Error {
    K3Error::TensorBoundsError {
        tensor_name: view.name().to_owned(),
        offset: view.offset(),
        byte_len: view.byte_len(),
        buffer_len: view.as_bytes().len(),
        reason: reason.into(),
    }
}

pub(crate) fn missing_tensor_err(name: &str, reason: impl Into<String>) -> K3Error {
    K3Error::TensorBoundsError {
        tensor_name: name.to_owned(),
        offset: 0,
        byte_len: 0,
        buffer_len: 0,
        reason: reason.into(),
    }
}

pub(crate) fn capability_err(arch: &str, reason: impl Into<String>) -> K3Error {
    K3Error::CapabilityMismatch {
        architecture: arch.to_owned(),
        reason: reason.into(),
    }
}

pub(crate) const fn is_supported_gemma_quant(quant: QuantType) -> bool {
    matches!(
        quant,
        QuantType::F32
            | QuantType::F16
            | QuantType::BF16
            | QuantType::Q4_0
            | QuantType::Q4_1
            | QuantType::Q5_0
            | QuantType::Q5_1
            | QuantType::Q8_0
            | QuantType::Q2_K
            | QuantType::Q3_K
            | QuantType::Q4_K_M
            | QuantType::Q5_K
            | QuantType::Q6_K
            | QuantType::IQ4_NL
    )
}

fn matvec_any_slice(
    out: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    match weights.quant() {
        QuantType::F32 => matvec_f32(out, weights, input),
        QuantType::F16 => vec_dot_f16(out, weights, input),
        QuantType::BF16 => vec_dot_bf16(out, weights, input),
        QuantType::Q4_0 => vec_dot_q4_0(out, weights, input),
        QuantType::Q4_1 => vec_dot_q4_1(out, weights, input),
        QuantType::Q5_0 => vec_dot_q5_0(out, weights, input),
        QuantType::Q5_1 => vec_dot_q5_1(out, weights, input),
        QuantType::Q8_0 => vec_dot_q8_0(out, weights, input),
        QuantType::Q2_K => vec_dot_q2_k(out, weights, input),
        QuantType::Q3_K => vec_dot_q3_k(out, weights, input),
        QuantType::Q4_K_M => BaselineKernel::new().vec_dot_q4_k_m(out, weights, input),
        QuantType::Q5_K => vec_dot_q5_k(out, weights, input),
        QuantType::Q6_K => vec_dot_q6_k(out, weights, input),
        QuantType::IQ4_NL => vec_dot_iq4_nl(out, weights, input),
        QuantType::I2_S => vec_dot_i2_s(out, weights, input),
        other => Err(K3Error::CapabilityMismatch {
            architecture: "gemma".to_owned(),
            reason: format!(
                "unsupported weight quantization {other:?} for tensor '{}'",
                weights.name()
            ),
        }),
    }
}

fn validate_matvec_shape(
    weights: &TensorView<'_>,
    rows: usize,
    cols: usize,
) -> Result<(), K3Error> {
    let shape = weights.shape();
    if shape.is_empty() {
        return Err(tensor_bounds_err(
            weights,
            "matvec weight tensor has empty shape",
        ));
    }
    if shape.len() == 2 {
        let (d0, d1) = (shape[0], shape[1]);
        if !((d0 == cols && d1 == rows) || (d0 == rows && d1 == cols)) {
            return Err(tensor_bounds_err(
                weights,
                format!("matrix shape [{d0}, {d1}] does not match out({rows}) x in({cols})"),
            ));
        }
    } else if shape.len() != 1 {
        return Err(tensor_bounds_err(
            weights,
            format!("expected 1D or 2D matrix shape, got {shape:?}"),
        ));
    }

    let expected_elems = rows
        .checked_mul(cols)
        .ok_or_else(|| tensor_bounds_err(weights, "matvec dimension overflowed usize"))?;
    if weights.num_elements() != expected_elems {
        return Err(tensor_bounds_err(
            weights,
            format!(
                "tensor elements {} != out({rows}) * in({cols})",
                weights.num_elements()
            ),
        ));
    }
    Ok(())
}

/// Dispatches matrix-vector multiplication across all quantization formats supported by Gemma,
/// fanning out independent output row slices across active `rayon` worker threads when
/// `rayon::current_num_threads() > 1`.
///
/// Supports both GGUF `[in_dim, out_dim]` and `SafeTensors` row-major `[out_dim, in_dim]` layouts.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `out` or `input` is empty, tensor dimensions or
/// buffer bounds do not match `out.len() * input.len()`, or `F32` elements are non-finite.
/// Returns [`K3Error::CapabilityMismatch`] if `weights.quant()` is unsupported.
pub fn matvec_any(out: &mut [f32], weights: &TensorView<'_>, input: &[f32]) -> Result<(), K3Error> {
    if out.is_empty() || input.is_empty() {
        return Err(tensor_bounds_err(
            weights,
            "matvec output and input slices cannot be empty",
        ));
    }
    validate_matvec_shape(weights, out.len(), input.len())?;

    #[cfg(not(miri))]
    if out.len() >= 64 && weights.shape().len() == 2 && weights.quant() != QuantType::I2_S {
        let cols = input.len();
        let rows = out.len();
        let shape = weights.shape();
        let is_gguf_layout = shape[0] == cols && shape[1] == rows;
        let is_row_major_layout = shape[0] == rows && shape[1] == cols;
        if is_gguf_layout || is_row_major_layout {
            let n_threads = rayon::current_num_threads();
            if n_threads > 1 {
                return matvec_any_parallel(out, weights, input, n_threads, is_gguf_layout);
            }
        }
    }
    matvec_any_slice(out, weights, input)
}

#[cfg(not(miri))]
fn matvec_any_parallel(
    out: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
    n_threads: usize,
    is_gguf_layout: bool,
) -> Result<(), K3Error> {
    use rayon::prelude::*;

    let rows = out.len();
    let cols = input.len();
    let chunk_rows = rows.div_ceil(n_threads).clamp(16, 2048);

    if is_gguf_layout {
        return out.par_chunks_mut(chunk_rows).enumerate().try_for_each(
            |(chunk_idx, out_chunk)| {
                let row_start = chunk_idx * chunk_rows;
                let chunk_view = weights.slice_rows(row_start, out_chunk.len())?;
                matvec_any_slice(out_chunk, &chunk_view, input)
            },
        );
    }

    let block_elems = weights.quant().block_elements().max(1);
    if cols % block_elems != 0 {
        return Err(tensor_bounds_err(
            weights,
            format!(
                "input dimension {cols} is not divisible by {} block size {block_elems}",
                weights.quant()
            ),
        ));
    }
    let row_bytes = (cols / block_elems)
        .checked_mul(weights.quant().block_bytes())
        .ok_or_else(|| tensor_bounds_err(weights, "row byte stride overflowed usize"))?;
    let expected_bytes = rows
        .checked_mul(row_bytes)
        .ok_or_else(|| tensor_bounds_err(weights, "matrix byte length overflowed usize"))?;
    let raw = weights.as_bytes();
    if raw.len() < expected_bytes {
        return Err(tensor_bounds_err(
            weights,
            format!(
                "buffer length {} is smaller than required {expected_bytes}",
                raw.len()
            ),
        ));
    }

    out.par_chunks_mut(chunk_rows)
        .enumerate()
        .try_for_each(|(chunk_idx, out_chunk)| {
            let row_start = chunk_idx * chunk_rows;
            let byte_start = row_start * row_bytes;
            let byte_end = byte_start + out_chunk.len() * row_bytes;
            let chunk_dims = [out_chunk.len(), cols];
            let chunk_view = TensorView::from_bytes(
                weights.name(),
                weights.quant(),
                &chunk_dims,
                0,
                &raw[byte_start..byte_end],
            )?;
            matvec_any_slice(out_chunk, &chunk_view, input)
        })
}

fn matvec_f32(out: &mut [f32], weights: &TensorView<'_>, input: &[f32]) -> Result<(), K3Error> {
    let rows = out.len();
    let cols = input.len();
    validate_matvec_shape(weights, rows, cols)?;
    let expected_elems = rows * cols;
    let raw = weights.as_bytes();
    let required_bytes = expected_elems
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| tensor_bounds_err(weights, "F32 byte length overflowed usize"))?;
    if raw.len() < required_bytes {
        return Err(tensor_bounds_err(
            weights,
            format!(
                "F32 buffer length {} < required {required_bytes}",
                raw.len()
            ),
        ));
    }

    let mut row_buf = vec![0.0_f32; cols];
    for (r, out_slot) in out.iter_mut().enumerate() {
        let base = r * cols * size_of::<f32>();
        for (c, slot) in row_buf.iter_mut().enumerate() {
            let off = base + c * size_of::<f32>();
            let val = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
            if !val.is_finite() {
                return Err(tensor_bounds_err(
                    weights,
                    format!("non-finite F32 element ({val}) at row {r}, col {c}"),
                ));
            }
            *slot = val;
        }
        *out_slot = dot_product_f32_scalar(&row_buf, input);
    }
    Ok(())
}

pub(crate) fn lookup_tensor_aliases<'w>(
    weights: &'w dyn ModelWeights,
    aliases: &[&str],
) -> Result<TensorView<'w>, K3Error> {
    for &name in aliases {
        if weights.contains_tensor(name) {
            return weights.tensor(name);
        }
    }
    let primary = aliases.first().copied().unwrap_or("<missing>");
    weights.tensor(primary)
}

pub(crate) fn try_lookup_tensor_aliases<'w>(
    weights: &'w dyn ModelWeights,
    aliases: &[&str],
) -> Result<Option<TensorView<'w>>, K3Error> {
    for &name in aliases {
        if weights.contains_tensor(name) {
            return weights.tensor(name).map(Some);
        }
    }
    Ok(None)
}

fn find_present_layer_name(
    weights: &dyn ModelWeights,
    layer: usize,
    gguf_suffix: &str,
    hf_suffixes: &[&str],
) -> Option<String> {
    let gguf_name = format!("blk.{layer}.{gguf_suffix}");
    if weights.contains_tensor(&gguf_name) {
        return Some(gguf_name);
    }
    for prefix in HF_LAYER_PREFIXES {
        for hf_suffix in hf_suffixes {
            let candidate = format!("{prefix}{layer}.{hf_suffix}");
            if weights.contains_tensor(&candidate) {
                return Some(candidate);
            }
        }
    }
    None
}

pub(crate) fn lookup_layer_tensor<'w>(
    weights: &'w dyn ModelWeights,
    layer: usize,
    gguf_suffix: &str,
    hf_suffixes: &[&str],
) -> Result<TensorView<'w>, K3Error> {
    if let Some(name) = find_present_layer_name(weights, layer, gguf_suffix, hf_suffixes) {
        return weights.tensor(&name);
    }
    weights.tensor(&format!("blk.{layer}.{gguf_suffix}"))
}

pub(crate) fn try_lookup_layer_tensor<'w>(
    weights: &'w dyn ModelWeights,
    layer: usize,
    gguf_suffix: &str,
    hf_suffixes: &[&str],
) -> Result<Option<TensorView<'w>>, K3Error> {
    match find_present_layer_name(weights, layer, gguf_suffix, hf_suffixes) {
        Some(name) => weights.tensor(&name).map(Some),
        None => Ok(None),
    }
}

fn f16_to_f32_strict(bits: u16, view: &TensorView<'_>, idx: usize) -> Result<f32, K3Error> {
    let sign = u32::from((bits >> 15) & 0x0001);
    let exp = u32::from((bits >> 10) & 0x001F);
    let mant = u32::from(bits & 0x03FF);
    if exp == 31 {
        return Err(tensor_bounds_err(
            view,
            format!("non-finite F16 element (0x{bits:04x}) at index {idx}"),
        ));
    }
    if exp == 0 {
        if mant == 0 {
            return Ok(f32::from_bits(sign << 31));
        }
        let mut m = mant;
        let mut e = 127_u32 - 14;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03FF;
        return Ok(f32::from_bits((sign << 31) | (e << 23) | (m << 13)));
    }
    Ok(f32::from_bits(
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13),
    ))
}

fn bf16_to_f32_strict(bits: u16, view: &TensorView<'_>, idx: usize) -> Result<f32, K3Error> {
    let exp = (bits >> 7) & 0x00FF;
    if exp == 0x00FF {
        return Err(tensor_bounds_err(
            view,
            format!("non-finite BF16 element (0x{bits:04x}) at index {idx}"),
        ));
    }
    Ok(f32::from_bits(u32::from(bits) << 16))
}

pub(crate) fn decode_1d_f32_strict(
    view: &TensorView<'_>,
    arch: &str,
    expected_len: usize,
) -> Result<Vec<f32>, K3Error> {
    if expected_len == 0 {
        return Err(tensor_bounds_err(
            view,
            "1D weight tensor expected_len must be > 0",
        ));
    }
    if view.shape() != [expected_len] || view.num_elements() != expected_len {
        return Err(tensor_bounds_err(
            view,
            format!(
                "1D tensor '{}' shape {:?} ({} elems) does not match expected [{expected_len}]",
                view.name(),
                view.shape(),
                view.num_elements()
            ),
        ));
    }

    let raw = view.as_bytes();
    let mut out = vec![0.0_f32; expected_len];
    match view.quant() {
        QuantType::F32 => {
            let required = expected_len * size_of::<f32>();
            if raw.len() < required {
                return Err(tensor_bounds_err(
                    view,
                    format!("F32 tensor byte length {} < required {required}", raw.len()),
                ));
            }
            for (idx, slot) in out.iter_mut().enumerate() {
                let off = idx * size_of::<f32>();
                let v = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
                if !v.is_finite() {
                    return Err(tensor_bounds_err(
                        view,
                        format!("non-finite F32 weight ({v}) at index {idx}"),
                    ));
                }
                *slot = v;
            }
        }
        QuantType::F16 => {
            let required = expected_len * size_of::<u16>();
            if raw.len() < required {
                return Err(tensor_bounds_err(
                    view,
                    format!("F16 tensor byte length {} < required {required}", raw.len()),
                ));
            }
            for (idx, slot) in out.iter_mut().enumerate() {
                let off = idx * size_of::<u16>();
                let bits = u16::from_le_bytes([raw[off], raw[off + 1]]);
                *slot = f16_to_f32_strict(bits, view, idx)?;
            }
        }
        QuantType::BF16 => {
            let required = expected_len * size_of::<u16>();
            if raw.len() < required {
                return Err(tensor_bounds_err(
                    view,
                    format!(
                        "BF16 tensor byte length {} < required {required}",
                        raw.len()
                    ),
                ));
            }
            for (idx, slot) in out.iter_mut().enumerate() {
                let off = idx * size_of::<u16>();
                let bits = u16::from_le_bytes([raw[off], raw[off + 1]]);
                *slot = bf16_to_f32_strict(bits, view, idx)?;
            }
        }
        other => {
            return Err(capability_err(
                arch,
                format!(
                    "1D norm tensor '{}' uses unsupported quantization {other:?}",
                    view.name()
                ),
            ));
        }
    }
    Ok(out)
}

pub(crate) fn require_1d_f32(
    weights: &dyn ModelWeights,
    arch: &str,
    aliases: &[&str],
    expected_len: usize,
) -> Result<Vec<f32>, K3Error> {
    let view = lookup_tensor_aliases(weights, aliases)?;
    decode_1d_f32_strict(&view, arch, expected_len)
}

pub(crate) fn require_layer_1d_f32(
    weights: &dyn ModelWeights,
    arch: &str,
    layer: usize,
    gguf_suffix: &str,
    hf_suffixes: &[&str],
    expected_len: usize,
) -> Result<Vec<f32>, K3Error> {
    let view = lookup_layer_tensor(weights, layer, gguf_suffix, hf_suffixes)?;
    decode_1d_f32_strict(&view, arch, expected_len)
}

pub(crate) fn validate_2d_matvec_view<'w>(
    view: TensorView<'w>,
    arch: &str,
    in_dim: usize,
    out_dim: usize,
) -> Result<TensorView<'w>, K3Error> {
    if !is_supported_gemma_quant(view.quant()) {
        return Err(K3Error::unsupported_tensor_quant(
            arch,
            view.name(),
            view.quant(),
        ));
    }
    if in_dim == 0 || out_dim == 0 {
        return Err(tensor_bounds_err(
            &view,
            format!("2D projection dimensions ({in_dim} x {out_dim}) must be non-zero"),
        ));
    }
    let shape = view.shape();
    if shape.len() != 2
        || !((shape[0] == in_dim && shape[1] == out_dim)
            || (shape[0] == out_dim && shape[1] == in_dim))
    {
        return Err(tensor_bounds_err(
            &view,
            format!(
                "2D tensor '{}' shape {shape:?} does not match expected [{in_dim}, {out_dim}] or [{out_dim}, {in_dim}]",
                view.name()
            ),
        ));
    }
    let expected_elems = in_dim
        .checked_mul(out_dim)
        .ok_or_else(|| tensor_bounds_err(&view, "2D matrix dimensions overflowed usize"))?;
    if view.num_elements() != expected_elems {
        return Err(tensor_bounds_err(
            &view,
            format!(
                "2D tensor '{}' element count {} != {expected_elems}",
                view.name(),
                view.num_elements()
            ),
        ));
    }
    let required_bytes = view.quant().storage_bytes(view.name(), expected_elems)?;
    if view.as_bytes().len() < required_bytes {
        return Err(tensor_bounds_err(
            &view,
            format!(
                "tensor '{}' byte length {} < required {required_bytes}",
                view.name(),
                view.as_bytes().len()
            ),
        ));
    }
    Ok(view)
}

pub(crate) fn require_layer_2d_matvec<'w>(
    weights: &'w dyn ModelWeights,
    arch: &str,
    layer: usize,
    gguf_suffix: &str,
    hf_suffixes: &[&str],
    in_dim: usize,
    out_dim: usize,
) -> Result<TensorView<'w>, K3Error> {
    let view = lookup_layer_tensor(weights, layer, gguf_suffix, hf_suffixes)?;
    validate_2d_matvec_view(view, arch, in_dim, out_dim)
}

pub(crate) fn embed_token_hidden(
    arch: &str,
    token_id: u32,
    hidden_dim: usize,
    weights: &dyn ModelWeights,
) -> Result<Vec<f32>, K3Error> {
    let embd = lookup_tensor_aliases(weights, TOKEN_EMBD_ALIASES)?;
    if !is_supported_gemma_quant(embd.quant()) {
        return Err(K3Error::unsupported_tensor_quant(
            arch,
            embd.name(),
            embd.quant(),
        ));
    }
    let shape = embd.shape();
    if shape.len() != 2 {
        return Err(tensor_bounds_err(
            &embd,
            format!(
                "token_embd tensor '{}' must be 2D, got shape {shape:?}",
                embd.name()
            ),
        ));
    }
    let row_count = if shape[0] == hidden_dim {
        shape[1]
    } else if shape[1] == hidden_dim {
        shape[0]
    } else {
        return Err(tensor_bounds_err(
            &embd,
            format!("token_embd shape {shape:?} does not match hidden_dim={hidden_dim}"),
        ));
    };
    if row_count == 0 {
        return Err(tensor_bounds_err(
            &embd,
            "token_embd vocab row count must be > 0",
        ));
    }
    let token_idx = usize::try_from(token_id)
        .map_err(|_| tensor_bounds_err(&embd, "token_id exceeds usize range"))?;
    if token_idx >= row_count {
        return Err(tensor_bounds_err(
            &embd,
            format!("token_id {token_idx} out of bounds for embedding vocab rows {row_count}"),
        ));
    }

    let mut hidden = vec![0.0_f32; hidden_dim];
    dequantize_token_row(&embd, token_idx, &mut hidden)?;
    for (idx, val) in hidden.iter().enumerate() {
        if !val.is_finite() {
            return Err(tensor_bounds_err(
                &embd,
                format!("non-finite embedding value ({val}) for token {token_idx} at dim {idx}"),
            ));
        }
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "hidden_dim fits comfortably in f32"
    )]
    let scale = (hidden_dim as f32).sqrt();
    for v in &mut hidden {
        *v *= scale;
    }
    Ok(hidden)
}
