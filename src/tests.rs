use std::collections::BTreeMap;

use jero_core::gguf::test_helpers::write_gguf_value;
use jero_core::safetensors::test_helpers::create_synthetic_safetensors;
use jero_core::{
    ArchRecipe, ExecutionContext, GGUF_MAGIC, GGUF_VERSION_V3, GgufFile, GgufValue, K3Error,
    ModelWeights, QuantType, SafeTensorsFile, TensorView, align_up,
};

use super::{
    CRATE_LICENSE, CRATE_NAME, ExpertWeights, Gemma4Recipe, GemmaRecipe, apply_gemma4_ple_residual,
    crate_description, dispatch_sparse_moe_layer, matvec_any, select_top_k_experts,
    write_and_slice_sliding_kv,
};

struct OwnedTensor {
    quant: QuantType,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

struct MapWeights {
    tensors: BTreeMap<String, OwnedTensor>,
}

impl MapWeights {
    fn new() -> Self {
        Self {
            tensors: BTreeMap::new(),
        }
    }

    fn insert_f32(&mut self, name: &str, shape: &[usize], vals: &[f32]) {
        let mut bytes = Vec::with_capacity(vals.len() * 4);
        for &v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.tensors.insert(
            name.to_owned(),
            OwnedTensor {
                quant: QuantType::F32,
                shape: shape.to_vec(),
                bytes,
            },
        );
    }

    fn insert_raw(&mut self, name: &str, quant: QuantType, shape: &[usize], bytes: Vec<u8>) {
        self.tensors.insert(
            name.to_owned(),
            OwnedTensor {
                quant,
                shape: shape.to_vec(),
                bytes,
            },
        );
    }

    fn remove(&mut self, name: &str) {
        self.tensors.remove(name);
    }
}

impl ModelWeights for MapWeights {
    fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
        let (stored_name, entry) =
            self.tensors
                .get_key_value(name)
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: name.to_owned(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: 0,
                    reason: "tensor not present in test map".to_owned(),
                })?;
        TensorView::from_bytes(stored_name, entry.quant, &entry.shape, 0, &entry.bytes)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    fn total_weight_bytes(&self) -> usize {
        self.tensors.values().map(|t| t.bytes.len()).sum()
    }
}

fn encode_f32_vec(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn encode_bf16_vec(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        let bits = u16::try_from(v.to_bits() >> 16).expect("shifted bits fit u16");
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

fn encode_q8_0_row(val_per_elem_q: i8, scale_f16: u16, blocks: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(blocks * 34);
    for _ in 0..blocks {
        bytes.extend_from_slice(&scale_f16.to_le_bytes());
        bytes.resize(bytes.len() + 32, val_per_elem_q.to_ne_bytes()[0]);
    }
    bytes
}

fn encode_q4_0_row(nibble_byte: u8, scale_f16: u16, blocks: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(blocks * 18);
    for _ in 0..blocks {
        bytes.extend_from_slice(&scale_f16.to_le_bytes());
        bytes.resize(bytes.len() + 16, nibble_byte);
    }
    bytes
}

fn assert_logits_parity(lhs: &[f32], rhs: &[f32], label: &str) {
    assert_eq!(lhs.len(), rhs.len(), "{label}: logit lengths differ");
    for (idx, (&a, &b)) in lhs.iter().zip(rhs.iter()).enumerate() {
        #[cfg(not(miri))]
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{label} diverged at logit {idx}: {a} vs {b}"
        );
        // Miri intentionally perturbs low ULPs of libm transcendentals (sin/cos/tanh/exp).
        #[cfg(miri)]
        assert!(
            (a - b).abs() < 1e-5,
            "{label} exceeded Miri float tolerance at logit {idx}: {a} vs {b}"
        );
    }
}

fn deterministic_pattern(len: usize, seed: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            #[expect(clippy::cast_precision_loss, reason = "small test pattern index")]
            let raw = (((i * 7 + seed * 13) % 19) as f32) - 9.0;
            raw * scale
        })
        .collect()
}

#[derive(Clone, Copy)]
struct DenseWeightSpec {
    num_layers: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    vocab_size: usize,
    use_qk_norm: bool,
    include_explicit_output: bool,
}

impl DenseWeightSpec {
    const fn tiny(num_layers: usize) -> Self {
        Self {
            num_layers,
            hidden_dim: 8,
            ffn_dim: 8,
            n_heads_q: 2,
            n_heads_kv: 1,
            head_dim: 4,
            vocab_size: 16,
            use_qk_norm: true,
            include_explicit_output: true,
        }
    }
}

fn populate_dense_gemma_weights(map: &mut MapWeights, spec: DenseWeightSpec) {
    let q_dim = spec.n_heads_q * spec.head_dim;
    let kv_dim = spec.n_heads_kv * spec.head_dim;

    map.insert_f32(
        "token_embd.weight",
        &[spec.hidden_dim, spec.vocab_size],
        &deterministic_pattern(spec.hidden_dim * spec.vocab_size, 1, 0.05),
    );
    map.insert_f32(
        "output_norm.weight",
        &[spec.hidden_dim],
        &deterministic_pattern(spec.hidden_dim, 2, 0.02),
    );
    if spec.include_explicit_output {
        map.insert_f32(
            "output.weight",
            &[spec.hidden_dim, spec.vocab_size],
            &deterministic_pattern(spec.hidden_dim * spec.vocab_size, 3, 0.04),
        );
    }

    for l in 0..spec.num_layers {
        let s = l + 4;
        map.insert_f32(
            &format!("blk.{l}.attn_norm.weight"),
            &[spec.hidden_dim],
            &deterministic_pattern(spec.hidden_dim, s, 0.02),
        );
        map.insert_f32(
            &format!("blk.{l}.attn_q.weight"),
            &[spec.hidden_dim, q_dim],
            &deterministic_pattern(spec.hidden_dim * q_dim, s + 1, 0.03),
        );
        map.insert_f32(
            &format!("blk.{l}.attn_k.weight"),
            &[spec.hidden_dim, kv_dim],
            &deterministic_pattern(spec.hidden_dim * kv_dim, s + 2, 0.03),
        );
        map.insert_f32(
            &format!("blk.{l}.attn_v.weight"),
            &[spec.hidden_dim, kv_dim],
            &deterministic_pattern(spec.hidden_dim * kv_dim, s + 3, 0.03),
        );
        if spec.use_qk_norm {
            map.insert_f32(
                &format!("blk.{l}.attn_q_norm.weight"),
                &[spec.head_dim],
                &deterministic_pattern(spec.head_dim, s + 4, 0.02),
            );
            map.insert_f32(
                &format!("blk.{l}.attn_k_norm.weight"),
                &[spec.head_dim],
                &deterministic_pattern(spec.head_dim, s + 5, 0.02),
            );
        }
        map.insert_f32(
            &format!("blk.{l}.attn_output.weight"),
            &[q_dim, spec.hidden_dim],
            &deterministic_pattern(q_dim * spec.hidden_dim, s + 6, 0.03),
        );
        map.insert_f32(
            &format!("blk.{l}.post_attention_norm.weight"),
            &[spec.hidden_dim],
            &deterministic_pattern(spec.hidden_dim, s + 7, 0.02),
        );
        map.insert_f32(
            &format!("blk.{l}.ffn_norm.weight"),
            &[spec.hidden_dim],
            &deterministic_pattern(spec.hidden_dim, s + 8, 0.02),
        );
        map.insert_f32(
            &format!("blk.{l}.ffn_gate.weight"),
            &[spec.hidden_dim, spec.ffn_dim],
            &deterministic_pattern(spec.hidden_dim * spec.ffn_dim, s + 9, 0.03),
        );
        map.insert_f32(
            &format!("blk.{l}.ffn_up.weight"),
            &[spec.hidden_dim, spec.ffn_dim],
            &deterministic_pattern(spec.hidden_dim * spec.ffn_dim, s + 10, 0.03),
        );
        map.insert_f32(
            &format!("blk.{l}.ffn_down.weight"),
            &[spec.ffn_dim, spec.hidden_dim],
            &deterministic_pattern(spec.ffn_dim * spec.hidden_dim, s + 11, 0.03),
        );
        map.insert_f32(
            &format!("blk.{l}.post_ffw_norm.weight"),
            &[spec.hidden_dim],
            &deterministic_pattern(spec.hidden_dim, s + 12, 0.02),
        );
    }
}

fn tiny_gemma3_recipe(num_layers: usize) -> GemmaRecipe {
    let mut r = GemmaRecipe::new_gemma3();
    r.num_layers = num_layers;
    r.hidden_dim = 8;
    r.ffn_dim = 8;
    r.n_heads_q = 2;
    r.n_heads_kv = 1;
    r.head_dim = 4;
    r.vocab_size = 16;
    r.sliding_window = 2;
    r
}

fn tiny_gemma4_ple_recipe(num_layers: usize) -> Gemma4Recipe {
    let mut edge = Gemma4Recipe::new_edge_ple(32);
    edge.num_layers = num_layers;
    edge.hidden_dim = 8;
    edge.ffn_dim = 8;
    edge.n_heads_q = 2;
    edge.n_heads_kv = 1;
    edge.head_dim = 4;
    edge.vocab_size = 16;
    edge.sliding_window = 2;
    edge
}

fn tiny_gemma4_moe_recipe() -> Gemma4Recipe {
    let mut moe = Gemma4Recipe::new_moe_26b_a4b();
    moe.num_layers = 1;
    moe.hidden_dim = 8;
    moe.ffn_dim = 4;
    moe.n_heads_q = 2;
    moe.n_heads_kv = 1;
    moe.head_dim = 4;
    moe.vocab_size = 16;
    moe.sliding_window = 2;
    moe.num_experts = Some(3);
    moe.num_experts_per_tok = Some(2);
    moe
}

fn write_gguf_string(buf: &mut Vec<u8>, s: &str) {
    let len = u64::try_from(s.len()).expect("string len fits u64");
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn create_gguf_with_payloads(
    architecture: &str,
    alignment: usize,
    kvs: &[(&str, GgufValue<'_>)],
    tensors: &[(&str, QuantType, &[usize], &[u8])],
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    let tensor_count = u64::try_from(tensors.len()).expect("tensor count fits u64");
    buf.extend_from_slice(&tensor_count.to_le_bytes());

    let mut all_kvs = vec![
        ("general.architecture", GgufValue::String(architecture)),
        (
            "general.alignment",
            GgufValue::Uint32(u32::try_from(alignment).expect("alignment fits u32")),
        ),
    ];
    all_kvs.extend_from_slice(kvs);
    let kv_len = u64::try_from(all_kvs.len()).expect("kv len fits u64");
    buf.extend_from_slice(&kv_len.to_le_bytes());

    for (k, v) in &all_kvs {
        write_gguf_string(&mut buf, k);
        write_gguf_value(&mut buf, v);
    }

    let mut rel_offset = 0usize;
    let mut data_section = Vec::new();
    for &(name, quant, shape, raw_bytes) in tensors {
        let num_elems: usize = shape.iter().product();
        let expected_bytes = quant.storage_bytes(name, num_elems).expect("valid storage");
        assert_eq!(raw_bytes.len(), expected_bytes);

        let aligned_rel = align_up(rel_offset, alignment);
        data_section.resize(aligned_rel, 0);
        write_gguf_string(&mut buf, name);
        let rank = u32::try_from(shape.len()).expect("rank fits u32");
        buf.extend_from_slice(&rank.to_le_bytes());
        for &dim in shape {
            buf.extend_from_slice(&u64::try_from(dim).expect("dim fits u64").to_le_bytes());
        }
        buf.extend_from_slice(&quant.ggml_type_id().to_le_bytes());
        buf.extend_from_slice(
            &u64::try_from(aligned_rel)
                .expect("rel offset fits u64")
                .to_le_bytes(),
        );
        data_section.extend_from_slice(raw_bytes);
        rel_offset = aligned_rel + raw_bytes.len();
    }

    let aligned_header_end = align_up(buf.len(), alignment);
    buf.resize(aligned_header_end, 0);
    buf.extend_from_slice(&data_section);
    buf
}

#[test]
fn bootstrap_metadata_matches_policy() {
    assert_eq!(CRATE_NAME, "jero-arch-gemma");
    assert_eq!(
        CRATE_LICENSE,
        "LGPL-3.0-only WITH LGPL-3.0-linking-exception"
    );
    assert_ne!(crate_description(), "");
}

#[test]
fn test_gemma2_and_gemma3_sliding_window_attention() {
    let g2 = GemmaRecipe::new_gemma2();
    assert_eq!(g2.architecture_id(), "gemma2");
    assert!(!g2.is_global_layer(0));
    assert!(g2.is_global_layer(1));
    assert!(!g2.is_global_layer(2));
    assert!(g2.is_global_layer(3));
    assert_eq!(g2.sliding_window, 4096);
    assert_eq!(g2.attn_logit_softcap, Some(50.0));
    assert_eq!(g2.final_logit_softcap, Some(30.0));
    assert!(!g2.use_qk_norm);

    let g3 = GemmaRecipe::new_gemma3();
    assert_eq!(g3.architecture_id(), "gemma3");
    assert_eq!(g3.sliding_window, 1024);
    assert!(g3.use_qk_norm);
    // 5:1 interleaved: layer 5, 11, ... global; 0..4, 6..10 local
    for l in 0..12 {
        let expected_global = (l + 1) % 6 == 0;
        assert_eq!(g3.is_global_layer(l), expected_global);
        let theta = g3.layer_rope_theta(l);
        if expected_global {
            assert!((theta - 1_000_000.0).abs() < f32::EPSILON);
        } else {
            assert!((theta - 10_000.0).abs() < f32::EPSILON);
        }
    }
}

#[test]
fn test_gemma2_gemma3_and_functiongemma_recipe_forward() {
    let mut g2 = GemmaRecipe::new_gemma2();
    g2.num_layers = 2;
    g2.hidden_dim = 8;
    g2.ffn_dim = 8;
    g2.n_heads_q = 2;
    g2.n_heads_kv = 1;
    g2.head_dim = 4;
    g2.vocab_size = 16;
    g2.sliding_window = 2;
    g2.final_logit_softcap = Some(2.0);

    let mut weights_g2 = MapWeights::new();
    populate_dense_gemma_weights(
        &mut weights_g2,
        DenseWeightSpec {
            use_qk_norm: false,
            ..DenseWeightSpec::tiny(2)
        },
    );

    let mut logits_g2 = vec![0.0_f32; 16];
    let mut kv_g2 = vec![0.0_f32; 64];
    let mut ctx = ExecutionContext::new(&[1, 2, 3], 8, &mut logits_g2, &mut kv_g2)
        .expect("execution context");
    g2.prefill_step(&mut ctx, &weights_g2)
        .expect("gemma2 prefill succeeds");
    assert_eq!(ctx.position(), 3);
    assert!(ctx.logits().iter().all(|v| v.is_finite() && v.abs() < 2.0));
    assert!(ctx.logits().iter().any(|&v| v.abs() > 1e-6));

    ctx.prepare_decode(4).expect("prepare decode");
    g2.decode_step(&mut ctx, &weights_g2)
        .expect("gemma2 decode succeeds");
    assert_eq!(ctx.position(), 4);
    assert!(ctx.logits().iter().all(|v| v.is_finite() && v.abs() < 2.0));

    // Gemma 3 with tied token_embd.weight (no explicit output.weight) and QK-norm
    let mut g3 = GemmaRecipe::new_gemma3();
    g3.num_layers = 6;
    g3.hidden_dim = 8;
    g3.ffn_dim = 8;
    g3.n_heads_q = 2;
    g3.n_heads_kv = 1;
    g3.head_dim = 4;
    g3.vocab_size = 16;
    g3.sliding_window = 2;

    let mut weights_g3 = MapWeights::new();
    populate_dense_gemma_weights(
        &mut weights_g3,
        DenseWeightSpec {
            include_explicit_output: false,
            ..DenseWeightSpec::tiny(6)
        },
    );

    let mut logits_g3 = vec![0.0_f32; 16];
    let mut kv_g3 = vec![0.0_f32; 6 * 32];
    let mut ctx3 =
        ExecutionContext::new(&[5, 6], 8, &mut logits_g3, &mut kv_g3).expect("execution context");
    g3.prefill_step(&mut ctx3, &weights_g3)
        .expect("gemma3 prefill with tied LM head succeeds");
    ctx3.prepare_decode(7).expect("prepare decode");
    g3.decode_step(&mut ctx3, &weights_g3)
        .expect("gemma3 decode succeeds");
    assert_eq!(ctx3.position(), 3);
    assert!(ctx3.logits().iter().all(|v| v.is_finite()));
    assert!(ctx3.logits().iter().any(|&v| v.abs() > 1e-6));

    // FunctionGemma forward pass
    let mut fg = GemmaRecipe::new_functiongemma();
    fg.num_layers = 2;
    fg.hidden_dim = 8;
    fg.ffn_dim = 8;
    fg.n_heads_q = 2;
    fg.n_heads_kv = 1;
    fg.head_dim = 4;
    fg.vocab_size = 16;
    fg.sliding_window = 2;
    let mut logits_fg = vec![0.0_f32; 16];
    let mut kv_fg = vec![0.0_f32; 64];
    let mut ctx_fg = ExecutionContext::new(&[0, 1], 8, &mut logits_fg, &mut kv_fg).unwrap();
    fg.prefill_step(&mut ctx_fg, &weights_g3)
        .expect("functiongemma prefill succeeds");
    assert_eq!(ctx_fg.position(), 2);
}

#[test]
fn test_gemma4_ple_quantized_lookup() {
    let mut q8_bytes = encode_q8_0_row(1, 0x3C00, 2);
    q8_bytes.extend(encode_q8_0_row(2, 0x3C00, 2));
    let ple_q8 = TensorView::from_bytes(
        "per_layer_token_embd.weight",
        QuantType::Q8_0,
        &[64, 2],
        0,
        &q8_bytes,
    )
    .expect("q8 PLE view");

    let mut row1 = [0.0_f32; 64];
    jero_kernels_rvv::dequantize_token_row(&ple_q8, 1, &mut row1).expect("dequantize row 1");
    assert!(row1.iter().all(|&v| (v - 2.0).abs() < 1e-5));

    let mut q4_bytes = encode_q4_0_row(0xAA, 0x3800, 1);
    q4_bytes.extend(encode_q4_0_row(0xCC, 0x3800, 1));
    let ple_q4 = TensorView::from_bytes(
        "per_layer_token_embd.weight",
        QuantType::Q4_0,
        &[32, 2],
        0,
        &q4_bytes,
    )
    .expect("q4 PLE view");
    let mut row0_q4 = [0.0_f32; 32];
    jero_kernels_rvv::dequantize_token_row(&ple_q4, 0, &mut row0_q4).expect("dequantize q4 row 0");
    assert!(row0_q4.iter().all(|&v| (v - 1.0).abs() < 1e-5));
}

#[test]
fn test_gemma4_ple_gated_projection() {
    let mut q8_bytes = encode_q8_0_row(1, 0x3C00, 2);
    q8_bytes.extend(encode_q8_0_row(2, 0x3C00, 2));
    let ple_embd = TensorView::from_bytes(
        "per_layer_token_embd.weight",
        QuantType::Q8_0,
        &[64, 2],
        0,
        &q8_bytes,
    )
    .expect("ple embd view");

    let gate_weights = vec![0.25_f32; 32 * 4];
    let gate_bytes = encode_f32_vec(&gate_weights);
    let ple_inp_gate = TensorView::from_bytes(
        "blk.1.per_layer_inp_gate.weight",
        QuantType::F32,
        &[32, 4],
        0,
        &gate_bytes,
    )
    .expect("ple inp gate view");

    let proj_weights = vec![0.125_f32; 4 * 32];
    let proj_bytes = encode_f32_vec(&proj_weights);
    let ple_proj = TensorView::from_bytes(
        "blk.1.per_layer_proj.weight",
        QuantType::F32,
        &[4, 32],
        0,
        &proj_bytes,
    )
    .expect("ple proj view");

    let mut hidden = [1.0_f32, 1.0, 1.0, 1.0];
    let initial = hidden;
    let ple_norm_weight = [0.0_f32; 4];

    apply_gemma4_ple_residual(
        &mut hidden,
        1,
        1,
        &ple_embd,
        &ple_inp_gate,
        &ple_proj,
        &ple_norm_weight,
        1e-6,
    )
    .expect("apply_gemma4_ple_residual succeeds");

    for (&before, &after) in initial.iter().zip(hidden.iter()) {
        assert!(after.is_finite());
        assert!(after > before);
    }
}

#[test]
fn test_select_top_k_experts_deterministic() {
    let tied = [0.1_f32, 3.0, 0.5, 1.0, 3.0, 2.0];
    let top2 = select_top_k_experts(&tied, 2, true);
    assert_eq!(top2.len(), 2);
    assert_eq!(top2[0].0, 1);
    assert_eq!(top2[1].0, 4);
    assert!((top2[0].1 - 0.5).abs() < 1e-6);
    assert!((top2[1].1 - 0.5).abs() < 1e-6);

    let raw = select_top_k_experts(&tied, 2, false);
    assert!(raw[0].1 + raw[1].1 < 0.99);

    assert_eq!(select_top_k_experts(&[], 2, true), []);
    assert_eq!(select_top_k_experts(&tied, 0, true), []);
}

#[test]
fn test_gemma4_26b_a4b_moe_routing() {
    fn make_expert<'a>(
        g_b: &'a [u8],
        u_b: &'a [u8],
        d_b: &'a [u8],
        shape_gu: &'a [usize],
        shape_d: &'a [usize],
    ) -> ExpertWeights<'a> {
        ExpertWeights {
            gate: TensorView::from_bytes("g", QuantType::F32, shape_gu, 0, g_b).unwrap(),
            up: TensorView::from_bytes("u", QuantType::F32, shape_gu, 0, u_b).unwrap(),
            down: TensorView::from_bytes("d", QuantType::F32, shape_d, 0, d_b).unwrap(),
        }
    }

    let router_vals = [
        5.0_f32, 0.0, 0.0, 0.0, // expert 0
        -9.0, 0.0, 0.0, 0.0, // expert 1 (inactive)
        7.0, 0.0, 0.0, 0.0, // expert 2
        -9.0, 0.0, 0.0, 0.0, // expert 3 (inactive)
    ];
    let router_bytes = encode_f32_vec(&router_vals);
    let router_view = TensorView::from_bytes(
        "blk.0.ffn_gate_inp.weight",
        QuantType::F32,
        &[4, 4],
        0,
        &router_bytes,
    )
    .expect("router view");

    let ones_ffn = encode_f32_vec(&[1.0_f32; 2 * 4]);
    let twos_ffn = encode_f32_vec(&[2.0_f32; 2 * 4]);
    let bad_tiny = encode_f32_vec(&[0.0_f32; 1]);

    let experts = [
        make_expert(&ones_ffn, &ones_ffn, &ones_ffn, &[2, 4], &[4, 2]),
        make_expert(&bad_tiny, &bad_tiny, &bad_tiny, &[1, 1], &[1, 1]),
        make_expert(&twos_ffn, &twos_ffn, &twos_ffn, &[2, 4], &[4, 2]),
        make_expert(&bad_tiny, &bad_tiny, &bad_tiny, &[1, 1], &[1, 1]),
    ];

    let hidden = [1.0_f32, 0.5, 0.5, 0.5];
    let mut out = [0.0_f32; 4];
    dispatch_sparse_moe_layer(&mut out, &hidden, &router_view, &experts, 2)
        .expect("active top-2 MoE routing bypasses inactive experts cleanly");
    assert!(out.iter().all(|&v| v.is_finite() && v > 0.0));
}

#[test]
fn test_gemma4_ple_and_moe_descriptors() {
    let edge = Gemma4Recipe::new_edge_ple(32);
    assert_eq!(edge.architecture_id(), "gemma4");
    assert_eq!(edge.ple_dim, Some(32));

    let moe = Gemma4Recipe::new_moe_26b_a4b();
    assert_eq!(moe.architecture_id(), "gemma4");
    assert_eq!(moe.num_experts, Some(128));
    assert_eq!(moe.num_experts_per_tok, Some(8));
    assert!(moe.is_global_layer(5));
    assert!(!moe.is_global_layer(4));
    assert!((moe.layer_rope_theta(5) - 1_000_000.0).abs() < f32::EPSILON);
    assert!((moe.layer_rope_theta(4) - 10_000.0).abs() < f32::EPSILON);
}

fn build_gemma4_ple_weights(num_layers: usize, q8_rows: usize) -> MapWeights {
    let mut weights = MapWeights::new();
    populate_dense_gemma_weights(&mut weights, DenseWeightSpec::tiny(num_layers));
    let mut q8_bytes = encode_q8_0_row(1, 0x3C00, q8_rows);
    q8_bytes.extend(encode_q8_0_row(2, 0x3C00, q8_rows));
    weights.insert_raw(
        "per_layer_token_embd.weight",
        QuantType::Q8_0,
        &[q8_rows * 32, 2],
        q8_bytes,
    );
    for l in 0..num_layers {
        weights.insert_f32(
            &format!("blk.{l}.per_layer_inp_gate.weight"),
            &[32, 8],
            &[0.125; 32 * 8],
        );
        weights.insert_f32(
            &format!("blk.{l}.per_layer_proj.weight"),
            &[8, 32],
            &[0.125; 8 * 32],
        );
        weights.insert_f32(
            &format!("blk.{l}.per_layer_post_norm.weight"),
            &[8],
            &[0.05; 8],
        );
    }
    weights
}

fn build_gemma4_moe_base_weights() -> MapWeights {
    let mut weights = MapWeights::new();
    populate_dense_gemma_weights(
        &mut weights,
        DenseWeightSpec {
            ffn_dim: 4,
            ..DenseWeightSpec::tiny(1)
        },
    );
    weights.remove("blk.0.ffn_gate.weight");
    weights.remove("blk.0.ffn_up.weight");
    weights.remove("blk.0.ffn_down.weight");
    weights.insert_f32(
        "blk.0.ffn_gate_inp.weight",
        &[8, 3],
        &deterministic_pattern(8 * 3, 21, 0.2),
    );
    weights
}

#[test]
fn test_gemma4_ple_e2e() {
    let ple_weights = build_gemma4_ple_weights(2, 2);
    let edge = tiny_gemma4_ple_recipe(2);

    let mut logits = vec![0.0_f32; 16];
    let mut kv = vec![0.0_f32; 64];
    let mut ctx = ExecutionContext::new(&[0, 1], 8, &mut logits, &mut kv).unwrap();
    edge.prefill_step(&mut ctx, &ple_weights)
        .expect("gemma4 PLE prefill e2e");
    ctx.prepare_decode(1).unwrap();
    edge.decode_step(&mut ctx, &ple_weights)
        .expect("gemma4 PLE decode e2e");
    assert!(ctx.logits().iter().all(|v| v.is_finite()));
}

#[test]
fn test_gemma4_moe_split_and_stacked_3d_parity() {
    let mut moe_weights = build_gemma4_moe_base_weights();
    for e in 0..3 {
        moe_weights.insert_f32(
            &format!("blk.0.ffn_gate_exps.{e}.weight"),
            &[8, 4],
            &deterministic_pattern(8 * 4, 30 + e, 0.05),
        );
        moe_weights.insert_f32(
            &format!("blk.0.ffn_up_exps.{e}.weight"),
            &[8, 4],
            &deterministic_pattern(8 * 4, 40 + e, 0.05),
        );
        moe_weights.insert_f32(
            &format!("blk.0.ffn_down_exps.{e}.weight"),
            &[4, 8],
            &deterministic_pattern(4 * 8, 50 + e, 0.05),
        );
    }

    let moe = tiny_gemma4_moe_recipe();
    let mut logits_m = vec![0.0_f32; 16];
    let mut kv_m = vec![0.0_f32; 32];
    let mut ctx_m = ExecutionContext::new(&[1, 2], 8, &mut logits_m, &mut kv_m).unwrap();
    moe.prefill_step(&mut ctx_m, &moe_weights)
        .expect("gemma4 MoE prefill e2e");
    ctx_m.prepare_decode(3).unwrap();
    moe.decode_step(&mut ctx_m, &moe_weights)
        .expect("gemma4 MoE decode e2e");
    assert!(ctx_m.logits().iter().all(|v| v.is_finite()));

    let mut stacked_weights = build_gemma4_moe_base_weights();
    let mut gate_stack = Vec::new();
    let mut up_stack = Vec::new();
    let mut down_stack = Vec::new();
    for e in 0..3 {
        gate_stack.extend(deterministic_pattern(8 * 4, 30 + e, 0.05));
        up_stack.extend(deterministic_pattern(8 * 4, 40 + e, 0.05));
        down_stack.extend(deterministic_pattern(4 * 8, 50 + e, 0.05));
    }
    stacked_weights.insert_f32("blk.0.ffn_gate_exps.weight", &[8, 4, 3], &gate_stack);
    stacked_weights.insert_f32("blk.0.ffn_up_exps.weight", &[8, 4, 3], &up_stack);
    stacked_weights.insert_f32("blk.0.ffn_down_exps.weight", &[4, 8, 3], &down_stack);

    let mut logits_s = vec![0.0_f32; 16];
    let mut kv_s = vec![0.0_f32; 32];
    let mut ctx_s = ExecutionContext::new(&[1, 2], 8, &mut logits_s, &mut kv_s).unwrap();
    moe.prefill_step(&mut ctx_s, &stacked_weights)
        .expect("gemma4 stacked 3D MoE prefill");
    ctx_s.prepare_decode(3).unwrap();
    moe.decode_step(&mut ctx_s, &stacked_weights)
        .expect("gemma4 stacked 3D MoE decode");
    assert_logits_parity(&logits_m, &logits_s, "split vs 3D MoE");
}

#[test]
fn test_sliding_window_kv_ring_wraparound() {
    let window_size = 4;
    let kv_dim = 3;
    let mut layer_kv = vec![0.0_f32; 2 * window_size * kv_dim];

    for pos in 0..6 {
        let marker = f32::from(u16::try_from(pos + 1).expect("pos fits u16"));
        let k = [marker, marker * 2.0, marker * 3.0];
        let v = [-marker, -marker * 2.0, -marker * 3.0];
        let (k_slice, v_slice, active_len) =
            write_and_slice_sliding_kv(&mut layer_kv, pos, window_size, kv_dim, &k, &v)
                .expect("sliding kv write succeeds");
        assert_eq!(active_len, (pos + 1).min(window_size));
        assert_eq!(k_slice.len(), active_len * kv_dim);
        assert_eq!(v_slice.len(), active_len * kv_dim);
        let slot = pos % window_size;
        assert!((k_slice[slot * kv_dim] - marker).abs() < f32::EPSILON);
    }

    assert!(matches!(
        write_and_slice_sliding_kv(&mut layer_kv, 0, 0, kv_dim, &[1.0; 3], &[1.0; 3]),
        Err(K3Error::TensorBoundsError { .. })
    ));
    let mut short_kv = vec![0.0_f32; 4];
    assert!(matches!(
        write_and_slice_sliding_kv(&mut short_kv, 0, window_size, kv_dim, &[1.0; 3], &[1.0; 3]),
        Err(K3Error::TensorBoundsError { .. })
    ));
}

#[test]
fn test_adr045_missing_and_mismatched_weights_fail_closed() {
    let g3 = tiny_gemma3_recipe(1);

    // 1. Empty weights must fail closed on both prefill_step and decode_step
    let empty = MapWeights::new();
    let mut logits = vec![0.0_f32; 16];
    let mut kv = vec![0.0_f32; 32];
    let mut ctx = ExecutionContext::new(&[1, 2], 8, &mut logits, &mut kv).unwrap();
    assert!(matches!(
        g3.prefill_step(&mut ctx, &empty),
        Err(K3Error::TensorBoundsError { .. })
    ));
    ctx.prepare_decode(1).unwrap();
    assert!(matches!(
        g3.decode_step(&mut ctx, &empty),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // 2. Removing any mandatory tensor must fail closed
    let mandatory_tensors = [
        "token_embd.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_q_norm.weight",
        "blk.0.attn_k_norm.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "output_norm.weight",
    ];
    for missing_name in mandatory_tensors {
        let mut weights = MapWeights::new();
        populate_dense_gemma_weights(&mut weights, DenseWeightSpec::tiny(1));
        weights.remove(missing_name);

        let mut test_logits = vec![0.0_f32; 16];
        let mut test_kv = vec![0.0_f32; 32];
        let mut test_ctx = ExecutionContext::new(&[1], 8, &mut test_logits, &mut test_kv).unwrap();
        let err = g3
            .prefill_step(&mut test_ctx, &weights)
            .expect_err("missing mandatory tensor must fail closed");
        assert!(
            matches!(err, K3Error::TensorBoundsError { .. }),
            "unexpected error for missing {missing_name}: {err:?}"
        );
    }

    // 3. Dimension mismatch on mandatory tensors must fail closed
    let mismatch_cases: [(&str, &[usize], usize); 3] = [
        ("blk.0.attn_q.weight", &[8, 4], 32),
        ("output_norm.weight", &[4], 4),
        ("output.weight", &[8, 12], 96),
    ];
    for (tensor_name, shape, elems) in mismatch_cases {
        let mut bad = MapWeights::new();
        populate_dense_gemma_weights(&mut bad, DenseWeightSpec::tiny(1));
        bad.insert_f32(tensor_name, shape, &vec![0.1; elems]);
        let mut test_logits = vec![0.0_f32; 16];
        let mut test_kv = vec![0.0_f32; 32];
        let mut test_ctx = ExecutionContext::new(&[1], 8, &mut test_logits, &mut test_kv).unwrap();
        assert!(matches!(
            g3.prefill_step(&mut test_ctx, &bad),
            Err(K3Error::TensorBoundsError { .. })
        ));
    }
}

#[test]
fn test_adr045_bounds_quant_and_gemma4_fail_closed() {
    let g3 = tiny_gemma3_recipe(1);
    let mut test_logits = vec![0.0_f32; 16];
    let mut test_kv = vec![0.0_f32; 32];

    let mut valid_weights = MapWeights::new();
    populate_dense_gemma_weights(&mut valid_weights, DenseWeightSpec::tiny(1));
    let mut oob_ctx = ExecutionContext::new(&[99], 8, &mut test_logits, &mut test_kv).unwrap();
    assert!(matches!(
        g3.prefill_step(&mut oob_ctx, &valid_weights),
        Err(K3Error::TensorBoundsError { .. })
    ));

    let mut bad_quant = MapWeights::new();
    populate_dense_gemma_weights(&mut bad_quant, DenseWeightSpec::tiny(1));
    bad_quant.insert_raw(
        "blk.0.ffn_gate.weight",
        QuantType::TQ1_0,
        &[256, 1],
        vec![0; 54],
    );
    let mut quant_ctx = ExecutionContext::new(&[1], 8, &mut test_logits, &mut test_kv).unwrap();
    assert!(matches!(
        g3.prefill_step(&mut quant_ctx, &bad_quant),
        Err(K3Error::CapabilityMismatch { .. })
    ));

    let edge = tiny_gemma4_ple_recipe(1);
    let mut ple_ctx = ExecutionContext::new(&[1], 8, &mut test_logits, &mut test_kv).unwrap();
    assert!(matches!(
        edge.prefill_step(&mut ple_ctx, &valid_weights),
        Err(K3Error::TensorBoundsError { .. })
    ));

    let moe = tiny_gemma4_moe_recipe();
    let mut moe_ctx = ExecutionContext::new(&[1], 8, &mut test_logits, &mut test_kv).unwrap();
    assert!(matches!(
        moe.prefill_step(&mut moe_ctx, &valid_weights),
        Err(K3Error::TensorBoundsError { .. })
    ));
}

#[test]
fn test_gemma3_soft_prompt_layer0_conditioning() {
    let mut g3 = tiny_gemma3_recipe(2);
    g3.sliding_window = 4;

    let mut weights = MapWeights::new();
    populate_dense_gemma_weights(&mut weights, DenseWeightSpec::tiny(2));

    let soft_a = deterministic_pattern(3 * 8, 77, 0.25);
    let soft_b = deterministic_pattern(3 * 8, 99, 0.45);
    let mut logits_a = vec![0.0_f32; 16];
    let mut kv_a = vec![0.0_f32; 2 * 2 * 16 * 4];
    let mut ctx_a = ExecutionContext::new(&[2, 5], 16, &mut logits_a, &mut kv_a)
        .unwrap()
        .with_soft_prompt_embeds(&soft_a, 3, 8)
        .expect("attach soft prompt A");
    g3.prefill_step(&mut ctx_a, &weights)
        .expect("soft prompt + discrete prefill succeeds");
    assert_eq!(ctx_a.position(), 5);
    ctx_a.prepare_decode(4).unwrap();
    g3.decode_step(&mut ctx_a, &weights)
        .expect("decode after soft prompt succeeds");
    assert_eq!(ctx_a.position(), 6);
    assert!(logits_a.iter().all(|v| v.is_finite()));

    let mut logits_b = vec![0.0_f32; 16];
    let mut kv_b = vec![0.0_f32; 2 * 2 * 16 * 4];
    let mut ctx_b = ExecutionContext::new(&[2, 5], 16, &mut logits_b, &mut kv_b)
        .unwrap()
        .with_soft_prompt_embeds(&soft_b, 3, 8)
        .expect("attach soft prompt B");
    g3.prefill_step(&mut ctx_b, &weights)
        .expect("soft prompt B prefill succeeds");
    ctx_b.prepare_decode(4).unwrap();
    g3.decode_step(&mut ctx_b, &weights)
        .expect("decode after soft prompt B succeeds");
    assert!(
        logits_a
            .iter()
            .zip(logits_b.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6),
        "soft prompt embeddings must condition KV cache and logits"
    );
}

#[test]
fn test_gemma4_soft_prompt_and_validation() {
    let mut g3 = tiny_gemma3_recipe(2);
    g3.sliding_window = 4;
    let mut weights = MapWeights::new();
    populate_dense_gemma_weights(&mut weights, DenseWeightSpec::tiny(2));
    let soft_a = deterministic_pattern(3 * 8, 77, 0.25);

    // Soft-prompt-only prefill (empty discrete prompt_tokens)
    let mut logits_only = vec![0.0_f32; 16];
    let mut kv_only = vec![0.0_f32; 2 * 2 * 16 * 4];
    let mut ctx_only = ExecutionContext::new(&[], 16, &mut logits_only, &mut kv_only)
        .unwrap()
        .with_soft_prompt_embeds(&soft_a, 3, 8)
        .unwrap();
    g3.prefill_step(&mut ctx_only, &weights)
        .expect("soft-prompt-only prefill succeeds");
    assert_eq!(ctx_only.position(), 3);
    ctx_only.prepare_decode(1).unwrap();
    g3.decode_step(&mut ctx_only, &weights).unwrap();
    assert_eq!(ctx_only.position(), 4);

    // Invalid soft-prompt dim != hidden_dim must fail closed
    let bad_dim_soft = vec![0.1_f32; 2 * 4];
    let mut bad_ctx = ExecutionContext::new(&[1], 16, &mut logits_only, &mut kv_only)
        .unwrap()
        .with_soft_prompt_embeds(&bad_dim_soft, 2, 4)
        .unwrap();
    assert!(matches!(
        g3.prefill_step(&mut bad_ctx, &weights),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // Soft-prompt + discrete prompt exceeding n_ctx must fail closed
    let mut oob_ctx = ExecutionContext::new(&[1, 2, 3], 5, &mut logits_only, &mut kv_only)
        .unwrap()
        .with_soft_prompt_embeds(&soft_a, 3, 8)
        .unwrap();
    assert!(matches!(
        g3.prefill_step(&mut oob_ctx, &weights),
        Err(K3Error::ContextLengthExceeded { .. })
    ));

    // Gemma4 Edge PLE + soft prompt conditioning
    let ple_weights = build_gemma4_ple_weights(1, 1);
    let mut edge = tiny_gemma4_ple_recipe(1);
    edge.sliding_window = 4;
    let mut logits_g4 = vec![0.0_f32; 16];
    let mut kv_g4 = vec![0.0_f32; 2 * 16 * 4];
    let mut ctx_g4 = ExecutionContext::new(&[0], 16, &mut logits_g4, &mut kv_g4)
        .unwrap()
        .with_soft_prompt_embeds(&soft_a, 3, 8)
        .unwrap();
    edge.prefill_step(&mut ctx_g4, &ple_weights)
        .expect("gemma4 PLE soft-prompt prefill succeeds");
    assert_eq!(ctx_g4.position(), 4);
}

struct ParityPayloads {
    embd: Vec<u8>,
    out_norm: Vec<u8>,
    out_head: Vec<u8>,
    layers: Vec<Vec<u8>>,
}

impl ParityPayloads {
    fn new_tiny(num_layers: usize) -> Self {
        let (hidden_dim, ffn_dim, q_dim, kv_dim, head_dim, vocab_size) = (8, 8, 8, 4, 4, 16);
        let mut layers = Vec::new();
        for l in 0..num_layers {
            let s = 20 + l * 16;
            layers.push(encode_bf16_vec(&deterministic_pattern(hidden_dim, s, 0.03)));
            layers.push(encode_f32_vec(&deterministic_pattern(
                hidden_dim * q_dim,
                s + 1,
                0.03,
            )));
            layers.push(encode_f32_vec(&deterministic_pattern(
                hidden_dim * kv_dim,
                s + 2,
                0.03,
            )));
            layers.push(encode_f32_vec(&deterministic_pattern(
                hidden_dim * kv_dim,
                s + 3,
                0.03,
            )));
            layers.push(encode_bf16_vec(&deterministic_pattern(
                head_dim,
                s + 4,
                0.02,
            )));
            layers.push(encode_bf16_vec(&deterministic_pattern(
                head_dim,
                s + 5,
                0.02,
            )));
            layers.push(encode_f32_vec(&deterministic_pattern(
                q_dim * hidden_dim,
                s + 6,
                0.03,
            )));
            layers.push(encode_bf16_vec(&deterministic_pattern(
                hidden_dim,
                s + 7,
                0.02,
            )));
            layers.push(encode_bf16_vec(&deterministic_pattern(
                hidden_dim,
                s + 8,
                0.02,
            )));
            layers.push(encode_f32_vec(&deterministic_pattern(
                hidden_dim * ffn_dim,
                s + 9,
                0.03,
            )));
            layers.push(encode_f32_vec(&deterministic_pattern(
                hidden_dim * ffn_dim,
                s + 10,
                0.03,
            )));
            layers.push(encode_f32_vec(&deterministic_pattern(
                ffn_dim * hidden_dim,
                s + 11,
                0.03,
            )));
            layers.push(encode_bf16_vec(&deterministic_pattern(
                hidden_dim,
                s + 12,
                0.02,
            )));
        }
        Self {
            embd: encode_f32_vec(&deterministic_pattern(hidden_dim * vocab_size, 10, 0.05)),
            out_norm: encode_bf16_vec(&deterministic_pattern(hidden_dim, 11, 0.05)),
            out_head: encode_f32_vec(&deterministic_pattern(hidden_dim * vocab_size, 12, 0.04)),
            layers,
        }
    }

    fn gguf_specs(&self, num_layers: usize) -> Vec<(String, QuantType, Vec<usize>, &[u8])> {
        let mut out = vec![
            (
                "token_embd.weight".to_owned(),
                QuantType::F32,
                vec![8, 16],
                self.embd.as_slice(),
            ),
            (
                "output_norm.weight".to_owned(),
                QuantType::BF16,
                vec![8],
                self.out_norm.as_slice(),
            ),
            (
                "output.weight".to_owned(),
                QuantType::F32,
                vec![8, 16],
                self.out_head.as_slice(),
            ),
        ];
        for l in 0..num_layers {
            let b = l * 13;
            let specs = [
                ("attn_norm.weight", QuantType::BF16, vec![8], b),
                ("attn_q.weight", QuantType::F32, vec![8, 8], b + 1),
                ("attn_k.weight", QuantType::F32, vec![8, 4], b + 2),
                ("attn_v.weight", QuantType::F32, vec![8, 4], b + 3),
                ("attn_q_norm.weight", QuantType::BF16, vec![4], b + 4),
                ("attn_k_norm.weight", QuantType::BF16, vec![4], b + 5),
                ("attn_output.weight", QuantType::F32, vec![8, 8], b + 6),
                (
                    "post_attention_norm.weight",
                    QuantType::BF16,
                    vec![8],
                    b + 7,
                ),
                ("ffn_norm.weight", QuantType::BF16, vec![8], b + 8),
                ("ffn_gate.weight", QuantType::F32, vec![8, 8], b + 9),
                ("ffn_up.weight", QuantType::F32, vec![8, 8], b + 10),
                ("ffn_down.weight", QuantType::F32, vec![8, 8], b + 11),
                ("post_ffw_norm.weight", QuantType::BF16, vec![8], b + 12),
            ];
            for (suffix, q, shape, idx) in specs {
                out.push((
                    format!("blk.{l}.{suffix}"),
                    q,
                    shape,
                    self.layers[idx].as_slice(),
                ));
            }
        }
        out
    }

    fn st_specs(&self, num_layers: usize) -> Vec<(String, QuantType, Vec<usize>, &[u8])> {
        let mut out = vec![
            (
                "language_model.model.embed_tokens.weight".to_owned(),
                QuantType::F32,
                vec![16, 8],
                self.embd.as_slice(),
            ),
            (
                "language_model.model.norm.weight".to_owned(),
                QuantType::BF16,
                vec![8],
                self.out_norm.as_slice(),
            ),
            (
                "language_model.lm_head.weight".to_owned(),
                QuantType::F32,
                vec![16, 8],
                self.out_head.as_slice(),
            ),
        ];
        for l in 0..num_layers {
            let b = l * 13;
            let specs = [
                ("input_layernorm.weight", QuantType::BF16, vec![8], b),
                ("self_attn.q_proj.weight", QuantType::F32, vec![8, 8], b + 1),
                ("self_attn.k_proj.weight", QuantType::F32, vec![4, 8], b + 2),
                ("self_attn.v_proj.weight", QuantType::F32, vec![4, 8], b + 3),
                ("self_attn.q_norm.weight", QuantType::BF16, vec![4], b + 4),
                ("self_attn.k_norm.weight", QuantType::BF16, vec![4], b + 5),
                ("self_attn.o_proj.weight", QuantType::F32, vec![8, 8], b + 6),
                (
                    "post_attention_layernorm.weight",
                    QuantType::BF16,
                    vec![8],
                    b + 7,
                ),
                (
                    "pre_feedforward_layernorm.weight",
                    QuantType::BF16,
                    vec![8],
                    b + 8,
                ),
                ("mlp.gate_proj.weight", QuantType::F32, vec![8, 8], b + 9),
                ("mlp.up_proj.weight", QuantType::F32, vec![8, 8], b + 10),
                ("mlp.down_proj.weight", QuantType::F32, vec![8, 8], b + 11),
                (
                    "post_feedforward_layernorm.weight",
                    QuantType::BF16,
                    vec![8],
                    b + 12,
                ),
            ];
            for (suffix, q, shape, idx) in specs {
                out.push((
                    format!("language_model.model.layers.{l}.{suffix}"),
                    q,
                    shape,
                    self.layers[idx].as_slice(),
                ));
            }
        }
        out
    }
}

#[test]
fn test_gguf_vs_safetensors_bit_identical_parity_gemma3() {
    let payloads = ParityPayloads::new_tiny(2);
    let gguf_kvs = [
        ("gemma3.block_count", GgufValue::Uint32(2)),
        ("gemma3.embedding_length", GgufValue::Uint32(8)),
        ("gemma3.feed_forward_length", GgufValue::Uint32(8)),
        ("gemma3.attention.head_count", GgufValue::Uint32(2)),
        ("gemma3.attention.head_count_kv", GgufValue::Uint32(1)),
        ("gemma3.attention.key_length", GgufValue::Uint32(4)),
        ("gemma3.attention.sliding_window", GgufValue::Uint32(2)),
        ("gemma3.vocab_size", GgufValue::Uint32(16)),
        (
            "gemma3.attention.layer_norm_rms_epsilon",
            GgufValue::Float32(1e-6),
        ),
        ("gemma3.rope.freq_base", GgufValue::Float32(1_000_000.0)),
    ];
    let gguf_specs = payloads.gguf_specs(2);
    let st_specs = payloads.st_specs(2);
    let gguf_refs: Vec<(&str, QuantType, &[usize], &[u8])> = gguf_specs
        .iter()
        .map(|(n, q, s, b)| (n.as_str(), *q, s.as_slice(), *b))
        .collect();
    let st_refs: Vec<(&str, QuantType, &[usize], &[u8])> = st_specs
        .iter()
        .map(|(n, q, s, b)| (n.as_str(), *q, s.as_slice(), *b))
        .collect();

    let gguf_bytes = create_gguf_with_payloads("gemma3", 32, &gguf_kvs, &gguf_refs);
    let st_bytes = create_synthetic_safetensors(
        &[
            ("general.architecture", "gemma3"),
            ("num_hidden_layers", "2"),
            ("hidden_size", "8"),
            ("intermediate_size", "8"),
            ("num_attention_heads", "2"),
            ("num_key_value_heads", "1"),
            ("head_dim", "4"),
            ("sliding_window", "2"),
            ("vocab_size", "16"),
            ("rms_norm_eps", "1e-6"),
            ("rope_theta", "1000000.0"),
        ],
        &st_refs,
    );

    let gguf_file = GgufFile::parse(&gguf_bytes).expect("parse gguf");
    let st_file = SafeTensorsFile::from_bytes(&st_bytes).expect("parse safetensors");
    let recipe_gguf = GemmaRecipe::from_gguf(&gguf_file).expect("hydrate gemma3 gguf");
    let recipe_st = GemmaRecipe::from_safetensors(&st_file).expect("hydrate gemma3 safetensors");
    assert_eq!(recipe_gguf.sliding_window, recipe_st.sliding_window);

    let soft = deterministic_pattern(16, 55, 0.2);
    let mut logits_gguf = vec![0.0_f32; 16];
    let mut kv_gguf = vec![0.0_f32; 64];
    let mut ctx_gguf = ExecutionContext::new(&[1, 3, 7], 8, &mut logits_gguf, &mut kv_gguf)
        .unwrap()
        .with_soft_prompt_embeds(&soft, 2, 8)
        .unwrap();
    recipe_gguf.prefill_step(&mut ctx_gguf, &gguf_file).unwrap();
    ctx_gguf.prepare_decode(9).unwrap();
    recipe_gguf.decode_step(&mut ctx_gguf, &gguf_file).unwrap();

    let mut logits_st = vec![0.0_f32; 16];
    let mut kv_st = vec![0.0_f32; 64];
    let mut ctx_st = ExecutionContext::new(&[1, 3, 7], 8, &mut logits_st, &mut kv_st)
        .unwrap()
        .with_soft_prompt_embeds(&soft, 2, 8)
        .unwrap();
    recipe_st.prefill_step(&mut ctx_st, &st_file).unwrap();
    ctx_st.prepare_decode(9).unwrap();
    recipe_st.decode_step(&mut ctx_st, &st_file).unwrap();

    assert_logits_parity(&logits_gguf, &logits_st, "GGUF vs SafeTensors parity");
}

#[test]
fn test_safetensors_hydration_gemma2_and_gemma4() {
    let payloads = ParityPayloads::new_tiny(2);
    let st_specs = payloads.st_specs(2);
    let st_refs: Vec<(&str, QuantType, &[usize], &[u8])> = st_specs
        .iter()
        .map(|(n, q, s, b)| (n.as_str(), *q, s.as_slice(), *b))
        .collect();

    let g2_st_bytes = create_synthetic_safetensors(
        &[
            ("model_type", "gemma2"),
            ("num_hidden_layers", "2"),
            ("hidden_size", "8"),
            ("intermediate_size", "8"),
            ("num_attention_heads", "2"),
            ("num_key_value_heads", "1"),
            ("head_dim", "4"),
            ("sliding_window", "2"),
            ("attn_logit_softcapping", "50.0"),
            ("final_logit_softcapping", "30.0"),
        ],
        &st_refs,
    );
    let g2_st = SafeTensorsFile::from_bytes(&g2_st_bytes).expect("gemma2 safetensors");
    let g2_recipe = GemmaRecipe::from_safetensors(&g2_st).expect("gemma2 recipe");
    assert_eq!(g2_recipe.attn_logit_softcap, Some(50.0));
    assert_eq!(g2_recipe.final_logit_softcap, Some(30.0));

    let g4_st_bytes = create_synthetic_safetensors(
        &[
            ("general.architecture", "gemma4"),
            ("num_hidden_layers", "2"),
            ("hidden_size", "8"),
            ("intermediate_size", "8"),
            ("num_attention_heads", "2"),
            ("num_key_value_heads", "1"),
            ("head_dim", "4"),
            ("sliding_window", "2"),
            ("vocab_size", "16"),
        ],
        &st_refs,
    );
    let g4_st = SafeTensorsFile::from_bytes(&g4_st_bytes).expect("gemma4 safetensors");
    let g4_recipe = Gemma4Recipe::from_safetensors(&g4_st).expect("gemma4 recipe");
    let mut logits_g4 = vec![0.0_f32; 16];
    let mut kv_g4 = vec![0.0_f32; 64];
    let mut ctx_g4 = ExecutionContext::new(&[1, 3], 8, &mut logits_g4, &mut kv_g4).unwrap();
    g4_recipe
        .prefill_step(&mut ctx_g4, &g4_st)
        .expect("gemma4 safetensors prefill succeeds");
    assert!(logits_g4.iter().all(|v| v.is_finite()));
}

#[test]
fn test_validate_model_enforces_adr045() {
    let g3 = tiny_gemma3_recipe(1);

    let mut weights = MapWeights::new();
    populate_dense_gemma_weights(&mut weights, DenseWeightSpec::tiny(1));
    let empty_gguf_bytes = create_gguf_with_payloads("gemma3", 32, &[], &[]);
    let empty_meta = GgufFile::parse(&empty_gguf_bytes).expect("parse empty metadata");
    g3.validate_model(empty_meta.metadata(), &weights)
        .expect("complete resident weights pass validate_model");

    weights.remove("blk.0.ffn_down.weight");
    assert!(matches!(
        g3.validate_model(empty_meta.metadata(), &weights),
        Err(K3Error::TensorBoundsError { .. })
    ));
}

#[test]
fn test_matvec_any_gguf_and_safetensors_sequential() {
    let rows = 12;
    let cols = 8;
    let vals = deterministic_pattern(rows * cols, 5, 0.1);
    let raw = encode_f32_vec(&vals);
    let input = deterministic_pattern(cols, 9, 0.2);

    let gguf_shape = [cols, rows];
    let st_shape = [rows, cols];
    let gguf_view =
        TensorView::from_bytes("gguf_proj", QuantType::F32, &gguf_shape, 0, &raw).unwrap();
    let st_view = TensorView::from_bytes("st_proj", QuantType::F32, &st_shape, 0, &raw).unwrap();

    let mut gguf_out = vec![0.0_f32; rows];
    let mut st_out = vec![0.0_f32; rows];
    matvec_any(&mut gguf_out, &gguf_view, &input).expect("sequential gguf matvec");
    matvec_any(&mut st_out, &st_view, &input).expect("sequential st matvec");

    for i in 0..rows {
        assert_eq!(gguf_out[i].to_bits(), st_out[i].to_bits());
    }
}

#[cfg(not(miri))]
#[test]
fn test_matvec_any_rayon_parallel_gguf_and_safetensors_parity() {
    let rows = 96;
    let cols = 16;
    let vals = deterministic_pattern(rows * cols, 5, 0.1);
    let raw = encode_f32_vec(&vals);
    let input = deterministic_pattern(cols, 9, 0.2);

    let gguf_shape = [cols, rows];
    let st_shape = [rows, cols];
    let gguf_view =
        TensorView::from_bytes("gguf_proj", QuantType::F32, &gguf_shape, 0, &raw).unwrap();
    let st_view = TensorView::from_bytes("st_proj", QuantType::F32, &st_shape, 0, &raw).unwrap();

    let mut seq_out = vec![0.0_f32; rows];
    matvec_any(&mut seq_out, &gguf_view, &input).expect("sequential matvec succeeds");

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("rayon pool");
    let mut par_gguf_out = vec![0.0_f32; rows];
    let mut par_st_out = vec![0.0_f32; rows];
    pool.install(|| {
        matvec_any(&mut par_gguf_out, &gguf_view, &input).expect("parallel gguf matvec");
        matvec_any(&mut par_st_out, &st_view, &input).expect("parallel st matvec");
    });

    for i in 0..rows {
        assert_eq!(seq_out[i].to_bits(), par_gguf_out[i].to_bits());
        assert_eq!(seq_out[i].to_bits(), par_st_out[i].to_bits());
    }
}
