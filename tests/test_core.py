import math

import torch

from samatnext_bit.bitlinear import BitLinear, base3_lut5_table, pack_ternary_2bit, pack_ternary_base3, ternarize_shadow, unpack_ternary_2bit, unpack_ternary_base3
from samatnext_bit.data import ByteBatcher
from samatnext_bit.model import DecoderLM
from samatnext_bit.train import train_mode


def test_batcher_shapes_cpu():
    b = ByteBatcher("tiny_code", 2, 16, torch.device("cpu"))
    x, y = b.next_batch()
    assert x.shape == y.shape == (2, 16)
    assert x.max() < 256


def test_model_forward_cpu():
    m = DecoderLM(256, seq_len=16, hidden=32, layers=2, heads=4)
    x = torch.randint(0, 256, (2, 16))
    logits, loss = m(x, x)
    assert logits.shape == (2, 16, 256)
    assert loss.item() > 0


def test_bitlinear_fake_backward():
    m = BitLinear(8, 4, backend="fake_ternary")
    x = torch.randn(3, 8, requires_grad=True)
    y = m(x).sum()
    y.backward()
    assert m.weight.grad is not None
    assert x.grad is not None


def test_ternary_active_values_only():
    m = BitLinear(17, 3, backend="fake_ternary")
    _, ternary, _ = ternarize_shadow(m.weight)
    assert set(ternary.unique().tolist()).issubset({-1.0, 0.0, 1.0})


def test_pack_ternary_uses_int32_words_and_16_weights_per_word():
    ternary = torch.tensor([[0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1]], dtype=torch.float32)
    packed = pack_ternary_2bit(ternary)
    assert packed.dtype == torch.int32
    assert packed.shape == (1, math.ceil(ternary.shape[1] / 16))
    unpacked = unpack_ternary_2bit(packed, ternary.shape[1])
    assert torch.equal(unpacked, ternary)


def test_base3_pack_unpack_roundtrip_and_storage():
    ternary = torch.tensor([[-1, 0, 1, 1, 0, -1, 1, 0, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1]], dtype=torch.float32)
    packed = pack_ternary_base3(ternary)
    assert packed.dtype == torch.int32
    assert packed.shape == (1, math.ceil(ternary.shape[1] / 20))
    assert packed.numel() * 32 / (packed.shape[0] * packed.shape[1] * 20) == 1.6
    unpacked = unpack_ternary_base3(packed, ternary.shape[1])
    assert torch.equal(unpacked, ternary)


def test_base3_lut_decode_roundtrip():
    table = base3_lut5_table().view(243, 5)
    for chunk in [0, 1, 2, 3, 17, 80, 121, 242]:
        packed = torch.tensor([[chunk]], dtype=torch.int32)
        expected = unpack_ternary_base3(packed, 5)[0]
        assert torch.equal(table[chunk], expected)


def test_base3_forward_matches_fake_on_small_cuda_if_available():
    if not torch.cuda.is_available():
        return
    fake = BitLinear(31, 7, backend="fake_ternary").cuda()
    packed = BitLinear(31, 7, backend="base3_packed_ternary").cuda()
    packed.weight.data.copy_(fake.weight.data)
    packed.bias.data.copy_(fake.bias.data)
    packed.refresh_packed_cache()
    x = torch.randn(5, 31, device="cuda")
    y_fake = fake(x)
    y_packed = packed(x)
    torch.testing.assert_close(y_packed, y_fake, rtol=1e-3, atol=1e-3)
    assert packed.packed_weight().dtype == torch.int32
    assert packed.packed_weight().dtype != torch.int8
    assert packed.base3_packed
    assert not packed.two_bit_packed
    assert not packed.int8_ternary


def test_base3_lut_forward_matches_fake_on_small_cuda_if_available():
    if not torch.cuda.is_available():
        return
    fake = BitLinear(31, 7, backend="fake_ternary").cuda()
    packed = BitLinear(31, 7, backend="base3_lut_ternary").cuda()
    packed.weight.data.copy_(fake.weight.data)
    packed.bias.data.copy_(fake.bias.data)
    packed.refresh_packed_cache()
    x = torch.randn(5, 31, device="cuda")
    y_fake = fake(x)
    y_packed = packed(x)
    torch.testing.assert_close(y_packed, y_fake, rtol=1e-3, atol=1e-3)
    assert packed.decode_strategy == "lut"
    assert packed.base3_packed
    assert packed.storage_bits_per_weight == 1.6


def test_base3_tile_dot_forward_matches_fake_on_supported_cuda_if_available():
    if not torch.cuda.is_available():
        return
    fake = BitLinear(128, 32, backend="fake_ternary").cuda()
    packed = BitLinear(128, 32, backend="base3_tile_dot_ternary").cuda()
    packed.weight.data.copy_(fake.weight.data)
    packed.bias.data.copy_(fake.bias.data)
    packed.refresh_packed_cache()
    x = torch.randn(9, 128, device="cuda")
    y_fake = fake(x)
    y_packed = packed(x)
    torch.testing.assert_close(y_packed, y_fake, rtol=1e-3, atol=1e-3)
    assert packed.decode_strategy == "tile_dequant_dot"
    assert packed.base3_packed
    assert packed.storage_bits_per_weight == 1.6
    assert packed.uses_tensor_cores
    assert not packed.fallback_used


def test_base3_tile_dot_no_decoded_global_weight_buffer():
    layer = BitLinear(128, 32, backend="base3_tile_dot_ternary")
    buffer_names = {name for name, _ in layer.named_buffers()}
    assert "packed_cache" in buffer_names
    assert "gamma_cache" in buffer_names
    assert all("decoded" not in name for name in buffer_names)


def test_base3_tile_dot_unsupported_shape_fails_cuda_if_available():
    if not torch.cuda.is_available():
        return
    layer = BitLinear(129, 8, backend="base3_tile_dot_ternary").cuda()
    layer.refresh_packed_cache()
    x = torch.randn(2, 129, device="cuda")
    try:
        _ = layer(x)
    except RuntimeError as exc:
        assert "requires K divisible by 128" in str(exc)
    else:
        raise AssertionError("unsupported K should fail clearly")


def test_base3_persistent_cache_reused_and_refresh_explicit_cuda_if_available():
    if not torch.cuda.is_available():
        return
    layer = BitLinear(20, 4, backend="base3_packed_ternary").cuda()
    x = torch.randn(2, 20, device="cuda")
    layer.refresh_packed_cache()
    ptr = layer.packed_cache.data_ptr()
    count = layer.packed_cache_refresh_count
    _ = layer(x)
    _ = layer(x)
    assert layer.packed_cache.data_ptr() == ptr
    assert layer.packed_cache_refresh_count == count
    with torch.no_grad():
        layer.weight.add_(0.01)
    layer.mark_packed_cache_dirty()
    layer.refresh_packed_cache()
    assert layer.packed_cache.data_ptr() == ptr
    assert layer.packed_cache_refresh_count == count + 1


def test_base3_lut_train_one_step_cpu():
    cfg = {"dataset": "tiny_code", "vocab_size": 256, "hidden": 32, "layers": 1, "heads": 4, "batch_size": 2, "seq_len": 16, "steps": 1, "lr": 1e-3}
    r = train_mode(cfg, "bitnet_base3_lut_ternary_mono_update_every_2", torch.device("cpu"))
    assert r.completed
    assert r.backend == "base3_lut_ternary"
    assert r.decode_strategy == "lut"


def test_base3_tile_dot_train_one_step_cuda_if_available():
    if not torch.cuda.is_available():
        return
    cfg = {"dataset": "tiny_code", "vocab_size": 256, "hidden": 128, "layers": 1, "heads": 4, "batch_size": 2, "seq_len": 16, "steps": 1, "lr": 1e-3}
    r = train_mode(cfg, "bitnet_base3_tile_dot_ternary_mono_update_every_2", torch.device("cuda"))
    assert r.completed
    assert r.backend == "base3_tile_dot_ternary"
    assert r.decode_strategy == "tile_dequant_dot"
    assert r.storage_bits_per_weight == 1.6
    assert not r.fallback_used


def test_train_one_step_cpu():
    cfg = {"dataset": "tiny_code", "vocab_size": 256, "hidden": 32, "layers": 1, "heads": 4, "batch_size": 2, "seq_len": 16, "steps": 1, "lr": 1e-3}
    r = train_mode(cfg, "bitnet_fake_ternary_mono_update_every_2", torch.device("cpu"))
    assert r.completed
    assert r.params > 0


def test_benchmark_metadata_for_packed_cpu_fallback():
    cfg = {"dataset": "tiny_code", "vocab_size": 256, "hidden": 32, "layers": 1, "heads": 4, "batch_size": 2, "seq_len": 16, "steps": 1, "lr": 1e-3}
    r = train_mode(cfg, "bitnet_base3_packed_ternary_mono_update_every_2", torch.device("cpu"))
    assert r.backend == "base3_packed_ternary"
    assert r.packed_ternary
    assert r.base3_packed
    assert not r.two_bit_packed
    assert not r.int8_ternary
    assert r.storage_bits_per_weight == 1.6
    assert r.ideal_entropy_bits_per_weight == 1.585
    assert not r.packed_1p58bit
    assert r.ste_backward

    r_lut = train_mode(cfg, "bitnet_base3_lut_ternary_mono_update_every_2", torch.device("cpu"))
    assert r_lut.backend == "base3_lut_ternary"
    assert r_lut.decode_strategy == "lut"
    assert r_lut.persistent_packed_cache
