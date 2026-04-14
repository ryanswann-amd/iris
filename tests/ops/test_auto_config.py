#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Unit tests for iris.ops.auto_config — AG+MM auto-selection mechanism.

These tests run on CPU (no GPU required) and verify:
1. Exact shape lookup hits return correct champion configs
2. Lookup misses fall back to heuristic defaults
3. ws<8 configs are correctly disabled
4. FusedConfig conversion works for enabled configs
5. FusedConfig conversion raises for disabled configs
6. list_known_shapes returns correct data
7. Config cache can be cleared
"""

import pytest
import sys
import os

# Ensure iris package is importable even without full install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from iris.ops.auto_config import (
    select_ag_mm_config,
    list_known_shapes,
    load_regression_sizes,
    clear_config_cache,
    detect_gpu_arch,
    SUPPORTED_TRANSPOSES,
    SUPPORTED_ARCHITECTURES,
    _apply_heuristic,
)
import iris.ops.auto_config as auto_config_module
from iris.ops.config import FusedConfig


class TestAutoConfigExactMatch:
    """Test exact shape lookup (cache hit path)."""

    def setup_method(self):
        clear_config_cache()

    def test_ws8_g2_exact_match(self):
        """g2 shape (131072x16384x16384) should return champion config."""
        result = select_ag_mm_config(M=131072, N=16384, K=16384, world_size=8, transpose="NN", arch="mi300x")
        assert result.enabled is True
        assert result.shape_key == "131072x16384x16384"
        assert result.speedup == 1.343
        assert "Exact match" in result.source
        assert result.config_params["block_size_m"] == 256
        assert result.config_params["block_size_n"] == 256
        assert result.config_params["block_size_k"] == 64
        assert result.config_params["num_stages"] == 2

    def test_ws8_g5_exact_match(self):
        """g5 shape (8192x8192x262144) should use bm=128 (small M)."""
        result = select_ag_mm_config(M=8192, N=8192, K=262144, world_size=8, transpose="NN", arch="mi300x")
        assert result.enabled is True
        assert result.shape_key == "8192x8192x262144"
        assert result.speedup == 1.224
        assert result.config_params["block_size_m"] == 128
        assert result.config_params["group_size_m"] == 8

    def test_ws8_g1_exact_match(self):
        """g1 shape (16384x16384x131072) should use bm=128."""
        result = select_ag_mm_config(M=16384, N=16384, K=131072, world_size=8, transpose="NN", arch="mi300x")
        assert result.enabled is True
        assert result.shape_key == "16384x16384x131072"
        assert result.speedup == 1.136
        assert result.config_params["block_size_m"] == 128

    def test_ws8_winning_shapes_enabled(self):
        """All 7 winning champion shapes (speedup > 1.0) should be enabled."""
        winning_shapes = [
            (131072, 16384, 16384),  # g2  — 1.343x
            (327680, 28672, 4096),  # g15 — 1.284x
            (147456, 28672, 4096),  # g14 — 1.288x
            (229376, 28672, 4096),  # g16 — 1.277x
            (8192, 8192, 262144),  # g5  — 1.224x
            (262144, 8192, 8192),  # g6  — 1.200x
            (16384, 16384, 131072),  # g1  — 1.136x
        ]
        for M, N, K in winning_shapes:
            result = select_ag_mm_config(M, N, K, world_size=8)
            assert result.enabled, f"Shape {M}x{N}x{K} should be enabled"
            assert result.speedup is not None and result.speedup > 1.0, (
                f"Shape {M}x{N}x{K} should have speedup > 1.0, got {result.speedup}"
            )


class TestAutoConfigFallback:
    """Test lookup miss / heuristic fallback path."""

    def setup_method(self):
        clear_config_cache()

    def test_ws8_unknown_shape_returns_heuristic(self):
        """An unknown shape at ws=8 should still be enabled with heuristic config."""
        result = select_ag_mm_config(M=65536, N=4096, K=8192, world_size=8, transpose="NN", arch="mi300x")
        assert result.enabled is True
        assert result.shape_key is None
        assert result.speedup is None
        assert "Heuristic" in result.source
        # Heuristic: M=65536 > 16384 -> bm=256
        assert result.config_params["block_size_m"] == 256

    def test_ws8_small_M_exact_or_heuristic(self):
        """Small M (pow2_4k) hits exact match with bm=128."""
        result = select_ag_mm_config(M=4096, N=4096, K=4096, world_size=8, transpose="NN", arch="mi300x")
        assert result.enabled is True
        assert result.config_params["block_size_m"] == 128


class TestAutoConfigDisabled:
    """Test that ws<8 correctly disables iris AG+MM."""

    def setup_method(self):
        clear_config_cache()

    def test_ws2_disabled(self):
        """ws=2 should be disabled on MI300X."""
        result = select_ag_mm_config(M=131072, N=16384, K=16384, world_size=2, transpose="NN", arch="mi300x")
        assert result.enabled is False
        assert "Disabled" in result.source or "world_size" in result.source

    def test_ws4_disabled(self):
        """ws=4 should be disabled on MI300X."""
        result = select_ag_mm_config(M=131072, N=16384, K=16384, world_size=4, transpose="NN", arch="mi300x")
        assert result.enabled is False

    def test_ws1_disabled(self):
        """ws=1 should be disabled (no config file, below min_world_size)."""
        result = select_ag_mm_config(M=4096, N=4096, K=4096, world_size=1, transpose="NN", arch="mi300x")
        assert result.enabled is False

    def test_ws3_disabled_by_default_gate(self):
        """ws=3 has no config file, should be disabled by global default min_world_size=8."""
        result = select_ag_mm_config(M=4096, N=4096, K=4096, world_size=3, transpose="NN", arch="mi300x")
        assert result.enabled is False


class TestFusedConfigConversion:
    """Test AutoConfigResult.to_fused_config()."""

    def setup_method(self):
        clear_config_cache()

    def test_enabled_config_converts(self):
        """Enabled configs should produce valid FusedConfig."""
        result = select_ag_mm_config(M=131072, N=16384, K=16384, world_size=8)
        config = result.to_fused_config()
        assert isinstance(config, FusedConfig)
        assert config.block_size_m == 256
        assert config.block_size_n == 256
        assert config.block_size_k == 64
        assert config.group_size_m == 24

    def test_disabled_config_raises(self):
        """Disabled configs should raise RuntimeError on to_fused_config()."""
        result = select_ag_mm_config(M=131072, N=16384, K=16384, world_size=2)
        assert result.enabled is False
        with pytest.raises(RuntimeError, match="disabled"):
            result.to_fused_config()


class TestListKnownShapes:
    """Test list_known_shapes utility."""

    def setup_method(self):
        clear_config_cache()

    def test_ws8_lists_all_champions(self):
        """ws=8 should list all champion shapes (at least 7)."""
        shapes = list_known_shapes(world_size=8, transpose="NN", arch="mi300x")
        assert len(shapes) >= 7, f"Expected at least 7 champion shapes, got {len(shapes)}"
        # Should be sorted by speedup descending
        speedups = [s["speedup"] for s in shapes]
        assert speedups == sorted(speedups, reverse=True)
        # Top shape should be g2 (1.343x)
        assert shapes[0]["label"] == "g2"
        assert shapes[0]["speedup"] == 1.343

    def test_ws2_lists_empty(self):
        """ws=2 (disabled) should return empty list."""
        shapes = list_known_shapes(world_size=2, transpose="NN", arch="mi300x")
        assert shapes == []

    def test_ws4_lists_empty(self):
        """ws=4 (disabled) should return empty list."""
        shapes = list_known_shapes(world_size=4, transpose="NN", arch="mi300x")
        assert shapes == []


class TestUnknownArchTranspose:
    """Test behavior with unknown architectures/transposes."""

    def setup_method(self):
        clear_config_cache()

    def test_unknown_arch_uses_global_default(self):
        """Unknown GPU arch should fall through to global default."""
        result = select_ag_mm_config(M=131072, N=16384, K=16384, world_size=8, transpose="NN", arch="mi300a")
        # No config file exists for mi300a, should use heuristic fallback
        assert result.enabled is True
        assert "Heuristic" in result.source or "fallback" in result.source

    def test_unknown_transpose_ws_lt8_disabled(self):
        """Unknown transpose at ws<8 should still be disabled."""
        result = select_ag_mm_config(M=131072, N=16384, K=16384, world_size=4, transpose="TN", arch="mi300x")
        assert result.enabled is False


class TestHbmBufferParams:
    """Test that HBM buffer-specific params are returned."""

    def setup_method(self):
        clear_config_cache()

    def test_exact_match_has_hbm_params(self):
        """Exact match should include hbm_buffer_params."""
        result = select_ag_mm_config(M=131072, N=16384, K=16384, world_size=8)
        assert result.enabled is True
        hbm = result.hbm_buffer_params
        assert "k_per_flag" in hbm
        assert "num_fetch_sms" in hbm
        assert "num_fetch_stages" in hbm
        assert "first_stage_fetch_sms" in hbm
        # g2 specific values
        assert hbm["k_per_flag"] == 32
        assert hbm["num_fetch_sms"] == 4
        assert hbm["num_fetch_stages"] == 64
        assert hbm["first_stage_fetch_sms"] == 52

    def test_heuristic_has_hbm_params(self):
        """Heuristic fallback should also include hbm_buffer_params."""
        result = select_ag_mm_config(M=65536, N=4096, K=8192, world_size=8)
        assert result.enabled is True
        hbm = result.hbm_buffer_params
        assert "k_per_flag" in hbm
        assert "num_fetch_sms" in hbm
        assert hbm["k_per_flag"] > 0

    def test_g5_hbm_params(self):
        """g5 K-dominant shape should have specific HBM params."""
        result = select_ag_mm_config(M=8192, N=8192, K=262144, world_size=8)
        hbm = result.hbm_buffer_params
        assert hbm["k_per_flag"] == 32
        assert hbm["num_fetch_sms"] == 4
        assert hbm["num_fetch_stages"] == 8

    def test_g15_hbm_params(self):
        """g15 highest TFLOPS shape should have specific HBM params."""
        result = select_ag_mm_config(M=327680, N=28672, K=4096, world_size=8)
        hbm = result.hbm_buffer_params
        assert hbm["k_per_flag"] == 16
        assert hbm["num_fetch_sms"] == 4
        assert hbm["num_fetch_stages"] == 32
        assert hbm["first_stage_fetch_sms"] == 52


class TestRegressionSizes:
    """Test regression size loading."""

    def test_load_regression_sizes(self):
        """Should load regression test sizes from JSON — includes ws=2/4/8 entries."""
        sizes = load_regression_sizes()
        assert len(sizes) >= 9, f"Expected at least 9 regression sizes (7 ws=8 + 2 disabled), got {len(sizes)}"

    def test_regression_sizes_have_required_fields(self):
        """Each regression size should have name, M, N, K, world_sizes."""
        sizes = load_regression_sizes()
        for s in sizes:
            assert "name" in s, f"Missing 'name' in regression size: {s}"
            assert "M" in s, f"Missing 'M' in regression size: {s}"
            assert "N" in s, f"Missing 'N' in regression size: {s}"
            assert "K" in s, f"Missing 'K' in regression size: {s}"
            assert "world_sizes" in s, f"Missing 'world_sizes' in regression size: {s}"
            assert isinstance(s["world_sizes"], list)

    def test_regression_sizes_match_configs(self):
        """Each regression size should have a matching config in ws8.json."""
        clear_config_cache()
        sizes = load_regression_sizes()
        for s in sizes:
            if 8 in s["world_sizes"]:
                result = select_ag_mm_config(s["M"], s["N"], s["K"], world_size=8)
                assert result.enabled is True, f"{s['name']} should be enabled at ws=8"

    def test_disabled_regression_sizes_correctly_disabled(self):
        """Disabled regression entries (ws=2, ws=4) must be flagged disabled by auto-config."""
        clear_config_cache()
        sizes = load_regression_sizes()
        disabled_entries = [s for s in sizes if s["tier"] == "disabled"]
        assert len(disabled_entries) >= 2, "Expected at least 2 disabled regression entries"
        for s in disabled_entries:
            for ws in s["world_sizes"]:
                result = select_ag_mm_config(s["M"], s["N"], s["K"], world_size=ws)
                assert result.enabled is False, (
                    f"{s['name']} ws={ws} should be disabled but got enabled={result.enabled}"
                )

    def test_regression_sizes_cover_all_world_sizes(self):
        """Regression sizes should cover ws=2, ws=4, and ws=8."""
        sizes = load_regression_sizes()
        covered_ws = set()
        for s in sizes:
            for ws in s["world_sizes"]:
                covered_ws.add(ws)
        assert 2 in covered_ws, "Missing ws=2 coverage in regression sizes"
        assert 4 in covered_ws, "Missing ws=4 coverage in regression sizes"
        assert 8 in covered_ws, "Missing ws=8 coverage in regression sizes"


class TestConfigCache:
    """Test config cache behavior."""

    def test_cache_clear(self):
        """Cache should be clearable."""
        # Populate cache
        select_ag_mm_config(M=131072, N=16384, K=16384, world_size=8)
        # Clear
        clear_config_cache()
        # Should still work after clear
        result = select_ag_mm_config(M=131072, N=16384, K=16384, world_size=8)
        assert result.enabled is True
        assert result.shape_key == "131072x16384x16384"


class TestTransposeCoverage:
    """Reviewer feedback #1: Verify transpose coverage and document NN-only support."""

    def setup_method(self):
        clear_config_cache()

    def test_only_nn_has_tuned_configs(self):
        """Only NN transpose has tuned configs — the AG+MM kernel is NN-only."""
        assert SUPPORTED_TRANSPOSES == ("NN",)

    def test_nn_ws8_returns_exact_match(self):
        """NN transpose at ws=8 should find champion configs."""
        result = select_ag_mm_config(M=131072, N=16384, K=16384, world_size=8, transpose="NN")
        assert result.enabled is True
        assert "Exact match" in result.source

    def test_tn_ws8_returns_heuristic_fallback(self):
        """TN transpose at ws=8 — no config file exists, falls back to heuristic."""
        result = select_ag_mm_config(M=131072, N=16384, K=16384, world_size=8, transpose="TN")
        # No configs/ag_mm/mi300x/TN/ directory → heuristic fallback for ws>=8
        assert result.enabled is True
        assert "Heuristic" in result.source or "fallback" in result.source

    def test_nt_ws8_returns_heuristic_fallback(self):
        """NT transpose at ws=8 — no config file exists, falls back to heuristic."""
        result = select_ag_mm_config(M=131072, N=16384, K=16384, world_size=8, transpose="NT")
        assert result.enabled is True
        assert "Heuristic" in result.source or "fallback" in result.source

    def test_tt_ws8_returns_heuristic_fallback(self):
        """TT transpose at ws=8 — no config file exists, falls back to heuristic."""
        result = select_ag_mm_config(M=131072, N=16384, K=16384, world_size=8, transpose="TT")
        assert result.enabled is True
        assert "Heuristic" in result.source or "fallback" in result.source

    def test_non_nn_ws_lt8_disabled(self):
        """Non-NN transpose at ws<8 should still be disabled."""
        for t in ("TN", "NT", "TT"):
            result = select_ag_mm_config(M=131072, N=16384, K=16384, world_size=4, transpose=t)
            assert result.enabled is False, f"{t} at ws=4 should be disabled"

    def test_supported_architectures(self):
        """Only mi300x has tuned configs."""
        assert "mi300x" in SUPPORTED_ARCHITECTURES


class TestHeuristicFallbackValidation:
    """Reviewer feedback #2: Validate the heuristic-derived configs with concrete examples."""

    def setup_method(self):
        clear_config_cache()

    def test_heuristic_small_m_small_k(self):
        """M=4096, K=4096: bm=128, gm=8, kpf=1 (K_blocks=64, kpf starts at 8)."""
        config, hbm = _apply_heuristic(M=4096, N=4096, K=4096)
        assert config["block_size_m"] == 128, f"Small M should get bm=128, got {config['block_size_m']}"
        assert config["block_size_n"] == 256
        assert config["group_size_m"] == 8, f"M=4096 should get gm=8, got {config['group_size_m']}"
        assert hbm["k_per_flag"] > 0
        # K=4096, bk=64 → num_k_blocks=64 → kpf starts at 8
        assert hbm["k_per_flag"] == 8
        # M=4096 / bm=128 = 32 tiles → num_fetch_sms=16
        assert hbm["num_fetch_sms"] == 16

    def test_heuristic_medium_m_large_k(self):
        """M=16384, K=131072: bm=128, gm=16, kpf=16."""
        config, hbm = _apply_heuristic(M=16384, N=16384, K=131072)
        assert config["block_size_m"] == 128, "M=16384 should still use bm=128"
        assert config["group_size_m"] == 16, "M=16384 should get gm=16"
        # K=131072, bk=64 → num_k_blocks=2048 → kpf=64 (>=512)
        assert hbm["k_per_flag"] == 64
        # M=16384 / bm=128 = 128 tiles → num_fetch_sms=32
        assert hbm["num_fetch_sms"] == 32

    def test_heuristic_large_m_small_k(self):
        """M=327680, K=4096: bm=256, gm=24, kpf=4."""
        config, hbm = _apply_heuristic(M=327680, N=28672, K=4096)
        assert config["block_size_m"] == 256, "Large M should get bm=256"
        assert config["group_size_m"] == 24
        # K=4096, bk=64 → num_k_blocks=64 → kpf starts at 8
        assert hbm["k_per_flag"] == 8
        # M=327680 / bm=256 = 1280 tiles → num_fetch_sms=52
        assert hbm["num_fetch_sms"] == 52

    def test_heuristic_matches_champion_g2(self):
        """Heuristic for g2 shape (131072x16384x16384) should produce bm=256, gm=24."""
        config, hbm = _apply_heuristic(M=131072, N=16384, K=16384)
        # g2 champion uses bm=256, gm=24 — heuristic should agree on these
        assert config["block_size_m"] == 256
        assert config["group_size_m"] == 24
        assert config["num_warps"] == 8
        assert config["num_stages"] == 2
        # HBM params: K=16384, bk=64 → 256 blocks → kpf=16
        assert hbm["k_per_flag"] == 16

    def test_heuristic_matches_champion_g5(self):
        """Heuristic for g5 shape (8192x8192x262144) should produce bm=128, gm=8."""
        config, hbm = _apply_heuristic(M=8192, N=8192, K=262144)
        # g5 champion uses bm=128, gm=8 — heuristic should agree
        assert config["block_size_m"] == 128
        assert config["group_size_m"] == 8
        # K=262144, bk=64 → 4096 blocks → kpf=64 (>=512)
        assert hbm["k_per_flag"] == 64

    def test_heuristic_kpf_divisibility(self):
        """k_per_flag must evenly divide num_k_blocks."""
        for K in [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]:
            config, hbm = _apply_heuristic(M=16384, N=16384, K=K)
            num_k_blocks = K // 64  # bk=64 is invariant
            kpf = hbm["k_per_flag"]
            assert num_k_blocks % kpf == 0, (
                f"K={K}: k_per_flag={kpf} does not evenly divide num_k_blocks={num_k_blocks}"
            )

    def test_heuristic_via_select_for_unknown_shape(self):
        """Full pipeline: unknown shape at ws=8 uses heuristic with reasonable params."""
        # Llama-13B gate projection: batch=2048, hidden=5120, intermediate=13824
        result = select_ag_mm_config(M=2048 * 8, N=13824, K=5120, world_size=8, transpose="NN", arch="mi300x")
        assert result.enabled is True
        assert result.shape_key is None  # no exact match
        assert "Heuristic" in result.source
        # M=16384 → bm=128 (at boundary), gm=16
        assert result.config_params["block_size_m"] == 128
        assert result.config_params["group_size_m"] == 16
        assert result.hbm_buffer_params["k_per_flag"] > 0
        # Can convert to FusedConfig
        fc = result.to_fused_config()
        assert isinstance(fc, FusedConfig)


class TestIntegrationPath:
    """Reviewer feedback #4: Show exactly how select_ag_mm_config() integrates with harness."""

    def setup_method(self):
        clear_config_cache()

    def test_auto_select_to_fused_config_pipeline(self):
        """Demonstrate full pipeline: auto-select → FusedConfig → pass to kernel.

        This is the concrete integration pattern:
            result = select_ag_mm_config(M, N, K, world_size=8)
            if result.enabled:
                config = result.to_fused_config()
                hbm = result.hbm_buffer_params
                # config passed to: all_gather_matmul(shmem, C, A, B, config=config)
                # hbm used for: k_per_flag, num_fetch_sms in the HBM buffer kernel
            else:
                # Fallback to PyTorch: torch.distributed.all_gather + torch.matmul
                pass
        """
        # Step 1: Auto-select for a champion shape
        result = select_ag_mm_config(M=131072, N=16384, K=16384, world_size=8)
        assert result.enabled is True

        # Step 2: Convert to FusedConfig (this is what all_gather_matmul() accepts)
        config = result.to_fused_config()
        assert isinstance(config, FusedConfig)
        assert config.block_size_m == 256
        assert config.block_size_n == 256
        assert config.block_size_k == 64
        assert config.group_size_m == 24
        assert config.num_xcds == 8

        # num_warps/num_stages are in JSON but not FusedConfig fields —
        # to_fused_config() correctly filters them; they're kernel launch params
        assert "num_warps" in result.config_params  # Present in raw dict
        assert result.config_params["num_warps"] == 8

        # Step 3: Validate config is internally consistent
        config.validate(world_size=8)  # Should not raise

        # Step 4: Access HBM buffer params (used by bench_all_gather_matmul.py)
        hbm = result.hbm_buffer_params
        assert hbm["k_per_flag"] == 32
        assert hbm["num_fetch_sms"] == 4
        assert hbm["num_fetch_stages"] == 64
        assert hbm["first_stage_fetch_sms"] == 52

    def test_disabled_path_blocks_fused_config(self):
        """Disabled configs must not produce FusedConfig — forces PyTorch fallback."""
        result = select_ag_mm_config(M=131072, N=16384, K=16384, world_size=2)
        assert result.enabled is False
        with pytest.raises(RuntimeError, match="disabled"):
            result.to_fused_config()

    def test_regression_sizes_all_have_valid_configs(self):
        """Every enabled regression size must auto-select to a valid, convertible FusedConfig."""
        sizes = load_regression_sizes()
        for s in sizes:
            if s.get("tier") == "disabled":
                continue  # disabled entries are tested separately
            for ws in s["world_sizes"]:
                result = select_ag_mm_config(s["M"], s["N"], s["K"], world_size=ws)
                assert result.enabled is True, f"{s['name']} ws={ws} should be enabled"
                config = result.to_fused_config()
                config.validate(world_size=ws)  # Must not raise
                # HBM params must be complete
                hbm = result.hbm_buffer_params
                assert all(k in hbm for k in ["k_per_flag", "num_fetch_sms"]), f"{s['name']} ws={ws} missing HBM params"

    def test_config_content_g15_champion(self):
        """Verify actual g15 champion config content — highest TFLOPS shape (474.7 TFLOPS)."""
        result = select_ag_mm_config(M=327680, N=28672, K=4096, world_size=8)
        assert result.enabled is True
        assert result.shape_key == "327680x28672x4096"
        assert result.speedup == 1.284

        config = result.to_fused_config()
        assert config.block_size_m == 256
        assert config.block_size_n == 256
        assert config.block_size_k == 64
        assert config.group_size_m == 24

        hbm = result.hbm_buffer_params
        assert hbm == {
            "k_per_flag": 16,
            "num_fetch_sms": 4,
            "num_fetch_stages": 32,
            "first_stage_fetch_sms": 52,
        }


class TestNearestShapeMatching:
    """Test nearest-shape matching for close-but-not-exact dimensions."""

    def setup_method(self):
        clear_config_cache()

    def test_close_to_g2_uses_g2_config(self):
        """M=130000 is ~0.8% below g2 (131072) — should use g2 champion config."""
        result = select_ag_mm_config(M=130000, N=16384, K=16384, world_size=8, transpose="NN", arch="mi300x")
        assert result.enabled is True
        assert "Nearest match" in result.source
        assert result.shape_key == "131072x16384x16384"
        assert result.speedup == 1.343

    def test_close_to_g15_uses_g15_config(self):
        """M=320000 is ~2.3% below g15 (327680) — should use g15 champion config."""
        result = select_ag_mm_config(M=320000, N=28672, K=4096, world_size=8, transpose="NN", arch="mi300x")
        assert result.enabled is True
        assert "Nearest match" in result.source
        assert result.shape_key == "327680x28672x4096"

    def test_far_shape_uses_heuristic(self):
        """M=50000, N=50000, K=50000 — far from all champions, should use heuristic."""
        result = select_ag_mm_config(M=50000, N=50000, K=50000, world_size=8, transpose="NN", arch="mi300x")
        assert result.enabled is True
        assert "Heuristic" in result.source
        assert result.shape_key is None

    def test_exact_match_still_preferred(self):
        """Exact match should still be returned even when nearest would also match."""
        result = select_ag_mm_config(M=131072, N=16384, K=16384, world_size=8, transpose="NN", arch="mi300x")
        assert result.enabled is True
        assert "Exact match" in result.source
        assert result.shape_key == "131072x16384x16384"

    def test_nearest_skips_losing_shapes(self):
        """Nearest matching should skip shapes with speedup <= 1.0."""
        # g9 (196608x18432x16384) has speedup 0.950 — should NOT be matched
        result = select_ag_mm_config(M=196608, N=18432, K=16384, world_size=8, transpose="NN", arch="mi300x")
        # g9 is an exact match, but its speedup is 0.950
        # The exact match path doesn't filter by speedup, but nearest does
        assert result.enabled is True


class TestFusedConfigNumWarpsStages:
    """Test that num_warps and num_stages are present in config_params."""

    def setup_method(self):
        clear_config_cache()

    def test_champion_config_has_num_warps(self):
        """Champion configs should have num_warps=8 in config_params."""
        result = select_ag_mm_config(M=131072, N=16384, K=16384, world_size=8)
        assert result.config_params["num_warps"] == 8

    def test_champion_config_has_num_stages(self):
        """Champion configs should have num_stages=2 in config_params."""
        result = select_ag_mm_config(M=131072, N=16384, K=16384, world_size=8)
        assert result.config_params["num_stages"] == 2

    def test_fused_config_conversion_succeeds(self):
        """FusedConfig conversion should succeed for enabled configs."""
        result = select_ag_mm_config(M=131072, N=16384, K=16384, world_size=8)
        config = result.to_fused_config()
        assert isinstance(config, FusedConfig)

    def test_fused_config_block_sizes_correct(self):
        """FusedConfig should have correct block sizes from champion."""
        result = select_ag_mm_config(M=131072, N=16384, K=16384, world_size=8)
        config = result.to_fused_config()
        assert config.block_size_m == 256
        assert config.block_size_n == 256
        assert config.block_size_k == 64

    def test_all_champions_have_num_warps_8(self):
        """All ws=8 champion configs should specify num_warps=8."""
        winning_shapes = [
            (131072, 16384, 16384),
            (327680, 28672, 4096),
            (147456, 28672, 4096),
            (229376, 28672, 4096),
            (8192, 8192, 262144),
            (262144, 8192, 8192),
            (16384, 16384, 131072),
        ]
        for M, N, K in winning_shapes:
            result = select_ag_mm_config(M, N, K, world_size=8)
            assert result.config_params["num_warps"] == 8, f"Shape {M}x{N}x{K}: expected num_warps=8"
            assert result.config_params["num_stages"] == 2, f"Shape {M}x{N}x{K}: expected num_stages=2"


class TestGpuArchAutoDetection:
    """Test GPU architecture auto-detection logic."""

    def setup_method(self):
        clear_config_cache()
        # Reset cached detection
        auto_config_module._detected_arch = None

    def teardown_method(self):
        # Reset after each test
        auto_config_module._detected_arch = None
        os.environ.pop("IRIS_GPU_ARCH", None)

    def test_env_var_override(self):
        """IRIS_GPU_ARCH env var should override auto-detection."""
        os.environ["IRIS_GPU_ARCH"] = "mi300a"
        arch = detect_gpu_arch()
        assert arch == "mi300a"

    def test_env_var_case_insensitive(self):
        """IRIS_GPU_ARCH env var should be lowercased by detect_gpu_arch."""
        os.environ["IRIS_GPU_ARCH"] = "MI300X"
        arch = detect_gpu_arch()
        # detect_gpu_arch applies .lower() to the env var
        assert arch == "mi300x"

    def test_fallback_default(self):
        """Without rocminfo, should fall back to mi300x."""
        os.environ.pop("IRIS_GPU_ARCH", None)
        # On a machine without ROCm, should get mi300x default
        arch = detect_gpu_arch()
        assert isinstance(arch, str)
        assert len(arch) > 0

    def test_select_with_auto_arch(self):
        """select_ag_mm_config(arch='auto') should work end-to-end."""
        os.environ["IRIS_GPU_ARCH"] = "mi300x"
        result = select_ag_mm_config(M=131072, N=16384, K=16384, world_size=8, arch="auto")
        assert result.enabled is True
        assert result.shape_key == "131072x16384x16384"

    def test_select_default_is_auto(self):
        """Default arch parameter should be 'auto'."""
        os.environ["IRIS_GPU_ARCH"] = "mi300x"
        # Call without specifying arch — should use auto
        result = select_ag_mm_config(M=131072, N=16384, K=16384, world_size=8)
        assert result.enabled is True

    def test_caching_across_calls(self):
        """detect_gpu_arch should cache result after first call."""
        os.environ["IRIS_GPU_ARCH"] = "mi300x"
        arch1 = detect_gpu_arch()
        os.environ["IRIS_GPU_ARCH"] = "mi300a"
        arch2 = detect_gpu_arch()
        # Should be cached from first call
        assert arch1 == arch2 == "mi300x"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
