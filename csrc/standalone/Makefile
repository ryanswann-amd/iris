# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Build the standalone P2P VMem atomic examples.
# Requires ROCm with hipcc and libhsa-runtime64.

HIPCC    ?= hipcc
ROCM_DIR ?= /opt/rocm
CXXFLAGS  = -O2 -g -std=c++17 -I$(ROCM_DIR)/include
LDFLAGS   = -L$(ROCM_DIR)/lib -Wl,-rpath,$(ROCM_DIR)/lib

.PHONY: all clean

all: p2p_atomics_hsa p2p_atomics_hip

# PATH 3 — HSA fine-grained VMem (correct, P2P atomics pass)
p2p_atomics_hsa: p2p_atomics_hsa.cpp
	$(HIPCC) $(CXXFLAGS) -o $@ $< $(LDFLAGS) -lhsa-runtime64

# PATH 2 — HIP VMem (coarse-grained, P2P atomics cause GPU page fault)
p2p_atomics_hip: p2p_atomics_hip.cpp
	$(HIPCC) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f p2p_atomics_hsa p2p_atomics_hip
