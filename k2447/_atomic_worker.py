"""Atomic-add worker invoked under rocprofv2 for PMC capture.
Args: dcc_mode block wgp n_iter
"""
import os, sys, torch, triton, triton.language as tl

DCC_ALIGN = {'dcc_disabled': 64, 'dcc_uncompressed': 256, 'dcc_2to1': 1024, 'dcc_4to1': 4096}
DCC_STRIDE_MULT = {'dcc_disabled': 1, 'dcc_uncompressed': 2, 'dcc_2to1': 4, 'dcc_4to1': 8}


@triton.jit
def _kernel(dst_ptr, src_ptr, idx_ptr, N, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(src_ptr + offs, mask=mask)
    i = tl.load(idx_ptr + offs, mask=mask) * STRIDE
    tl.atomic_add(dst_ptr + i, v, mask=mask, sem='acq_rel')


def main():
    dcc = sys.argv[1]
    block = int(sys.argv[2])
    wgp = int(sys.argv[3])
    n_iter = int(sys.argv[4])
    stride_mult = DCC_STRIDE_MULT[dcc]
    align = DCC_ALIGN[dcc]
    N = wgp * block
    dst_n = N * stride_mult + align
    raw = torch.zeros(dst_n + align // 4, dtype=torch.float32, device='cuda')
    base_addr = raw.data_ptr()
    pad = ((-base_addr) % align) // 4
    dst = raw[pad:pad + dst_n]
    src = torch.ones(N, dtype=torch.float32, device='cuda')
    idx = torch.arange(N, dtype=torch.int32, device='cuda') % (dst_n // stride_mult)
    # Warm-up
    for _ in range(2):
        _kernel[(wgp,)](dst, src, idx, N, stride_mult, block)
    torch.cuda.synchronize()
    for _ in range(n_iter):
        _kernel[(wgp,)](dst, src, idx, N, stride_mult, block)
    torch.cuda.synchronize()
    print(f"WORKER_OK dcc={dcc} N={N} env_dcc={os.environ.get('HSA_ENABLE_DCC')}")


if __name__ == '__main__':
    main()
