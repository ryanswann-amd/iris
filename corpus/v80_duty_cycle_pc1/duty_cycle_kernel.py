"""K-2380 duty-cycle atomic ordering-cost kernel.

Single-rank Triton kernel: each program issues BATCHES * BATCH_SIZE atomic ops on a
shared L2 counter, with __builtin_amdgcn_s_sleep N inserted between batches to vary
the temporal duty cycle. Duty cycle = active_cycles / (active_cycles + sleep_cycles).

Four atomic ordering classes (per K-2317 taxonomy):
  XCHG_ACQREL  — atomic_xchg with sem='acq_rel'  (unconditional RMW)
  MAX_ACQREL   — atomic_max  with sem='acq_rel'  (reduction RMW)
  CAS_ACQREL   — atomic_cas  with sem='acq_rel'  (conditional RMW; separate manifold per K-2317)
  FADD_RELEASE — atomic_add  with sem='release'  (float reduction; matches K-2325 family)

We measure end-to-end kernel time at fixed total atomic count (TOTAL_ATOMS) so a
duty-cycle change moves only the gap distribution, not the work amount. The
ordering-cost is then the per-atom amortized latency = kernel_time / TOTAL_ATOMS.
"""
import os
import torch
import triton
import triton.language as tl

# s_sleep operand → wait cycles ≈ 64*N + 1 (per gfx9/CDNA3 ISA). Max operand = 127.
SLEEP_OP_MAX = 127

# NOTE: Triton's @triton.jit dialect doesn't allow Python builtins like str() inside
# the kernel body. So we hard-code `s_sleep 127` (max single-instruction stall on
# CDNA3 = 64*127+1 ≈ 8129 cycles) and tune duty-cycle by varying SLEEP_REPS only.
SLEEP_ASM = "s_sleep 127"

@triton.jit
def _xchg_acqrel(addr_ptr, value, BATCHES: tl.constexpr, BATCH_SIZE: tl.constexpr,
                 SLEEP_REPS: tl.constexpr):
    pid = tl.program_id(0)
    addr = addr_ptr + (pid % 1)  # all programs hammer the same L2 line
    v = value + pid
    for b in range(BATCHES):
        for i in range(BATCH_SIZE):
            _ = tl.atomic_xchg(addr, v + i, sem='acq_rel', scope='gpu')
        for s in range(SLEEP_REPS):
            tl.inline_asm_elementwise(
                "s_sleep 127", "=r,r", [v], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _max_acqrel(addr_ptr, value, BATCHES: tl.constexpr, BATCH_SIZE: tl.constexpr,
                SLEEP_REPS: tl.constexpr):
    pid = tl.program_id(0)
    addr = addr_ptr + (pid % 1)
    v = value + pid
    for b in range(BATCHES):
        for i in range(BATCH_SIZE):
            _ = tl.atomic_max(addr, v + i, sem='acq_rel', scope='gpu')
        for s in range(SLEEP_REPS):
            tl.inline_asm_elementwise(
                "s_sleep 127", "=r,r", [v], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _cas_acqrel(addr_ptr, value, BATCHES: tl.constexpr, BATCH_SIZE: tl.constexpr,
                SLEEP_REPS: tl.constexpr):
    pid = tl.program_id(0)
    addr = addr_ptr + (pid % 1)
    v = value + pid
    for b in range(BATCHES):
        for i in range(BATCH_SIZE):
            _ = tl.atomic_cas(addr, v + i - 1, v + i, sem='acq_rel', scope='gpu')
        for s in range(SLEEP_REPS):
            tl.inline_asm_elementwise(
                "s_sleep 127", "=r,r", [v], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _fadd_release(addr_ptr, value_f, BATCHES: tl.constexpr, BATCH_SIZE: tl.constexpr,
                  SLEEP_REPS: tl.constexpr):
    pid = tl.program_id(0)
    addr = addr_ptr + (pid % 1)
    vf = value_f + pid.to(tl.float32)
    for b in range(BATCHES):
        for i in range(BATCH_SIZE):
            _ = tl.atomic_add(addr, vf + i.to(tl.float32), sem='release', scope='gpu')
        for s in range(SLEEP_REPS):
            tl.inline_asm_elementwise(
                "s_sleep 127", "=r,r", [pid], dtype=tl.int32, is_pure=False, pack=1)


KERNELS = {
    'XCHG_ACQREL':  (_xchg_acqrel,  torch.int32),
    'MAX_ACQREL':   (_max_acqrel,   torch.int32),
    'CAS_ACQREL':   (_cas_acqrel,   torch.int32),
    'FADD_RELEASE': (_fadd_release, torch.float32),
}


def calibrate_active_cycles(op_class, batch_size, device='cuda'):
    """Empirically measure active-cycles per BATCH (one program, no sleep, single block).
    We use this once to size SLEEP_REPS for each duty-cycle target."""
    fn, dt = KERNELS[op_class]
    addr = torch.zeros(8, device=device, dtype=dt)
    val = torch.tensor(0, dtype=dt).item() if dt == torch.int32 else 0.0
    # warm
    fn[(1,)](addr, val, BATCHES=4, BATCH_SIZE=batch_size, SLEEP_REPS=0)
    torch.cuda.synchronize()
    # measure
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(50):
        fn[(1,)](addr, val, BATCHES=4, BATCH_SIZE=batch_size, SLEEP_REPS=0)
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / 50.0
    # MI300X core clock ~ 2.1 GHz nominal; convert ms → cycles per BATCH (4 batches/launch)
    ns_per_batch = (ms * 1e6) / 4.0
    cycles_per_batch = ns_per_batch * 2.1  # ~cycles
    return cycles_per_batch


def sleep_reps_for_duty(duty_pct, active_cycles_per_batch, sleep_op=SLEEP_OP_MAX):
    """Compute SLEEP_REPS so that active/(active+sleep) ≈ duty_pct/100."""
    if duty_pct >= 100:
        return 0
    target_sleep = active_cycles_per_batch * (100.0 / duty_pct - 1.0)
    cycles_per_sleep_inst = 64 * sleep_op + 1  # ≈ 8129 cycles at op=127
    reps = max(1, int(round(target_sleep / cycles_per_sleep_inst)))
    return reps


def launch(op_class, wgp_count, batch_size, batches_per_pgm, duty_pct,
           active_cycles_per_batch, device='cuda'):
    """Build and launch one configured kernel; return (fn_callable, expected_atoms)."""
    fn, dt = KERNELS[op_class]
    addr = torch.zeros(8, device=device, dtype=dt)
    val = 0 if dt == torch.int32 else 0.0
    sleep_reps = sleep_reps_for_duty(duty_pct, active_cycles_per_batch)
    grid = (wgp_count,)
    def _run():
        fn[grid](addr, val,
                 BATCHES=batches_per_pgm, BATCH_SIZE=batch_size,
                 SLEEP_REPS=sleep_reps)
    expected_atoms = wgp_count * batches_per_pgm * batch_size
    return _run, expected_atoms, sleep_reps
