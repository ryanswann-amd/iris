import torch, torch.distributed as dist, os
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
dist.init_process_group(backend='nccl', init_method='env://')
import iris
from iris.ccl import Config

ctx = iris.iris(heap_size=2 << 30)
rank = ctx.get_rank()
world = ctx.get_num_ranks()

for variant in ('one_shot', 'two_shot'):
    for sz in (1024, 4096, 16384):
        elem_size = 2  # bf16
        N = sz // elem_size
        M = world
        # Round up
        if (M * N) % world:
            N = ((N // world) + 1) * world
        inp = ctx.zeros((M, N), dtype=torch.bfloat16)
        inp.fill_(float(rank + 1))
        out = ctx.zeros((M, N), dtype=torch.bfloat16)
        cfg = Config(block_size_m=32, block_size_n=64,
                     all_reduce_variant=variant, all_reduce_distribution=1)
        from iris.ccl.all_reduce import all_reduce_preamble
        ws = all_reduce_preamble(out, inp, ctx, config=cfg)
        ws.prepared = True
        ctx.barrier()  # ensure all ranks ready
        out.zero_()
        torch.cuda.synchronize()
        ctx.barrier()  # sync all ranks before AR
        ctx.ccl.all_reduce(out, inp, config=cfg, workspace=ws)
        ctx.device_barrier()
        torch.cuda.synchronize()
        val = float(out.flatten()[0].cpu().item())
        if rank == 0:
            print(f"variant={variant:10s} bytes={sz:6d} rank=0 val={val} expected=36.0 correct={abs(val - 36.0) < 1e-3 * 36.0}")
        ctx.barrier()
