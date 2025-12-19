# irisrun

`irisrun` is a command-line tool for launching distributed Iris programs, similar to `torchrun`. It automatically manages distributed initialization by finding free ports and setting up the environment for multi-GPU execution.

## Features

- **Automatic Port Management**: Finds and uses free TCP ports, avoiding conflicts when processes crash
- **Environment Setup**: Automatically sets `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT` environment variables
- **Compatible with Existing Scripts**: Scripts can work with both `irisrun` and standalone execution

## Installation

After installing Iris, `irisrun` is automatically available:

```bash
pip install -e .
```

## Usage

Basic usage:

```bash
irisrun --nproc_per_node=N script.py [script_args...]
```

### Arguments

- `--nproc_per_node`: Number of processes to launch per node (typically the number of GPUs)
- `--master_addr`: Master node address (default: `127.0.0.1`)
- `--master_port`: Master node port (default: auto-selected free port)
- `script`: Python script to run
- `script_args`: Arguments to pass to the script

### Examples

Run the load benchmark on 2 GPUs:

```bash
irisrun --nproc_per_node=2 examples/00_load/load_bench.py --verbose
```

Run the store benchmark on 4 GPUs with custom buffer size:

```bash
irisrun --nproc_per_node=4 examples/01_store/store_bench.py --buffer_size 8192 --verbose
```

Run with a specific master port:

```bash
irisrun --nproc_per_node=2 --master_port=29600 examples/00_load/load_bench.py
```

## How It Works

1. `irisrun` finds a free TCP port (unless `--master_port` is specified)
2. It spawns `N` processes using `torch.multiprocessing.spawn`
3. Each process gets environment variables set:
   - `RANK`: The process rank (0 to N-1)
   - `LOCAL_RANK`: Same as `RANK` for single-node execution
   - `WORLD_SIZE`: Total number of processes
   - `MASTER_ADDR`: Address of the master node
   - `MASTER_PORT`: Port for distributed communication
4. The script executes in each process with these environment variables available

## Updating Scripts to Support irisrun

Scripts can support both `irisrun` and standalone execution by checking for environment variables:

```python
def _worker(local_rank, world_size, init_url, args):
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    
    # Check if running via irisrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", "29500")
        init_method = f"tcp://{master_addr}:{master_port}"
        
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            device_id=torch.device(f"cuda:{rank}"),
        )
    else:
        # Standalone execution with hardcoded port
        dist.init_process_group(
            backend=backend,
            init_method=init_url,
            world_size=world_size,
            rank=local_rank,
            device_id=torch.device(f"cuda:{local_rank}"),
        )

def main():
    args = parse_args()
    
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Running via irisrun - already spawned
        _worker(None, None, None, args)
    else:
        # Standalone - spawn processes
        init_url = "tcp://127.0.0.1:29500"
        mp.spawn(
            fn=_worker,
            args=(args["num_ranks"], init_url, args),
            nprocs=args["num_ranks"],
            join=True,
        )
```

## Benefits

- **No Port Conflicts**: Automatically finds free ports, eliminating the common issue of port conflicts when scripts crash
- **Easier Development**: Simplifies multi-GPU development by handling distributed setup automatically
- **Cleaner Code**: Separates infrastructure concerns from application logic
- **Familiar Interface**: Similar to `torchrun`, making it easy for PyTorch users to adopt

## Troubleshooting

### Port Already in Use

If you specify `--master_port` and get a "port already in use" error, let `irisrun` auto-select a port by omitting the `--master_port` argument.

### CUDA Device Mismatch

Ensure `--nproc_per_node` matches the number of available GPUs or that `ROCR_VISIBLE_DEVICES` is set correctly.

### Script Not Found

Use absolute or relative paths to the script. For example:

```bash
irisrun --nproc_per_node=2 ./examples/00_load/load_bench.py
```
