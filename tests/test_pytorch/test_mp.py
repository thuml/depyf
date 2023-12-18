import torch

@torch.compile(backend="eager")
def f(x):
    return x + 1

def main(rank, world_size):
    x = torch.randn(5)
    y = f(x)

def wrap(rank, world_size):
    import depyf
    with depyf.prepare_debug("./depyf_output/multiprocessing"):
        main(rank, world_size)

if __name__ == "__main__":
    world_size = 3
    torch.multiprocessing.spawn(wrap, args=(world_size,), nprocs=world_size, join=True)
