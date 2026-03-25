# %%
from collections.abc import Sequence
import concurrent.futures
import cloudpickle
from multiprocessing import shared_memory, reduction, Queue, Manager
from typing import Union, Callable
from tqdm.auto import tqdm
import numpy as np
from numpy.typing import NDArray
import tempfile, os
import time
import gc

reduction.ForkingPickler.dumps = cloudpickle.dumps  # type: ignore


def _worker(
    func: Sequence[Callable],
    block_args: Sequence[tuple],
    idx: int,
    temp_dir: str,
    shared_mem: Union[tuple[str, tuple[int, ...], type], None],
    progress_queue: Queue = None,
) -> str:
    """
    Worker process function that executes a block of tasks and saves results to disk.

    Each task result is saved as an individual '.npy' file to manage memory pressure.
    Progress is reported back to the parent process via a shared multiprocessing Queue.

    Parameters
    ----------
    func : Sequence[Callable]
        Sequence of functions to apply to each tuple in `block_args`.
    block_args : Sequence[tuple]
        Sequence of argument tuples for each function call.
    idx : int
        Index of the current block, used for directory naming.
    temp_dir : str
        Base directory for temporary storage.
    shared_mem : tuple or None
        Metadata (name, shape, dtype) for an optional shared memory buffer.
    progress_queue : multiprocessing.Queue, optional
        Queue used to send increment signals for the progress bar.

    Returns
    -------
    str
        Path to the subdirectory containing the saved result files for this block.
    """
    block_folder = os.path.join(temp_dir, f"block_{idx}")
    os.makedirs(block_folder, exist_ok=True)

    shm = None
    shared_array = None
    if shared_mem is not None:
        shm_name, shape, dtype = shared_mem
        shm = shared_memory.SharedMemory(name=shm_name)
        shared_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    for i, args in enumerate(block_args):
        if shared_array is not None:
            result = func[i](*args, shared_array)
        else:
            result = func[i](*args)
        fname = os.path.join(block_folder, f"res_{i}.npy")
        to_save = np.empty(1, dtype=object)
        to_save[0] = result
        np.save(fname, to_save, allow_pickle=True)
        del result
        if progress_queue is not None:
            progress_queue.put(1)

    if shm is not None:
        shm.close()

    gc.collect()
    return block_folder


def parallel_blocks_inner(
    funcs: Sequence[Callable],
    all_args: Sequence[tuple],
    num_blocks: int,
    verbose: bool,
    pbar_title: str,
    disable_parallel: bool,
    shared_mem_last_arg: Union[NDArray, None],
) -> list:
    """
    Orchestrate the execution of tasks in parallel blocks with disk-based result caching.

    Divides tasks into groups (blocks), assigns them to worker processes, and monitors
    overall progress. Results are temporarily written to disk to keep the memory footprint
    low during parallel execution, then reloaded sequentially into a final list.

    Parameters
    ----------
    funcs : Sequence[Callable]
        List of functions to execute.
    all_args : Sequence[tuple]
        List of argument tuples matching `funcs`.
    num_blocks : int
        Number of worker processes to use.
    verbose : bool
        If True, displays progress bars for both the execution and gathering phases.
    pbar_title : str
        Base title for the progress bars.
    disable_parallel : bool
        If True, executes tasks sequentially in the main process.
    shared_mem_last_arg : NDArray or None
        An optional NumPy array to be placed in shared memory and passed to every task.

    Returns
    -------
    list
        Ordered list of results.
    """

    n_tasks = len(all_args)

    # Runs in sequential if necessary
    if disable_parallel:
        if shared_mem_last_arg is not None:
            all_args = [tuple(list(args) + [shared_mem_last_arg]) for args in all_args]
        results = []
        for i, (func, args) in enumerate(
            tqdm(
                zip(funcs, all_args),
                total=n_tasks,
                desc=pbar_title,
                disable=not verbose,
            )
        ):
            results.append(func(*args))
        return results

    # Create the shared memory buffer if necessary
    if shared_mem_last_arg is not None:
        shm = shared_memory.SharedMemory(create=True, size=shared_mem_last_arg.nbytes)
        np.ndarray(
            shared_mem_last_arg.shape, dtype=shared_mem_last_arg.dtype, buffer=shm.buf
        )[:] = shared_mem_last_arg
        shared_mem = (shm.name, shared_mem_last_arg.shape, shared_mem_last_arg.dtype)
    else:
        shared_mem = None

    try:
        # Split the functions and arguments into blocks
        nb_each, extras = divmod(n_tasks, num_blocks)
        sizes = extras * [nb_each + 1] + (num_blocks - extras) * [nb_each]
        blocks = []
        funcs_blocks = []
        start = 0
        for size in sizes:
            end = start + size
            blocks.append(all_args[start:end])
            funcs_blocks.append(funcs[start:end])
            start = end

        temp_dir = tempfile.mkdtemp(prefix="parallel_blocks_")

        with Manager() as manager:  # Utiliser 'with' pour un cleanup automatique
            progress_queue = manager.Queue()

            args = [
                (func_block, block, i, temp_dir, shared_mem, progress_queue)
                for i, (func_block, block) in enumerate(zip(funcs_blocks, blocks))
            ]

            with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_blocks
            ) as executor:
                futures = [executor.submit(_worker, *a) for a in args]

                with tqdm(
                    total=n_tasks, desc=f"{pbar_title}: Run", disable=not verbose
                ) as pbar:
                    while not all(f.done() for f in futures):
                        try:
                            msg = progress_queue.get(timeout=0.1)
                            pbar.update(msg)
                        except:
                            if any(
                                f.done() and f.exception() is not None for f in futures
                            ):
                                break
                files = [f.result() for f in futures]

    finally:
        # Free the memory space allocated to the shared array
        if shared_mem_last_arg is not None:
            shm.close()  # type: ignore
            shm.unlink()  # type: ignore

    # Load and collect results from each file
    with tqdm(total=n_tasks, desc=f"{pbar_title}: Gather", disable=not verbose) as pbar:
        results = [None] * n_tasks
        current_idx = 0
        for idx, block_folder in enumerate(files):
            for i in range(len(blocks[idx])):
                fpath = os.path.join(block_folder, f"res_{i}.npy")
                results[current_idx] = np.load(fpath, allow_pickle=True)[0]
                os.remove(fpath)
                current_idx += 1
                pbar.update(1)
            os.rmdir(block_folder)
        os.rmdir(temp_dir)
    return results


def parallel_blocks(
    funcs: Union[Callable, Sequence[Callable]],
    all_args: Union[Sequence[tuple], None] = None,
    num_blocks: Union[int, None] = None,
    verbose: bool = True,
    pbar_title: str = "Processing blocks",
    disable_parallel: bool = False,
    est_proc_cost: float = 0.5,
    shared_mem_last_arg: Union[NDArray, None] = None,
) -> list:
    """
    Execute tasks in parallel with disk-based caching and shared memory support.

    This function orchestrates the execution of multiple independent tasks. It
    automatically handles process creation, progress reporting via a single
    global bar, and manages memory by swapping intermediate results to disk.

    Parameters
    ----------
    funcs : Callable or Sequence[Callable]
        - If a single Callable: Applied to every tuple in `all_args` (Mapping mode).
        - If a Sequence of Callables: Each function is executed with its
          corresponding arguments (Task Pool mode).
    all_args : Sequence[tuple], optional
        A list of tuples containing positional arguments for each task.
        Example: `[(arg1,), (arg1, arg2)]`. If None, functions are called without args.
    num_blocks : int, optional
        Number of worker processes. Defaults to half of the available CPU cores.
    verbose : bool, optional
        If True, displays real-time progress bars for execution and gathering.
    pbar_title : str, optional
        Prefix text for the progress bars.
    disable_parallel : bool, optional
        If True, forces sequential execution in the main process.
    est_proc_cost : float, optional
        Estimated overhead (in seconds) to spawn a new process. If the first
        task's duration is significantly lower than the overhead divided by
        efficiency gain, the remaining tasks switch to sequential mode.
    shared_mem_last_arg : NDArray, optional
        A NumPy array placed in a shared memory segment. It is automatically
        appended as the **last argument** to every function call. This avoids
        duplicating large read-only data across worker processes.

    Returns
    -------
    list
        A list of results in the same order as the input tasks.

    Notes
    -----
    - **Memory Efficiency**: Intermediate results are saved as `.npy` files in
      a temporary directory. This prevents the RAM from filling up if workers
      produce large objects faster than the main process can collect them.
    - **Error Handling**: If any worker fails, the exception is captured,
      other workers are terminated, and the error is raised in the main process.
    - **Shared Memory**: The `shared_mem_last_arg` is read-only. Workers access
      the original memory buffer without copying, which is critical for
      processing large datasets (GBs).

    Examples
    --------
    >>> # 1. Mapping mode (Single function, multiple arguments)
    >>> def square(n): return n*n
    >>> args = [(i,) for i in range(10)]
    >>> results = parallel_blocks(square, args)

    >>> # 2. Task Pool mode (Different functions/lambdas)
    >>> tasks = [lambda: os.getpid(), lambda: time.time()]
    >>> results = parallel_blocks(tasks)

    >>> # 3. Using Shared Memory (The large array is the LAST argument)
    >>> def lookup(index, big_array):
    ...     return big_array[index]
    >>> data = np.linspace(0, 100, 1000000)
    >>> query_indices = [(10,), (500,), (9999,)]
    >>> results = parallel_blocks(lookup, query_indices, shared_mem_last_arg=data)
    """
    if num_blocks is None:
        num_blocks = max(1, os.cpu_count() // 2)  # type: ignore

    if callable(funcs):
        assert (
            all_args is not None
        ), "If 'funcs' is a single callable, 'all_args' must be provided as a list of argument tuples."
        funcs = [funcs] * len(all_args)
    else:
        if all_args is None:
            all_args = [()] * len(funcs)
        assert len(funcs) == len(
            all_args
        ), "If 'funcs' is an iterable of callables, its length must match the number of argument tuples in 'all_args'."

    n_tasks = len(all_args)
    if disable_parallel or num_blocks == 1 or n_tasks <= 1:
        return parallel_blocks_inner(
            funcs, all_args, num_blocks, verbose, pbar_title, True, shared_mem_last_arg
        )

    t0 = time.time()
    first_result = (
        funcs[0](*all_args[0])
        if shared_mem_last_arg is None
        else funcs[0](*all_args[0], shared_mem_last_arg)
    )
    t_first = time.time() - t0
    t_thresh = (num_blocks / (num_blocks - 1)) * (num_blocks / n_tasks) * est_proc_cost
    disable_parallel = t_first <= t_thresh
    if verbose:
        print(
            f"First task time: {t_first:.3f}s, t_seuil: {t_thresh:.3f}s -> "
            f"{'Sequential' if disable_parallel else 'Parallel'}"
        )
    results_rest = parallel_blocks_inner(
        funcs[1:],
        all_args[1:],
        num_blocks,
        verbose,
        pbar_title,
        disable_parallel,
        shared_mem_last_arg,
    )
    results = [first_result] + results_rest

    return results


if __name__ == "__main__":

    import numpy as np
    import pyvista as pv

    def make_sphere(center):
        sphere = pv.Sphere(
            radius=1.0, center=center, theta_resolution=10, phi_resolution=10
        )
        sphere["elevation"] = sphere.points[:, 2]
        return sphere

    ms = lambda c: make_sphere(c)

    data_input = [(np.random.randint(-10, 10, size=3),) for _ in range(2000)]
    spheres = parallel_blocks(ms, data_input)

    plotter = pv.Plotter()
    plotter.add_mesh(pv.merge(spheres), scalars="elevation", show_edges=True)
    plotter.show()

# %%
