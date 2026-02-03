import multiprocessing as mp
from multiprocessing import shared_memory
import threading
import queue
from typing import Iterable, Union, Callable
from tqdm import tqdm
import numpy as np
import tempfile, os
import time
import gc


def _save_worker(save_queue: queue.Queue):
    """
    Background worker that saves data to disk from a queue.

    This function runs in a separate thread. It waits for filename-result pairs
    pushed into the `save_queue`, and writes them to disk using `np.save`.
    When it receives a `None` sentinel value, it terminates.

    Parameters
    ----------
    save_queue : queue.Queue
        A thread-safe queue containing `(fname, result)` pairs to be saved. The thread
        stops when it receives a `None` item.

    Notes
    -----
    - Each result is saved as a `numpy.ndarray` with `dtype=object` using `np.save`.
    - The queue can be bounded to control memory usage in the main thread.
    """
    while True:
        item = save_queue.get()
        if item is None:
            break  # stop signal
        fname, result = item
        to_save = np.empty(1, dtype=object)
        to_save[0] = result
        np.save(fname, to_save, allow_pickle=True)
        save_queue.task_done()


def _worker(
    func: Iterable[Callable],
    block_args: Iterable[tuple],
    idx: int,
    temp_dir: str,
    verbose: bool,
    jupyter: bool,
    pbar_title_prefix: str,
    shared_mem: Union[tuple[str, tuple[int, ...], type], None],
) -> str:
    """
    Apply a function to a sequence of argument tuples, saving results asynchronously during computation.

    This function iterates over `block_args`, applying `func[i]` to each tuple of arguments,
    and delegates saving to disk to a background thread. This allows computation to proceed
    without waiting for disk writes to complete, reducing idle CPU time and improving throughput.

    Parameters
    ----------
    func : Iterable[Callable]
        Functions to apply to each tuple in `block_args`. Each function should accept unpacked arguments
        from its corresponding tuple.
    block_args : Iterable[tuple]
        Iterable of `tuple` arguments to be passed to `func`. Each `tuple` is unpacked as arguments.
    idx : int
        Index of the current block, used for progress bar positioning and output file naming.
    temp_dir : str
        Path to the temporary directory where result files will be saved.
    verbose : bool
        If `True`, enables the progress bar. If `False`, disables it.
    jupyter : bool
        If `True`, sets the progress bar position to `0` for Jupyter notebook compatibility.
        If `False`, uses `idx` as the position.
    pbar_title_prefix : str
        Prefix string for the progress bar description.

    Returns
    -------
    str
        Path to the folder containing the saved `.npy` result files for this block.

    Notes
    -----
    - A background thread is launched to handle saving to disk via `np.save`.
    - A bounded queue is used to limit memory usage when saving is slower than computation.
    - Result files are saved as `"res_{i}.npy"` in a folder named `"block_{idx}"` within `temp_dir`.
    - Each file contains a single `numpy.ndarray` with `dtype=object` and one element.
    - Explicit garbage collection is triggered after saving.
    """
    position = 0 if jupyter else idx
    block_folder = os.path.join(temp_dir, f"block_{idx}")
    os.makedirs(block_folder, exist_ok=True)

    # File d’écriture limitée pour éviter surcharge mémoire
    save_queue = queue.Queue(maxsize=10)
    writer_thread = threading.Thread(target=_save_worker, args=(save_queue,))
    writer_thread.start()

    with tqdm(
        block_args,
        desc=f"{pbar_title_prefix}: Block {idx}",
        disable=not verbose,
        position=position,
    ) as pbar:
        if shared_mem is not None:
            shm_name, shape, dtype = shared_mem
            shm = shared_memory.SharedMemory(name=shm_name)
            try:
                array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                for i, args in enumerate(pbar):
                    result = func[i](*args, array)
                    fname = os.path.join(block_folder, f"res_{i}.npy")
                    save_queue.put((fname, result))  # enqueue (blocks if full)
            finally:
                shm.close()
        else:
            for i, args in enumerate(pbar):
                result = func[i](*args)
                fname = os.path.join(block_folder, f"res_{i}.npy")
                save_queue.put((fname, result))  # mise en file (bloque si trop plein)

    # Finalisation propre
    save_queue.put(None)  # signal de fin
    writer_thread.join()  # on attend que tout soit écrit
    gc.collect()
    return block_folder


def parallel_blocks_inner(
    funcs: Iterable[Callable],
    all_args: Iterable[tuple],
    num_blocks: int,
    verbose: bool,
    pbar_title: str,
    disable_parallel: bool,
    shared_mem_last_arg: Union[np.ndarray, None],
) -> list:
    """
    Execute a list of functions with their corresponding argument tuples, optionally in parallel blocks.

    This function performs the actual execution of tasks either sequentially or in parallel,
    depending on the `disable_parallel` flag. When running in parallel, the tasks are divided into
    `num_blocks` groups (blocks), each executed by a separate worker process. Intermediate results
    are temporarily saved to disk as `.npy` files to limit memory usage and are reloaded sequentially
    after all processes complete.

    Parameters
    ----------
    funcs : Iterable[Callable]
        List of functions to execute. Must have the same length as `all_args`.
        Each function is called as `func(*args)` for its corresponding argument tuple.
    all_args : Iterable[tuple]
        List of tuples, each containing the arguments for the corresponding function in `funcs`.
    num_blocks : int
        Number of parallel blocks (i.e., worker processes) to use when `disable_parallel` is False.
        Determines how many subsets of tasks will be distributed among processes.
    verbose : bool
        If True, enables progress bars and displays information about block processing and result gathering.
    pbar_title : str
        Title prefix used for progress bar descriptions.
    disable_parallel : bool
        If True, all tasks are executed sequentially in the current process. If False, tasks are divided
        into blocks and processed in parallel using a multiprocessing pool.
    shared_mem_last_arg : Union[np.ndarray, None]
        Optional NumPy array placed in shared memory and appended automatically as the last argument
        of each task. This is useful for sharing large read-only data (e.g., images, meshes) without
        duplicating memory across processes.

    Returns
    -------
    list
        List of results obtained from applying each function to its corresponding argument tuple,
        preserving the original task order.

    Notes
    -----
    - **Sequential mode:** if `disable_parallel` is True, all functions are executed in the current
      process with an optional progress bar.
    - **Parallel mode:**
        * The tasks are split into `num_blocks` subsets.
        * Each subset is processed by a separate worker via `multiprocessing.Pool`.
        * Each worker writes its results as `.npy` files inside a temporary subfolder.
        * After all workers complete, results are reloaded in the original order, and temporary files
          and folders are deleted.
    - **Shared memory:** if `shared_mem_last_arg` is provided, it is stored once in shared memory
      and accessible by all workers, avoiding redundant copies of large data arrays.
    - Compatible with both standard Python terminals and Jupyter notebooks (adaptive progress bars).
    - Intended for internal use by higher-level orchestration functions such as `parallel_blocks()`.
    """

    n_tasks = len(all_args)

    # Detect if running in a Jupyter environment
    try:
        from IPython import get_ipython

        jupyter = get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except:
        jupyter = False

    # Runs in sequential if necessary
    if disable_parallel:
        if shared_mem_last_arg is not None:
            all_args = [list(args) + [shared_mem_last_arg] for args in all_args]
        results = []
        position = 0 if jupyter else 0
        for i, (func, args) in enumerate(
            tqdm(
                zip(funcs, all_args),
                total=n_tasks,
                desc=pbar_title,
                disable=not verbose,
                position=position,
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
    args = [
        (func_block, block, i, temp_dir, verbose, jupyter, pbar_title, shared_mem)
        for i, (func_block, block) in enumerate(zip(funcs_blocks, blocks))
    ]

    # Start the worker processes
    with mp.Pool(num_blocks) as pool:
        files = pool.starmap(_worker, args)

    # Free the memory space allocated to the shared array
    if shared_mem_last_arg is not None:
        shm.close()
        shm.unlink()

    # Load and collect results from each file
    with tqdm(total=n_tasks, desc=f"{pbar_title}: Gather", disable=not verbose) as pbar:
        results = []
        for idx, block_folder in enumerate(files):
            for i in range(len(blocks[idx])):
                fpath = os.path.join(block_folder, f"res_{i}.npy")
                results.append(np.load(fpath, allow_pickle=True)[0])
                os.remove(fpath)
                pbar.update(1)
            os.rmdir(block_folder)
        os.rmdir(temp_dir)
    return results


def parallel_blocks(
    funcs: Union[Callable, Iterable[Callable]],
    all_args: Union[Iterable[tuple], None] = None,
    num_blocks: Union[int, None] = None,
    verbose: bool = True,
    pbar_title: str = "Processing blocks",
    disable_parallel: bool = False,
    est_proc_cost: float = 0.5,
    shared_mem_last_arg: Union[np.ndarray, None] = None,
) -> list:
    """
    Execute a set of independent tasks sequentially or in parallel, depending on their estimated cost.

    The function evaluates the runtime of the first task to decide whether parallelization is worth
    the overhead of process creation. If parallel execution is deemed beneficial, the remaining tasks
    are distributed across several blocks processed in parallel. Otherwise, all tasks are executed
    sequentially. This strategy is especially useful when task runtimes are variable or short compared
    to process spawning costs.

    Parameters
    ----------
    funcs : Union[Callable, Iterable[Callable]]
        Function or list of functions to execute.
        - If a single function is provided, it will be applied to each argument tuple in `all_args`.
        - If a list of functions is provided, it must have the same length as `all_args`, allowing each
          task to use a distinct callable.
    all_args : Union[Iterable[tuple], None], optional
        Iterable of tuples containing the positional arguments for each function call.
        If the function takes no arguments, set `all_args` to `None` (defaults to empty tuples).
    num_blocks : Union[int, None], optional
        Number of parallel blocks (i.e., worker processes) to use. Defaults to half the number of CPU cores.
        A value of 1 forces sequential execution.
    verbose : bool, optional
        If True, displays timing information and progress bars. Default is True.
    pbar_title : str, optional
        Title prefix displayed in the progress bar. Default is "Processing blocks".
    disable_parallel : bool, optional
        If True, forces all computations to run sequentially regardless of estimated profitability.
        Default is False.
    est_proc_cost : float, optional
        Estimated process creation cost in seconds. Used to determine whether parallelization
        will yield a net speedup. Default is 0.5 s.
    shared_mem_last_arg : Union[np.ndarray, None], optional
        Shared-memory NumPy array to be appended automatically as the last argument in each task.
        This allows tasks to read from a large, read-only array without duplicating it in memory.
        Default is None.

    Returns
    -------
    list
        List of results, one per task, preserving the input order.

    Notes
    -----
    - The first task is executed sequentially to estimate its runtime.
    - Parallelization is enabled only if the estimated time saved exceeds the cost of process creation.
    - When parallel mode is used, tasks are executed in blocks, and intermediate results are stored
      temporarily on disk to limit memory usage, then reloaded and combined sequentially.
    - Compatible with Jupyter progress bars (`tqdm.notebook`).
    """
    if num_blocks is None:
        num_blocks = max(1, os.cpu_count() // 2)

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
