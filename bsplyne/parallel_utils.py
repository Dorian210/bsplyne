import multiprocessing as mp
import threading
import queue
from typing import Iterable, Union, Callable
from tqdm import tqdm
import numpy as np
import tempfile, os
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

def _worker(func: Iterable[Callable], block_args: Iterable[tuple], idx: int, temp_dir: str, verbose: bool, jupyter: bool, pbar_title_prefix: str) -> str:
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

    for i, args in enumerate(tqdm(block_args,
                                  desc=f"{pbar_title_prefix}: Block {idx}",
                                  disable=not verbose,
                                  position=position)):
        result = func[i](*args)
        fname = os.path.join(block_folder, f"res_{i}.npy")
        save_queue.put((fname, result))  # mise en file (bloque si trop plein)

    # Finalisation propre
    save_queue.put(None)         # signal de fin
    writer_thread.join()         # on attend que tout soit écrit
    gc.collect()
    return block_folder

def parallel_blocks(funcs: Union[Callable, Iterable[Callable]], 
                    all_args: Union[Iterable[tuple], None]=None, 
                    num_blocks: Union[int, None]=None, 
                    verbose: bool=True, 
                    pbar_title: str="Processing blocks", 
                    disable_parallel: bool=False) -> list:
    """
    Execute a function in parallel over blocks of arguments, saving each block's results to disk and 
    reloading them sequentially.

    This function divides the input `all_args` into a specified number of blocks, applies `func` to each 
    tuple of arguments in parallel using multiple processes, and collects the results into a single list. 
    Intermediate results are stored as `.npy` files in a temporary directory and are loaded and concatenated 
    after processing.

    Parameters
    ----------
    funcs : Union[Callable, Iterable[Callable]]
        If multiple functions are passed, their number should match the number of arguments passed.
        Function to apply to each tuple of arguments in `all_args`.
    all_args : Union[Iterable[tuple], None], optional
        Iterable of tuples, where each tuple contains the arguments for a single call to `func`.
        If the function takes no argument, set `all_args` to `None`. By default, None.
    num_blocks : Union[int, None], optional
        Number of blocks to split `all_args` into for parallel processing.
        If `None`, defaults to half the number of available CPU cores.
        By default, None.
    verbose : bool, optional
        If `True`, displays a progress bar for both block processing and result gathering.
        By default, True.
    pbar_title : str, optional
        Title for the progress bar.
        By default, "Processing blocks".
    disable_parallel : bool, optional
        If `True`, disables parallel processing and runs sequentially. By default, False.

    Returns
    -------
    list
        List containing the results from applying `func` to each tuple of arguments in `all_args`, in the 
        original order.

    Notes
    -----
    - Each block is processed in a separate process and results are saved as `.npy` files in a temporary 
    directory.
    - The temporary directory and files are removed after results are gathered.
    - If running in a Jupyter environment, progress bars are displayed in a single position for better 
    visualization.
    """
    if num_blocks is None:
        num_blocks = max(1, os.cpu_count()//2)
    
    if callable(funcs):
        assert all_args is not None, "If 'funcs' is a single callable, 'all_args' must be provided as a list of argument tuples."
        funcs = [funcs]*len(all_args)
    else:
        if all_args is None:
            all_args = [()]*len(funcs)
        assert len(funcs)==len(all_args), "If 'funcs' is an iterable of callables, its length must match the number of argument tuples in 'all_args'."

    # Detect if running in a Jupyter environment
    try:
        from IPython import get_ipython
        jupyter = get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except:
        jupyter = False
    
    # Runs in sequential if necessary
    if disable_parallel or num_blocks==1:
        results = []
        position = 0 if jupyter else 0
        for i, (func, args) in enumerate(tqdm(zip(funcs, all_args), 
                                               total=len(all_args), 
                                               desc=pbar_title,
                                               disable=not verbose, 
                                               position=position)):
            results.append(func(*args))
        return results
    
    # Split the functions and arguments into blocks
    funcs = np.array(funcs, dtype=object)
    funcs_blocks = np.array_split(funcs, num_blocks)
    all_args = np.array(all_args, dtype=object)
    blocks = np.array_split(all_args, num_blocks)

    temp_dir = tempfile.mkdtemp(prefix="parallel_blocks_")
    args = [(func_block, block, i, temp_dir, verbose, jupyter, pbar_title) 
            for i, (func_block, block) in enumerate(zip(funcs_blocks, blocks))]

    # Start the worker processes
    with mp.Pool(num_blocks) as pool:
        files = pool.starmap(_worker, args)

    # Load and collect results from each file
    with tqdm(total=all_args.shape[0], desc=f"{pbar_title}: Gather", disable=not verbose) as pbar:
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