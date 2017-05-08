# -*- coding: utf-8 -*-
"""
© Michael Widrich, Markus Hofmarcher, 2017

Handling of worker subprocesses and threads for plotting

"""

from multiprocessing import Pool


def stop_plotting_daemon(plotting_queue, plotting_proc):
    """End plotting daemon process properly"""
    # tell process to finish
    plotting_queue.put(0)
    # reap process
    plotting_proc.join()


def launch_proc(target, arguments, wait=False, daemon=True):
    import multiprocessing as mp
    proc = mp.Process(target=target, args=arguments)
    proc.daemon = daemon
    proc.start()
    if wait:
        proc.wait()
    return proc


def plotting_demon(plotting_queue, multicore):
    print("Starting plotting daemon...", end=" ")
    pool = Pool(processes=multicore)
    print("Done!")
    while True:
        rec = plotting_queue.get()
        if rec == 0:
            break
        func, arguments = rec
        
        pool.apply_async(func, arguments)
    
    pool.close()
    pool.join()
    del pool
    print("Plotting daemon terminated.")
    exit(0)


def start_plotting_daemon(wait=False, multicore=3):
    import multiprocessing as mp
    plotting_queue = mp.Queue()
    proc = launch_proc(target=plotting_demon, arguments=[plotting_queue, multicore], daemon=False, wait=wait)
    return (plotting_queue, proc)