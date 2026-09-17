"""Slave-side entry point. Must stay a plain module-level function: on
Windows, multiprocessing uses 'spawn', which re-imports this module in the
child and pickles the Process target — closures and bound methods don't
survive that.
"""

import time
from multiprocessing import Queue

from hpga.hp_model import evaluate_fitness

STOP = None  # sentinel


def worker_loop(task_queue: "Queue", result_queue: "Queue", sequence: str, collision_penalty: float, worker_id: int) -> None:
    while True:
        task = task_queue.get()
        if task is STOP:
            break
        gen_id, ind_id, genome, t_dispatch = task

        t_start = time.perf_counter()
        fitness = evaluate_fitness(genome, sequence, collision_penalty)
        t_end = time.perf_counter()

        result_queue.put(
            {
                "gen_id": gen_id,
                "ind_id": ind_id,
                "genome": genome,
                "fitness": fitness,
                "worker_id": worker_id,
                "t_dispatch": t_dispatch,
                "t_start": t_start,
                "t_end": t_end,
                "t_calc": t_end - t_start,
            }
        )
