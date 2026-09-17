"""One island: a master (this class, running in whatever thread/process
constructs it) holding the subpopulation, plus a persistent pool of N slave
processes that only ever do fitness evaluation.

Workers are spawned once per island run and kept alive across generations
(matching always-on cores in Xue et al., rather than paying process-spawn
cost every generation). Dispatch goes through a single shared task queue —
the software analogue of the single injection channel — so T_interval here
is genuinely the master's serialisation cost, not an artifact of re-spawning
workers.
"""

import dataclasses
import multiprocessing as mp
import random
import time

from hpga.config import HPGAConfig
from hpga.instrumentation import GenerationRecord, RunRecorder
from hpga.operators import next_generation, random_population
from hpga.worker import STOP, worker_loop


class Island:
    def __init__(self, config: HPGAConfig, rng: random.Random | None = None):
        self.config = config
        self.rng = rng if rng is not None else random.Random(config.seed)
        self.task_queue: mp.Queue = mp.Queue()
        self.result_queue: mp.Queue = mp.Queue()
        self.workers: list[mp.Process] = []
        self.population = random_population(config.pop_size, config.genome_length, self.rng)

    def start_workers(self) -> None:
        cfg = self.config
        for wid in range(cfg.n_workers):
            p = mp.Process(
                target=worker_loop,
                args=(self.task_queue, self.result_queue, cfg.sequence, cfg.collision_penalty, wid),
                daemon=True,
            )
            p.start()
            self.workers.append(p)

    def shutdown(self) -> None:
        for _ in self.workers:
            self.task_queue.put(STOP)
        for p in self.workers:
            p.join(timeout=10)
        self.workers = []

    def _dispatch_and_collect(self, population: list, gen_id: int, gen_record: GenerationRecord) -> list[float]:
        n = len(population)
        for ind_id, genome in enumerate(population):
            t = time.perf_counter()
            self.task_queue.put((gen_id, ind_id, genome, t))
            gen_record.dispatch_timestamps.append(t)

        fitnesses: list[float | None] = [None] * n
        for _ in range(n):
            result = self.result_queue.get()
            t_arrival = time.perf_counter()
            gen_record.arrival_timestamps.append(t_arrival)
            fitnesses[result["ind_id"]] = result["fitness"]
            gen_record.t_calc_list.append(result["t_calc"])
            w = result["worker_id"]
            gen_record.per_worker_calc_time[w] = gen_record.per_worker_calc_time.get(w, 0.0) + result["t_calc"]

        assert all(f is not None for f in fitnesses)
        return fitnesses  # type: ignore[return-value]

    def run(self) -> RunRecorder:
        cfg = self.config
        recorder = RunRecorder(n_workers=cfg.n_workers, config_summary=dataclasses.asdict(cfg))

        # Worker spawn is a one-time deployment cost (dominated by Windows
        # process-creation overhead, not by anything the bottleneck analysis
        # is about), so it's measured but excluded from the timed window —
        # otherwise short sweeps would just be measuring process-startup
        # noise instead of steady-state dispatch/compute behaviour.
        t0 = time.perf_counter()
        self.start_workers()
        recorder.startup_time = time.perf_counter() - t0
        recorder.run_start = time.perf_counter()
        try:
            population = self.population
            for gen_id in range(cfg.n_generations):
                gen_record = GenerationRecord(gen_id=gen_id, dis_time=0.0, ga_time=0.0)

                dis_start = time.perf_counter()
                fitnesses = self._dispatch_and_collect(population, gen_id, gen_record)
                gen_record.dis_time = time.perf_counter() - dis_start

                recorder.best_fitness_by_gen.append(max(fitnesses))

                ga_start = time.perf_counter()
                population = next_generation(
                    population,
                    fitnesses,
                    cfg.pop_size,
                    cfg.tournament_k,
                    cfg.crossover_rate,
                    cfg.mutation_rate,
                    cfg.elitism,
                    self.rng,
                )
                gen_record.ga_time = time.perf_counter() - ga_start

                recorder.generations.append(gen_record)
            self.population = population
        finally:
            recorder.finish()
            t1 = time.perf_counter()
            self.shutdown()
            recorder.shutdown_time = time.perf_counter() - t1
        return recorder
