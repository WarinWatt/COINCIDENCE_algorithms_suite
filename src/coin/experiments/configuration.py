from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExperimentConfiguration:
    algorithms: tuple[str, ...]
    objectives: tuple[str, ...]
    population_size: int = 40
    evaluation_budget: int = 400
    maximum_generations: int = 100
    seeds: tuple[int, ...] = (42,)
    reward_ratio: int = 25
    punishment_ratio: int = 25
    training_rate: int = 5
    rewards_enabled: bool = True
    punishments_enabled: bool = True
    learning_mode: str = "conservative"
    crossover_probability: float = 0.9
    mutation_probability: float = 0.2
    eliminate_duplicates: bool = True
    hbsa_selection_ratio: int = 50
    hbsa_bias_ratio: float = 0.005
    hbsa_sampling_mode: str = "wo"
    hbsa_template_sample_ratio: int = 50
    parameter_grid: bool = False
    parameter_grid_algorithms: tuple[str, ...] = ()
    rose_selection_ratio: int = 20
    rose_roll_mode: str = "random"
    rose_fixed_roll: int = 3
    rose_max_roll: int = 5
    rose_node_weight: float = 0.25
    rose_temperature: float = 1.0
    rose_smoothing: float = 1.0
    rose_template_sample_ratio: int = 50
    rose_reference_selection: str = "uniform"

    def __post_init__(self):
        if not self.algorithms:
            raise ValueError("at least one algorithm is required")
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.evaluation_budget < self.population_size or self.evaluation_budget % self.population_size:
            raise ValueError("evaluation_budget must be a positive multiple of population_size")
        if self.evaluation_budget // self.population_size > self.maximum_generations:
            raise ValueError("maximum_generations is smaller than the evaluation budget")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique")
        if self.training_rate <= 0:
            raise ValueError("training_rate must be positive")
        if not 1 <= self.hbsa_selection_ratio <= 100:
            raise ValueError("hbsa_selection_ratio must be between 1 and 100")
        if self.hbsa_bias_ratio <= 0:
            raise ValueError("hbsa_bias_ratio must be positive")
        if self.hbsa_sampling_mode not in ("wo", "wt"):
            raise ValueError("hbsa_sampling_mode must be 'wo' or 'wt'")
        if not 1 <= self.hbsa_template_sample_ratio <= 100:
            raise ValueError("hbsa_template_sample_ratio must be between 1 and 100")
        allowed_grids = {"edge_coin", "position_coin", "ehbsa", "nhbsa"}
        if set(self.parameter_grid_algorithms) - allowed_grids:
            raise ValueError("parameter_grid_algorithms contains an unsupported algorithm")
        if not 1 <= self.rose_selection_ratio <= 100: raise ValueError("invalid ROSE selection ratio")
        if self.rose_roll_mode not in ("fixed", "random", "all"): raise ValueError("invalid ROSE roll mode")
        if self.rose_fixed_roll < 1 or self.rose_max_roll < 1: raise ValueError("ROSE roll sizes must be positive")
        if not 0 <= self.rose_node_weight <= 1: raise ValueError("ROSE node weight must be between 0 and 1")
        if self.rose_temperature <= 0 or self.rose_smoothing <= 0: raise ValueError("ROSE temperature and smoothing must be positive")
        if not 0 <= self.rose_template_sample_ratio <= 100: raise ValueError("invalid ROSE template ratio")
        if self.rose_reference_selection not in ("uniform", "confidence"): raise ValueError("invalid ROSE reference selection")
