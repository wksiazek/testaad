import time
import typing
from dataclasses import dataclass, field

from DET.models.member import Member
from DET.models.population import Population


class MetricHelper:

    @staticmethod
    def calculate_start_metrics(population: Population, log_population: bool = False):
        sorted_members = population.get_best_members(population.size)
        best_inv = sorted_members[0]
        worst_inv = sorted_members[-1]

        # Metrics
        pop_mean = population.mean()
        pop_std = population.std()

        metric = Metric(
            epoch=0,
            best_individual=best_inv,
            worst_individual=worst_inv,
            population_mean=pop_mean,
            population_std=pop_std,
            execution_time=0.0
        )
        if log_population:
            metric.population = population.members

        return metric

    @staticmethod
    def calculate_metrics(population: Population, start_time, epoch, log_population: bool = False):
        sorted_members = population.get_best_members(population.size)
        best_inv = sorted_members[0]
        worst_inv = sorted_members[-1]

        # Metrics
        pop_mean = population.mean()
        pop_std = population.std()

        end_time = time.time()
        execution_time = end_time - start_time

        metric = Metric(
            epoch=epoch + 1,
            best_individual=best_inv,
            worst_individual=worst_inv,
            population_mean=pop_mean,
            population_std=pop_std,
            execution_time=execution_time
        )
        if log_population:
            metric.population = population.members

        return metric


@dataclass
class Metric:
    epoch: int
    best_individual: Member
    worst_individual: Member
    population_mean: float
    population_std: float
    execution_time: float
    population: typing.Optional[typing.List] = field(default_factory=list)
