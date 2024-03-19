"""Search strategies."""

from array import array
from collections import deque
import random
from functools import partial

import numpy as np

from .mutations import MUTATIONS
from .individual import IndividualBuilder
from .parameters import FunctionSet, Parameters
from .utils import handle_invalid_decorator, UnknownMutationException

@handle_invalid_decorator
def simple_es(X, y, cost_function, params,
              target_fitness=None,
              population_size=5,
              evaluations=5000,
              random_state=None,
              mutation='point',
              mutation_probability=0.25,
              verbose=False,
              log=None,
              seed_individual=None):
    """Optimize a CGP system using a simple evolutionary strategy.

    Args:
        X (numpy.ndarray): input data, number of columns have to match the n_inputs
            parameter of Parameters object
        y (numpy.ndarray): target output data, number of columns have to match
            the n_outputs parameter of Parameters object
        cost_function (callable): cost function to minimize. It has two
            arguments `(y_true, y_pred)`, where `y_true` is target output data
            and `y_pred` is output of CGP individual
        params (Parameters): instance of Parameters class
        target_fitness (number or None): fitness, at which evolution will stop. If None, it is not considered
        population_size (int): size of population including parent
        evaluations (int): maximum number of cost function evaluations
        random_state (int): seed for random number generator
        mutation (string): type of mutation to use, accept values `'point',
            'active', 'single', 'probabilistic'`
        mutation_probability (float): probability of mutating a given gene (
            used only when `mutation` argument is set to `'probabilistic'`
        verbose (bool): if `True`, outputs evolution info every 100 generations
        log (list): if provided with a list, best fitness of each generation
            is stored here
        seed_individual (Individual): if provided with instance of Individual class,
            the initial population is created according to this object - parent of first
            generation.

    Returns:
        List of individuals, the last generation of evolution
    """

    if mutation not in MUTATIONS:
        raise UnknownMutationException("Provided type of mutation is not implemented.")

    move = MUTATIONS[mutation]
    if mutation == 'probabilistic':
        move = partial(move, probability=mutation_probability)

    if random_state:
        random.seed(random_state)

    # initial generation
    # 初始化种群
    ib = IndividualBuilder(params)

    if seed_individual:
        population = [seed_individual.apply(move(seed_individual)) for _ in range(population_size - 1)]
        population += [seed_individual]
    else:
        population = [ib.create() for _ in range(population_size)]

    n_evals = 0  # 迭代次数

    generation = 0  # 种群代数

    # 计算fitness，评估种群中的个体
    for individual in population:
        # 如果cf_individual设置为true，计算cost_function的时候就用真实值和个体作为参数；如果为false，就用真实值和经过transform()后的输出值作为参数
        if params.cf_individual:
            individual.fitness = cost_function(y, individual)
        else:
            output = individual.transform(X)  # X是一个2维数组
            individual.fitness = cost_function(y, output)
        n_evals += 1

    # 迭代种群，每次循环都生成新的一代种群generation
    while n_evals <= evaluations:
        generation += 1

        parent = min(population, key=lambda x: x.fitness)  # 从当前种群中选择适应度最高（fitness最小）的个体作为父代

        if log is not None:
            log.append(parent.fitness)

        if target_fitness is not None and parent.fitness <= target_fitness:
            population.sort(key=lambda x: x.fitness)
            return population

        population = [parent.apply(move(parent)) for _ in range(population_size - 1)]  # 对父代进行突变操作move，生成新一代种群。move()是突变操作，apply()和move()是绑定的

        population += [parent]  #这里应该就是1+λ

        # 评估新一代个体的fitness
        for individual in population:
            if params.cf_individual:
                individual.fitness = cost_function(y, individual)
            else:
                output = individual.transform(X)
                individual.fitness = cost_function(y, output)
            n_evals += 1

        # 输出进化信息
        if verbose and generation % verbose == 0:
            print(f'Gen: {generation}, population: {sorted([x.fitness for x in population])}')
            print(f"Best individual: {population[0].get_expression()}")  # 新增一条输出最好个体的表达式

    # 返回最后一代种群中的最佳个体
    population.sort(key=lambda x: x.fitness)

    if log is not None:
        log.append(population[0].fitness)

    return population

