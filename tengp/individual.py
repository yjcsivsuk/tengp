""" Module holding some interesting stuff. """

from abc import ABC, abstractmethod
from functools import reduce

import numpy as np

from .genotype_factory import GenotypeFactory
from .utils import map_to_np_phenotype, active_paths, join_lists


class Individual(ABC):
    """ Individual class """


    def __init__(self, genes, bounds, params):
        """
        """
        self.fitness = None
        self.genes = genes
        self.bounds = bounds
        self.paths = active_paths(self.nodes)
        self.active_nodes = set(reduce(join_lists, self.paths))
        self.params = params

    def __eq__(self, other):
        if len(self.active_nodes) != len(other.active_nodes):
            return False

        for me, them in zip(self.active_nodes, other.active_nodes):
            if self.nodes[me].fun != other.nodes[them].fun:
                return False

            if self.nodes[me].inputs != other.nodes[them].inputs:
                return False

        return True

    def __repr__(self):
        return f'Program, f:{self.fitness}'

    @abstractmethod
    def transform(self, X):
        pass

    def active_gene(self, gene_index):
        """ Checks, whether given index is index of a active gene """
        arity = self.params.function_set.max_arity
        if gene_index >= self.params.n_nodes*(arity + 1):
            return True
        node_id = (gene_index//(arity + 1))+self.params.n_inputs
        if node_id in self.active_nodes:
            return True
        return False

    # 获取活跃路径上的基因型的id
    def get_active_genes(self):
        """ Return all active genes. """
        active_genes = []
        arity = self.params.function_set.max_arity
        for node in self.active_nodes:
            if self.nodes[node].is_input:
                continue

            if self.nodes[node].is_output:
                gindex = self.params.n_nodes * arity + node - self.params.n_inputs
                active_genes.append(gindex)
                continue

            start_index = (arity+1)*(node-self.params.n_inputs)
            active_genes += range(start_index, start_index + arity + 1)
        return active_genes

    def get_expression(self):
        """ Return string representation of expression (phenotype)."""
        stack = []
        result = []

        for path in self.paths:
            for node in path:
                current_node = self.nodes[node]

                if current_node.is_output:
                    result.append(stack.pop())
                elif current_node.is_input:
                    stack.append(f'x{node}')
                else:
                    operands = [stack.pop() for _ in range(0, current_node.arity)]
                    stack.append('{}({})'.format(current_node.fun.__name__, ','.join(operands)))

        return result


    def apply(self, move):
        """ Return new individual, as a result of applying given move to current individual."""
        genes = self.genes[:]

        for index, value in zip(move.indicies, move.changes):
            genes[index] = value

        return self.params.individual_class(genes, self.bounds, self.params)


class NPIndividual(Individual):

    def __init__(self, genes, bounds, params):
        self.nodes = map_to_np_phenotype(genes, params)
        Individual.__init__(self, genes, bounds, params)

    def transform(self, X):
        """Transforms the input data with expression encoded in individual.

        Args:
            X (array-like): 2D Numpy array, or tensor (if use_tensors was set to true in Parameters)

        Returns:
            Transformed data. If use_tensors was set to true, then list
            containing output tensors is returned. Otherwise Numpy array
            is returned.
        """
        n_dims = X.dim() if X.__class__.__name__ == 'Tensor' else X.ndim
        if n_dims == 0 or n_dims == 1:
            raise ValueError(
                    "Expected 2D array, got scalar or 1D instead."
                    "If X is single sample, use array.reshape(1, -1)."
                    "If X has single feature, use array.reshape(-1, 1).")

        for path in self.paths:
            for index in path:
                current_node = self.nodes[index]

                if current_node.is_input: # is input node 
                    current_node.value = X[:, index]
                elif current_node.is_output:
                    input_index = current_node.inputs[0]
                    current_node.value = self.nodes[input_index].value
                else:
                    values = [self.nodes[i].value for i in current_node.inputs[:current_node.arity]]
                    current_node.value = current_node.fun(*values)

        output = []
        for i in range(1, self.params.n_outputs + 1):
            output.append(self.nodes[-i].value)

        if self.params.use_tensors:
            # for now
            return output
        else:
            return np.array(output).T


class IndividualBuilder():
    def __init__(self, params):
        self.params = params
        self.g_factory = GenotypeFactory(params)

    def create(self):
        genes, bounds = self.g_factory.create()

        return self.params.individual_class(genes, bounds, self.params)


