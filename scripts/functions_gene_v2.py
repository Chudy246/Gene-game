import numpy as np

from functions_basic import *
from collections import deque


class World:
    def __init__(self, animal_alive: pd.Series, animal_age: pd.Series, animal_genes: pd.DataFrame,
                 kill_factors: pd.Series, proportion_mating=0.2, mult_factor=10, food_multiplier=0.2,
                 printout=False):
        """
        animal_alive -> pd.Series with True for alive animals and False for dead animals
        animal_age -> pd.Series of integers starting from zero
        animal_genes -> DataFrame containing numerical values of genes w.r.t. kill_factors, each row is another animal
        kill_factors -> pd.Series of length equal to columns of animal_genes,
                        describes numerical factors determining survivability of animals from the environment
        """
        # initiate variables in the World
        self.animal_alive = animal_alive.copy()
        self.animal_age = animal_age.copy()
        self.animal_genes = animal_genes.copy()
        self.time = 0
        self.kill_factors = kill_factors
        self.animal_number = animal_alive.sum()
        self.max_pop = len(animal_alive)  # maximum length of the array
        self.PROPORTION_MATING = proportion_mating
        self.MULT_FACTOR = mult_factor
        self.FOOD_MULTIPLIER = food_multiplier
        self.PRINTOUT = printout

        # stack for tracking dead animals
        dead_stack = deque()
        dead_animals = animal_alive.apply(lambda x: not x)
        dead_animals = dead_animals[
            dead_animals].index.values  # since dead animals is a boolean array we can obtain index values
        for i in range(len(dead_animals)):
            dead_stack.append(dead_animals[-1 - i])

        self.dead_stack = dead_stack

    def mutate(self):
        """
        Mutate genes of alive animals, use alive animals as a mask
        """
        # random gene change of maximum of 0.1 in either direction
        gene_change = 0.1 * np.random.uniform(-1, 1, size=self.animal_genes.shape)
        mask = self.animal_alive.apply(lambda x: int(x))
        self.animal_genes += (gene_change.T * mask.values).T  # adding change only to alive animals

    def survival(self):
        """
        Check for which animals are killed, update the dead_stack and alive animals
        """
        # algorithm for death from external factors - each death factor assumed independent of others
        odds = []
        for i in range(len(self.kill_factors)):
            odds.append(self.kill_factors[i] - self.animal_genes[i])

        # modified sigmoid as a way to convert odds to probabilities
        def sigmoid(x): return 1 / (1 + self.MULT_FACTOR * np.exp(-x))

        probabilities = [sigmoid(odd) for odd in odds]

        alive = [np.random.uniform(0, 1, size=len(probability)) > probability for probability in probabilities]
        mask = self.animal_alive
        alive = [cause & mask for cause in alive]

        # death from lack of food - depends on population size
        survival_probability = 1 - self.FOOD_MULTIPLIER * self.animal_number / self.max_pop
        food_allowed = np.random.uniform(0, 1, size=self.max_pop) < survival_probability
        food_allowed = food_allowed & mask

        alive.append(food_allowed)

        alive = pd.Series(np.logical_and.reduce(alive))  # combining the results using 'and' operator

        # update the animal_alive and dead_stack - bookkeeping
        new_dead = np.logical_xor(self.animal_alive, alive)
        self.animal_alive = alive
        self.animal_age = self.animal_age * alive.apply(lambda x: int(x))  # multiply by zero if dead, by 1 if alive
        self.animal_genes[self.animal_genes.columns] = (self.animal_genes.values.T * alive.apply(lambda x: int(x)).values).T

        new_dead = new_dead[new_dead].index.values
        self.animal_number -= len(new_dead)
        if self.PRINTOUT:
            print(f"{len(new_dead)} animals died this turn")

        for i in range(len(new_dead)):
            self.dead_stack.append(new_dead[-1 - i])

    def procreate(self):
        """
        Create new animals out of mating of existing ones, update the dead_stack and alive animals
        """
        # obtaining index of alive animals (index for which animal_alive is True)
        alive = self.animal_alive[self.animal_alive].index.values
        # updating animal number
        new_animals_number = int(self.PROPORTION_MATING * len(alive))
        self.animal_number += new_animals_number

        if self.PRINTOUT:
            print(f"{new_animals_number} animals were born this turn")

        # creating new animals by random combination of genes from other animals
        # TODO: change for loop to a combined operation for updating the main matrix
        for i in range(new_animals_number):
            chosen_animals = np.random.choice(alive, 2)
            a_genome, b_genome = (self.animal_genes.iloc[chosen_animals[0], :],
                                  self.animal_genes.iloc[chosen_animals[1], :])

            new_genome = np.where(np.random.uniform(0, 1, size=len(a_genome)) > 0.5, a_genome, b_genome)

            # update the main matrix
            index = self.dead_stack.pop()
            self.animal_age[index] = 0
            self.animal_alive[index] = True
            self.animal_genes.iloc[index, :] = new_genome

    def statistics(self, return_index=False):
        """
        :return: statistics on ages and genes in the current population,
                 stats include median, minimum, quartilees and maximum + mean and std
        """
        # returning the index
        if return_index:
            stat_list = ["_median", "_min", "_025", "_075", "_max", "_mean", "_std"]
            age_list = ["age" + stat for stat in stat_list]
            gene_list = [[f"gene_{column}" + stat for stat in stat_list] for column in self.animal_genes.columns]

            return_list = ["time", "animal_number"]
            return_list += [f"death_factor_{column}" for column in self.animal_genes.columns]
            return_list += age_list
            for gene in gene_list:
                return_list += gene

            return return_list

        # returning stats
        age, genes = (self.animal_age[self.animal_alive],
                      [self.animal_genes[i][self.animal_alive] for i in self.animal_genes.columns]
                      )

        def stats(array):
            """

            :param array:
            :return: median, min, 0.25 quantile, median, 0.75 quantile, max, mean and std of the array
            """
            return [np.median(array), np.min(array), np.quantile(array, 0.25), np.quantile(array, 0.75), np.max(array),
                    np.mean(array), np.std(array)]

        # include number of animals + death factors -> most useful later e.g. with oscillatory diseases
        return_list = [self.time, self.animal_number]
        return_list += [death_factor for death_factor in self.kill_factors]
        return_list += stats(age)
        for gene in genes:
            return_list += stats(gene)

        return return_list

    def progress_time(self):
        self.time += 1
        if self.PRINTOUT:
            print(f"Turn: {self.time}")
        self.animal_age += self.animal_alive.apply(lambda x: int(x))
        if self.PRINTOUT:
            print(f"Average animal age is {np.sum(self.animal_age) / self.animal_number}")
            print(f"Average animal age is {np.mean(self.animal_age[self.animal_alive])}")

        self.mutate()
        self.survival()
        self.procreate()

        if self.PRINTOUT:
            print(f"The total number of animals is {self.animal_number}")
