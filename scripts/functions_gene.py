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

        self.animal_alive = animal_alive
        self.animal_age = animal_age
        self.animal_genes = animal_genes
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
        gene_change = 0.1 * np.random.uniform(-1, 1, size=self.animal_genes.shape)
        mask = self.animal_alive.apply(lambda x: int(x))
        self.animal_genes += (gene_change.T * mask.values).T  # adding change only to alive animals

    def survival(self):
        """
        Check for which animals are killed, update the dead_stack and alive animals
        """
        # algorithm for death from external factors
        odds = []
        for i in range(len(self.kill_factors)):
            odds.append(self.kill_factors[i] - self.animal_genes[i])

        def sigmoid(x): return 1 / (1 + self.MULT_FACTOR * np.exp(-x))

        probabilities = [sigmoid(odd) for odd in odds]

        alive = [np.random.uniform(0, 1, size=len(probability)) > probability for probability in probabilities]
        mask = self.animal_alive
        alive = [cause & mask for cause in alive]  # converts dead from boolean to int

        # death from lack of food - depends on population size
        survival_probability = 1 - self.FOOD_MULTIPLIER * self.animal_number / self.max_pop
        food_allowed = np.random.uniform(0, 1, size=self.max_pop) < survival_probability
        food_allowed = food_allowed & mask

        alive.append(food_allowed)

        alive = pd.Series(np.logical_and.reduce(alive))

        # update the animal_alive and dead_stack - bookkeeping
        new_dead = np.logical_xor(self.animal_alive, alive)
        self.animal_alive = alive
        self.animal_age = self.animal_age * alive.apply(lambda x: int(x))
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
        alive = self.animal_alive[self.animal_alive].index.values
        new_animals_number = int(self.PROPORTION_MATING * len(alive))
        self.animal_number += new_animals_number

        if self.PRINTOUT:
            print(f"{new_animals_number} animals were born this turn")

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

    def statistics(self):
        age, genes = (self.animal_age[self.animal_alive],
                      [self.animal_genes[i][self.animal_alive] for i in self.animal_genes.columns]
                      )
        def stats(array):
            return [np.median(array), np.min(array), np.quantile(array, 0.25), np.quantile(array, 0.75), np.max(array),
                    np.mean(array), np.std(array)]

        return_list = stats(age)
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
