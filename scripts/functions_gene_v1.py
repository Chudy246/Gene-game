from functions_basic import *
from numpy.random import default_rng
import random

class Animal:
    # should contain its genome and possibility of changing it - genome as the thing that's causing the change
    def __init__(self, age, genome):
        self.age = age
        self.genome = genome

    def survival(self, kill_factors):
        # check if the Animal is going to survive this turn - this depends on killing factors from the world + probably age

        def sigmoid(x): return 1/(1 + np.exp(-x))

        probabilities = sigmoid(np.array(kill_factors) - np.array(self.genome.genes))
        rng = default_rng()

        return max(probabilities) < rng.uniform()


    def mutate(self):
        # allow for a slight chance for mutations during its life + also including destruction of genes? (maybe later)
        self.genome.genes = self.genome.genes + 0.1* np.random.uniform(-1, 1, size=len(self.genome.genes))
        pass


class Genome:
    # should contain genes
    def __init__(self, genes: list):
        self.genes = genes


class World:
    # should contain all animals and allow for their procreation
    # + the world includes the killing factors which can change with time
    def __init__(self, animals, killing_factors):
        self.animals = animals
        self.killing_factors = killing_factors
        self.time = 0

    def procreate(self):
        for i in range(int(len(self.animals)/5)):
            chosen_animals = random.sample(self.animals, 2)
            rng = default_rng()
            new_genome = [a_genome if rng.uniform() > 0.5 else b_genome for i, (a_genome, b_genome) in enumerate(zip(chosen_animals[0].genome.genes, chosen_animals[1].genome.genes))]
            self.animals.append(Animal(0, Genome(new_genome)))

    def survival_check(self):
        alive_animals = []
        for animal in self.animals:
            if animal.survival(self.killing_factors):
                alive_animals.append(animal)

        self.animals = alive_animals


    def progress_time(self):
        self.time += 1
        for animal in self.animals:
            animal.age += 1
            animal.mutate()

        if self.time % 5== 0:
            self.survival_check()

        self.procreate()
        print(f"Current time: {self.time}")
        print(f"Number of alive animals: {len(self.animals)}")
#         return [animal.age for animal in self.animals], [animal.genome.genes for animal in self.animals]
