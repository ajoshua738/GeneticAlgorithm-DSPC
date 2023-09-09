#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <omp.h>

const int numItems = 10;
const int knapsackCapacity = 100;
const int populationSize = 750;
const int numGenerations = 750;
const double mutationRate = 0.1;

struct Item {
    int value;
    int weight;
};

std::vector<Item> initializeItems() {
    std::vector<Item> items(numItems);

    for (int i = 0; i < numItems; ++i) {
        items[i].value = 100 + i;    // Random value between 1 and 100
        items[i].weight = 10 + i + i * i;   // Random weight between 1 and 50
    }

    return items;
}

void initializePopulation(std::vector<std::vector<int>>& population) {
    for (int i = 0; i < populationSize; ++i) {
        std::vector<int> individual(numItems);
        for (int j = 0; j < numItems; ++j) {
            individual[j] = rand() % 2; // Binary representation
        }
        population.push_back(individual);
    }
}

int evaluateFitness(const std::vector<int>& individual, const std::vector<Item>& items) {
    int totalValue = 0;
    int totalWeight = 0;
    for (int i = 0; i < numItems; ++i) {
        if (individual[i] == 1) {
            totalValue += items[i].value;
            totalWeight += items[i].weight;
        }
    }
    if (totalWeight > knapsackCapacity) {
        return 0; // Penalize solutions exceeding capacity
    }
    return totalValue;
}

std::vector<int> crossover(const std::vector<int>& parent1, const std::vector<int>& parent2) {
    int crossoverPoint = rand() % numItems;
    std::vector<int> child(numItems);
    for (int i = 0; i < crossoverPoint; ++i) {
        child[i] = parent1[i];
    }
    for (int i = crossoverPoint; i < numItems; ++i) {
        child[i] = parent2[i];
    }
    return child;
}

void mutate(std::vector<int>& individual) {
    for (int i = 0; i < numItems; ++i) {
        if (static_cast<double>(rand()) / RAND_MAX < mutationRate) {
            individual[i] = 1 - individual[i]; // Flip the bit
        }
    }
}

void selectParents(const std::vector<std::vector<int>>& population, const std::vector<Item>& items, std::vector<int>& parent1, std::vector<int>& parent2) {
    const int tournamentSize = 5; // Size of the tournament
    int populationSize = population.size();

    for (int t = 0; t < 2; ++t) {
        int bestFitness = -1;
        int bestIndex = -1;
        for (int i = 0; i < tournamentSize; ++i) {
            int randomIndex = rand() % populationSize;
            int fitness = evaluateFitness(population[randomIndex], items);

            if (fitness > bestFitness) {
                bestFitness = fitness;
                bestIndex = randomIndex;
            }
        }

        if (t == 0) {
            parent1 = population[bestIndex];
        }
        else {
            parent2 = population[bestIndex];
        }
    }
}



void replaceLeastFit(std::vector<std::vector<int>>& population, const std::vector<int>& child, const std::vector<Item>& items) {
    int leastFitIndex = 0;
    int leastFitFitness = evaluateFitness(population[0], items);

    for (int i = 1; i < populationSize; ++i) {
        int currentFitness = evaluateFitness(population[i], items);
        if (currentFitness < leastFitFitness) {
            leastFitFitness = currentFitness;
            leastFitIndex = i;
        }
    }

    population[leastFitIndex] = child;
}

int main() {
    srand(static_cast<unsigned int>(time(nullptr)));

    std::vector<Item> items = initializeItems();

    std::vector<std::vector<int>> population;
    initializePopulation(population);
    std::vector<std::vector<int>> bestChildren(populationSize);
    double start_time = omp_get_wtime();
#pragma omp parallel for
    for (int generation = 0; generation < numGenerations; ++generation) {
#pragma omp parallel for
        for (int i = 0; i < populationSize; ++i) {
            int fitness = evaluateFitness(population[i], items);

            std::vector<int> parent1, parent2;
            selectParents(population, items, parent1, parent2);

            std::vector<int> child = crossover(parent1, parent2);
            mutate(child);
            bestChildren[i] = child;
        }
#pragma omp parallel for
        for (int i = 0; i < populationSize; ++i) {
#pragma omp critical
            {
                replaceLeastFit(population, bestChildren[i], items);
            }
        }
    }

    double end_time = omp_get_wtime(); // Stop measuring time


    int bestFitness = evaluateFitness(population[0], items);
    int bestIndex = 0;
    for (int i = 1; i < populationSize; ++i) {
        int fitness = evaluateFitness(population[i], items);
        if (fitness > bestFitness) {
            bestFitness = fitness;
            bestIndex = i;
        }
    }

    std::cout << "Best solution found:" << std::endl;
    std::cout << "Fitness: " << bestFitness << std::endl;
    std::cout << "Items selected: ";
    for (int i = 0; i < numItems; ++i) {
        if (population[bestIndex][i] == 1) {
            std::cout << i << " ";
        }
    }
    std::cout << std::endl;


    std::cout << "\nItems and their fitness values:" << std::endl;
    for (int i = 0; i < numItems; ++i) {
        std::cout << "Item " << i << ": Value = " << items[i].value << ", Weight = " << items[i].weight
            << std::endl;
    }

    std::cout << "Time taken: " << (end_time - start_time) << " seconds" << std::endl;

    return 0;

}