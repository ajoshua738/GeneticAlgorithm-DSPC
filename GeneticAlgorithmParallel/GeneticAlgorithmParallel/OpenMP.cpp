#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <string>
#include <omp.h> // Include the OpenMP header

using namespace std;

struct Item {
    string name;
    int weight;
    int value;
};

const int POPULATION_SIZE = 5000;
const int GENERATION_COUNT = 100;
const double MUTATION_RATE = 0.1;
const int KNAPSACK_CAPACITY = 3000;

bool compareItems(const Item& item1, const Item& item2) {
    return (static_cast<double>(item1.value) / item1.weight) > (static_cast<double>(item2.value) / item2.weight);
}

vector<bool> generateRandomSolution(size_t itemCount) {
    vector<bool> solution(itemCount);
    for (size_t i = 0; i < itemCount; ++i) {
        solution[i] = rand() % 2;
    }
    return solution;
}

int calculateFitness(const vector<bool>& solution, int& totalWeight, const vector<Item>& items) {
    int totalValue = 0;
    totalWeight = 0;
    for (size_t i = 0; i < solution.size(); ++i) {
        if (solution[i]) {
            totalValue += items[i].value;
            totalWeight += items[i].weight;
        }
    }
    if (totalWeight > KNAPSACK_CAPACITY) {
        return 0;
    }
    return totalValue;
}

vector<bool> crossover(const vector<bool>& parent1, const vector<bool>& parent2) {
    vector<bool> child(parent1.size());
    int crossoverPoint = rand() % parent1.size();

    // Use #pragma omp critical to protect shared data access
#pragma omp parallel for
    for (int i = 0; i < crossoverPoint; ++i) {
        child[i] = parent1[i];
    }
    for (size_t i = crossoverPoint; i < parent2.size(); ++i) {
        child[i] = parent2[i];
    }
    return child;
}

void mutate(vector<bool>& solution) {
    for (size_t i = 0; i < solution.size(); ++i) {
        if (static_cast<double>(rand()) / RAND_MAX < MUTATION_RATE) {
            solution[i] = !solution[i];
        }
    }
}

int main() {
    srand(static_cast<unsigned int>(time(nullptr)));

    size_t itemCount = 77; // Define the number of items (adjust as needed)

    vector<Item> items;
    items.reserve(itemCount);

    for (size_t i = 0; i < itemCount; ++i) {
        Item newItem;
        newItem.name = "t" + to_string(i + 1);
        newItem.weight = i + 1;
        newItem.value = i + 1;
        items.push_back(newItem);
    }

    sort(items.begin(), items.end(), compareItems);

    vector<vector<bool>> population(POPULATION_SIZE);

    for (int i = 0; i < POPULATION_SIZE; ++i) {
        population[i] = generateRandomSolution(itemCount);
    }

    int bestFitness = 0;
    vector<bool> bestSolution;
    int bestWeight = 0;

    clock_t start_time = clock(); // Record the starting time

    omp_set_num_threads(4); // Adjust as needed

#pragma omp parallel
    {
        vector<vector<bool>> localPopulation(POPULATION_SIZE); // Each thread has its own population

        int localBestFitness = 0;
        vector<bool> localBestSolution(itemCount);
        int localBestWeight = 0;

        for (int generation = 0; generation < GENERATION_COUNT; ++generation) {
            vector<vector<bool>> newPopulation(POPULATION_SIZE);

#pragma omp for
            for (int i = 0; i < POPULATION_SIZE; ++i) {
                int parent1Index, parent2Index;
                do {
                    parent1Index = rand() % POPULATION_SIZE;
                    parent2Index = rand() % POPULATION_SIZE;
                } while (parent1Index == parent2Index);

                vector<bool> child = crossover(population[parent1Index], population[parent2Index]);

#pragma omp critical
                {
                    mutate(child);
                    int totalWeight;
                    int fitness = calculateFitness(child, totalWeight, items);

                    if (fitness > localBestFitness) {
                        localBestFitness = fitness;
                        localBestSolution = child;
                        localBestWeight = totalWeight;
                    }

                    newPopulation[i] = child;
                }
            }

            // Use reduction clause to find global best fitness and solution
#pragma omp critical
            {
                if (localBestFitness > bestFitness) {
                    bestFitness = localBestFitness;
                    bestSolution = localBestSolution;
                    bestWeight = localBestWeight;
                }
            }

#pragma omp barrier
#pragma omp single
            {
                localPopulation = newPopulation; // Update localPopulation outside the parallel region
                cout << "Generation " << generation << ": Best value = " << bestFitness << ", Best weight = " << bestWeight << endl;
            }
        }
    }

    clock_t end_time = clock(); // Record the ending time

    double elapsed_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC; // Calculate elapsed time

    cout << "Best solution: ";
    for (size_t i = 0; i < items.size(); ++i) {
        if (bestSolution[i]) {
            cout << items[i].name << " ";
        }
    }

    cout << endl;
    cout << "Elapsed time: " << elapsed_time << " seconds" << endl;

    return 0;
}
