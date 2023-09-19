#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <string>
#include <mpi.h>

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

vector<char> generateRandomSolution(size_t itemCount) {
    vector<char> solution(itemCount);
    for (size_t i = 0; i < itemCount; ++i) {
        solution[i] = rand() % 2;
    }
    return solution;
}

int calculateFitness(const vector<char>& solution, int& totalWeight, const vector<Item>& items) {
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

vector<char> crossover(const vector<char>& parent1, const vector<char>& parent2) {
    vector<char> child(parent1.size());
    int crossoverPoint = rand() % parent1.size();
    for (int i = 0; i < crossoverPoint; ++i) {
        child[i] = parent1[i];
    }
    for (size_t i = crossoverPoint; i < parent2.size(); ++i) {
        child[i] = parent2[i];
    }
    return child;
}

void mutate(vector<char>& solution) {
    for (size_t i = 0; i < solution.size(); ++i) {
        if (static_cast<double>(rand()) / RAND_MAX < MUTATION_RATE) {
            solution[i] = ~solution[i];
        }
    }
}

int main(int argc, char** argv) {

    srand(static_cast<unsigned int>(time(nullptr)));

    size_t itemCount = 77;

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

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<vector<char>> localPopulation(POPULATION_SIZE / size);
    vector<vector<char>> newLocalPopulation(POPULATION_SIZE / size);

    int bestFitness = 0;
    vector<char> bestSolution;
    int bestWeight = 0;

    clock_t start_time = clock();

    for (int generation = 0; generation < GENERATION_COUNT; ++generation) {
        for (int i = 0; i < POPULATION_SIZE / size; ++i) {
            localPopulation[i] = generateRandomSolution(itemCount);
        }

        for (int i = 0; i < POPULATION_SIZE / size; ++i) {
            int parent1Index = rand() % POPULATION_SIZE / size;
            int parent2Index = rand() % POPULATION_SIZE / size;

            vector<char> child = crossover(localPopulation[parent1Index], localPopulation[parent2Index]);

            mutate(child);

            int totalWeight;
            int fitness = calculateFitness(child, totalWeight, items);

            if (fitness > bestFitness) {
                bestFitness = fitness;
                bestSolution = child;
                bestWeight = totalWeight;
            }

            newLocalPopulation[i] = child;
        }

        localPopulation = newLocalPopulation;

        int globalBestFitness;
        MPI_Allreduce(&bestFitness, &globalBestFitness, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        bestFitness = globalBestFitness;

        if (rank == 0) {
            cout << "Generation " << generation << ": Best value = " << bestFitness << ", Best weight = " << bestWeight << endl;
        }
    }

    clock_t end_time = clock();

    double elapsed_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;

    if (rank == 0) {
        cout << "Best solution: ";
        for (size_t i = 0; i < items.size(); ++i) {
            if (bestSolution[i]) {
                cout << items[i].name << " ";
            }
        }
        cout << endl;
        cout << "Elapsed time: " << elapsed_time << " seconds" << endl;
    }

    MPI_Finalize();

    return 0;
}
