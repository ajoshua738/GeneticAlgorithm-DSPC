#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <string>
#include <mpi.h>
//SERIAL WITH GENERATED ITEMS
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

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(static_cast<unsigned int>(time(nullptr))); // Seed the random number generator

    size_t itemCount = 77; // Define the number of items (adjust as needed)
    vector<Item> items;
    vector<vector<bool>> subPopulation;

    if (rank == 0) {
        items.reserve(itemCount);
        // Generate items on process 0
        for (size_t i = 0; i < itemCount; ++i) {
            Item newItem;
            newItem.name = "t" + to_string(i + 1);
            newItem.weight = i + 1;
            newItem.value = i + 1;
            items.push_back(newItem);
        }

        sort(items.begin(), items.end(), compareItems);
    }

    // Broadcast item count and items to all processes
    MPI_Bcast(&itemCount, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        items.resize(itemCount);
    }
    MPI_Bcast(items.data(), itemCount * sizeof(Item), MPI_BYTE, 0, MPI_COMM_WORLD);

    int subPopulationSize = POPULATION_SIZE / size;
    subPopulation.resize(subPopulationSize);

    for (int i = 0; i < subPopulationSize; ++i) {
        subPopulation[i] = generateRandomSolution(itemCount);
    }

    int bestFitness = 0;
    vector<bool> bestSolution;
    int bestWeight = 0;

    clock_t start_time = clock(); // Record the starting time

    for (int generation = 0; generation < GENERATION_COUNT; ++generation) {
        vector<vector<bool>> newSubPopulation(subPopulationSize);

        for (int i = 0; i < subPopulationSize; ++i) {
            int parent1Index = rand() % subPopulationSize;
            int parent2Index = rand() % subPopulationSize;

            vector<bool> child = crossover(subPopulation[parent1Index], subPopulation[parent2Index]);

            mutate(child);

            int totalWeight;
            int fitness = calculateFitness(child, totalWeight, items);

            if (fitness > bestFitness) {
                bestFitness = fitness;
                bestSolution = child;
                bestWeight = totalWeight;
            }

            newSubPopulation[i] = child;
        }

        subPopulation = newSubPopulation;

        if (rank == 0) {
            int globalBestFitness;
            int globalBestWeight;
            MPI_Reduce(&bestFitness, &globalBestFitness, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&bestWeight, &globalBestWeight, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                cout << "Generation " << generation << ": Best value = " << globalBestFitness << ", Best weight = " << globalBestWeight << endl;
            }
        }
    }

    clock_t end_time = clock(); // Record the ending time

    double elapsed_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC; // Calculate elapsed time

    if (rank == 0) {
        int* globalBestSolution = new int[itemCount]; // Dynamically allocate memory
        for (size_t i = 0; i < itemCount; ++i) {
            globalBestSolution[i] = (bestSolution[i]) ? 1 : 0;
        }
        int* recvBestSolution = new int[itemCount]; // Dynamically allocate memory
        MPI_Reduce(globalBestSolution, recvBestSolution, itemCount, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            cout << "Best solution: ";
            for (size_t i = 0; i < itemCount; ++i) {
                if (recvBestSolution[i] > 0) {
                    cout << items[i].name << " ";
                }
            }
            cout << endl;
            cout << "Elapsed time: " << elapsed_time << " seconds" << endl;
        }

        delete[] globalBestSolution; // Free allocated memory
        delete[] recvBestSolution;   // Free allocated memory
    }


    MPI_Finalize();

    return 0;
}