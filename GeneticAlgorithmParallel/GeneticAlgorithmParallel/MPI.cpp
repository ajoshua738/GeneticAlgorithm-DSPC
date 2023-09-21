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
            solution[i] = !solution[i];
        }
    }
}

int main(int argc, char** argv) {
    srand(static_cast<unsigned int>(time(nullptr)));

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t itemCount = 77; // Define the number of items (adjust as needed)
    vector<Item> items;
    items.reserve(itemCount);

    if (rank == 0) {
        // Initialize items only on rank 0
        for (size_t i = 0; i < itemCount; ++i) {
            Item newItem;
            newItem.name = "t" + to_string(i + 1);
            newItem.weight = i + 1;
            newItem.value = i + 1;
            items.push_back(newItem);
        }

        sort(items.begin(), items.end(), compareItems);
    }

    vector<vector<char>> population(POPULATION_SIZE);

    for (int i = 0; i < POPULATION_SIZE; ++i) {
        population[i] = generateRandomSolution(itemCount);
    }

    const int localPopulationSize = POPULATION_SIZE / size;
    vector<vector<char>> localPopulation(localPopulationSize);

    // Scatter the initial population
    //MPI_Scatter(population.data(), localPopulationSize * itemCount, MPI_CHAR,
    //    localPopulation.data(), localPopulationSize * itemCount, MPI_CHAR, 0, MPI_COMM_WORLD);

    int bestFitness = 0;
    vector<char> bestSolution(itemCount);
    int bestWeight = 0;

    clock_t start_time = clock(); // Record the starting time

    for (int generation = 0; generation < GENERATION_COUNT; ++generation) {
        vector<vector<char>> newLocalPopulation(localPopulationSize);

        // Perform genetic operations on the local population
        for (int i = 0; i < localPopulationSize; ++i) {
            int parent1Index = rand() % localPopulationSize;
            int parent2Index = rand() % localPopulationSize;

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

        // Gather the best solution from all processes
        MPI_Allreduce(MPI_IN_PLACE, &bestFitness, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &bestWeight, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, bestSolution.data(), itemCount, MPI_CHAR, MPI_MAX, MPI_COMM_WORLD);

        if (rank == 0) {
            cout << "Generation " << generation << ": Best value = " << bestFitness << ", Best weight = " << bestWeight << endl;
        }
    }

    clock_t end_time = clock(); // Record the ending time

    double elapsed_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC; // Calculate elapsed time

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

//int main(int argc, char** argv) {
//    //srand(static_cast<unsigned int>(time(nullptr)));
//    srand(0);
//
//    size_t itemCount = 77;
//
//    vector<Item> items;
//    items.reserve(itemCount);
//
//    for (size_t i = 0; i < itemCount; ++i) {
//        Item newItem;
//        newItem.name = "t" + to_string(i + 1);
//        newItem.weight = i + 1;
//        newItem.value = i + 1;
//        items.push_back(newItem);
//    }
//
//    sort(items.begin(), items.end(), compareItems);
//
//    vector<vector<char>> globalPopulation(POPULATION_SIZE);
//
//    for (int i = 0; i < POPULATION_SIZE; ++i) {
//        globalPopulation[i] = generateRandomSolution(itemCount);
//    }
//
//    int bestFitness = 0;
//    vector<char> bestSolution;
//    int bestWeight = 0;
//
//    MPI_Init(&argc, &argv);
//
//    int rank, size;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    MPI_Comm_size(MPI_COMM_WORLD, &size);
//
//    vector<int> sendCounts(size, POPULATION_SIZE / size);
//    vector<int> displacements(size, POPULATION_SIZE / size);
//    for (int i = 0; i < POPULATION_SIZE % size; i++) {
//       sendCounts[i] += 1;
//    }
//    for (int i = 1; i < size; ++i) {
//        displacements[i] = displacements[i - 1] + sendCounts[i - 1];
//    }
//    vector<vector<char>> localPopulation(POPULATION_SIZE / size);
//    localPopulation.resize(sendCounts[rank]);
//
//    clock_t start_time = clock();
//    for (int generation = 0; generation < GENERATION_COUNT; ++generation) {
//        MPI_Barrier(MPI_COMM_WORLD);
//        MPI_Scatterv(globalPopulation.data(), sendCounts.data(), displacements.data(), MPI_CHAR,
//                    localPopulation.data(), sendCounts[rank], MPI_CHAR, 0, MPI_COMM_WORLD);
//        //cout << rank << ", " << localPopulation.size() << endl;
//        for (int i = 0; i < localPopulation.size(); ++i) {
//            int parent1Index = rand() % localPopulation.size();
//            int parent2Index = rand() % localPopulation.size();
//            cout << rank << ", " << parent1Index << ", " << parent2Index << ", " << localPopulation.size() << endl;
//            vector<char> child = crossover(localPopulation[parent1Index], localPopulation[parent2Index]);
//
//            mutate(child);
//
//            int totalWeight;
//            int fitness = calculateFitness(child, totalWeight, items);
//
//            if (fitness > bestFitness) {
//                bestFitness = fitness;
//                bestSolution = child;
//                bestWeight = totalWeight;
//            }
//
//            localPopulation[i] = child;
//        }
//        MPI_Barrier(MPI_COMM_WORLD);
//        MPI_Gatherv(localPopulation.data(), sendCounts[rank], MPI_CHAR,
//            globalPopulation.data(), sendCounts.data(), displacements.data(),
//            MPI_CHAR, 0, MPI_COMM_WORLD);
//
//        int globalBestFitness;
//        MPI_Allreduce(&bestFitness, &globalBestFitness, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
//        bestFitness = globalBestFitness;
//
//        if (rank == 0) {
//            cout << "Generation " << generation << ": Best value = " << bestFitness << ", Best weight = " << bestWeight << endl;
//        }
//    }
//
//    clock_t end_time = clock();
//
//    double elapsed_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;
//
//    if (rank == 0) {
//        cout << "Best solution: ";
//        for (size_t i = 0; i < items.size(); ++i) {
//            if (bestSolution[i]) {
//                cout << items[i].name << " ";
//            }
//        }
//        cout << endl << "Elapsed time: " << elapsed_time << " seconds" << endl;
//    }
//    MPI_Finalize();
//
//    return 0;
//}
