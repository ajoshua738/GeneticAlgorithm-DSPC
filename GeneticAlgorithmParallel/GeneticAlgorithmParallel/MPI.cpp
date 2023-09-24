//#include <iostream>
//#include <vector>
//#include <algorithm>
//#include <ctime>
//#include <cstdlib>
//#include <string>
//#include <mpi.h>
//#include <bitset>
//
//using namespace std;
//
//struct Item {
//    string name;
//    int weight;
//    int value;
//};
//
//const int POPULATION_SIZE = 5000;
//const int GENERATION_COUNT = 100;
//const double MUTATION_RATE = 0.1;
//const int KNAPSACK_CAPACITY = 3000;
//
//bool compareItems(const Item& item1, const Item& item2) {
//    return (static_cast<double>(item1.value) / item1.weight) > (static_cast<double>(item2.value) / item2.weight);
//}
//
//vector<char> generateRandomSolution(size_t itemCount) {
//    vector<char> solution(itemCount);
//    for (size_t i = 0; i < itemCount; ++i) {
//        solution[i] = rand() % 2;
//    }
//    return solution;
//}
//
//int calculateFitness(const vector<char>& solution, int& totalWeight, const vector<Item>& items) {
//    int totalValue = 0;
//    totalWeight = 0;
//    for (size_t i = 0; i < solution.size(); ++i) {
//        if (solution[i]) {
//            totalValue += items[i].value;
//            totalWeight += items[i].weight;
//        }
//    }
//    if (totalWeight > KNAPSACK_CAPACITY) {
//        return 0;
//    }
//    return totalValue;
//}
//
//vector<char> crossover(const vector<char>& parent1, const vector<char>& parent2) {
//    vector<char> child(parent1.size());
//    int crossoverPoint = rand() % parent1.size();
//    for (int i = 0; i < crossoverPoint; ++i) {
//        child[i] = parent1[i];
//    }
//    for (size_t i = crossoverPoint; i < parent2.size(); ++i) {
//        child[i] = parent2[i];
//    }
//    return child;
//}
//
//void mutate(vector<char>& solution) {
//    for (size_t i = 0; i < solution.size(); ++i) {
//        if (static_cast<double>(rand()) / RAND_MAX < MUTATION_RATE) {
//            solution[i] = !solution[i];
//        }
//    }
//}
//
//vector<char> flatten(vector<vector<char>> unflatVector, size_t itemCount) {
//    vector<char> result;
//
//    for (const vector<char>& innerVector : unflatVector) {
//        for (size_t i = 0; i < itemCount; i++) {
//            result.push_back(innerVector[i]);
//        }
//    }
//
//    return result;
//}
//
//
//vector<vector<char>> unflatten(vector<char> flatVector, size_t itemCount) {
//    vector<vector<char>> result;
//
//    size_t totalItems = flatVector.size();
//    size_t currentIndex = 0;
//
//    while (currentIndex < totalItems) {
//        vector<char> innerVector;
//
//        // Populate the inner vector with itemCount items or until the end of the flatVector
//        for (size_t i = 0; i < itemCount && currentIndex < totalItems; i++) {
//            innerVector.push_back(flatVector[currentIndex]);
//            currentIndex++;
//        }
//
//        result.push_back(innerVector);
//    }
//
//    return result;
//}
//
//
//int main(int argc, char** argv) {
//    srand(static_cast<unsigned int>(time(nullptr)));
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
//    vector<char> flatGlobalPopulation = flatten(globalPopulation, itemCount);
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
//    int localPopulationSize = POPULATION_SIZE / size;
//    int flatLocalPopulationSize = POPULATION_SIZE / size * itemCount;
//
//    vector<int> sendCounts(size, flatLocalPopulationSize);
//    vector<int> displacements(size, 0);
//    for (int i = 0; i < POPULATION_SIZE % size; i++) {
//        sendCounts[i] += itemCount;
//    }
//    for (int i = 1; i < size; ++i) {
//        displacements[i] = displacements[i - 1] + sendCounts[i - 1];
//    }
//    vector<vector<char>> localPopulation(localPopulationSize, vector<char>(itemCount));
//    localPopulation.resize(sendCounts[rank] / itemCount);
//    vector<char> flatLocalPopulation(flatLocalPopulationSize);
//    flatLocalPopulation.resize(sendCounts[rank]);
//    clock_t start_time = clock();
//
//    for (int generation = 0; generation < GENERATION_COUNT; ++generation) {
//        MPI_Barrier(MPI_COMM_WORLD);
//
//        MPI_Scatterv(flatGlobalPopulation.data(), sendCounts.data(), displacements.data(), MPI_CHAR,
//            flatLocalPopulation.data(), sendCounts[rank], MPI_CHAR, 0, MPI_COMM_WORLD);
//        localPopulation = unflatten(flatLocalPopulation, itemCount);
//        for (int i = 0; i < localPopulationSize; ++i) {
//            int parent1Index = rand() % localPopulationSize;
//            int parent2Index = rand() % localPopulationSize;
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
//        flatLocalPopulation = flatten(localPopulation, itemCount);
//
//        MPI_Barrier(MPI_COMM_WORLD);
//        MPI_Gatherv(flatLocalPopulation.data(), sendCounts[rank], MPI_CHAR,
//            flatGlobalPopulation.data(), sendCounts.data(), displacements.data(),
//            MPI_CHAR, 0, MPI_COMM_WORLD);
//        globalPopulation = unflatten(flatGlobalPopulation, itemCount);
//
//        int globalBestFitness, globalBestWeight;
//        MPI_Allreduce(&bestFitness, &globalBestFitness, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
//        MPI_Allreduce(&bestWeight, &globalBestWeight, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
//        MPI_Barrier(MPI_COMM_WORLD);
//        bestFitness = globalBestFitness;
//        bestWeight = globalBestWeight;
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
//        cout << endl;
//        cout << "Elapsed time: " << elapsed_time << " seconds" << endl;
//    }
//
//    MPI_Finalize();
//
//    return 0;
//}