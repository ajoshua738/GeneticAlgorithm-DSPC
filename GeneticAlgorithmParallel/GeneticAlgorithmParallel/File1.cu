//#include <iostream>
//#include <vector>
//#include <algorithm>
//#include <ctime>
//#include <cstdlib>
//#include <string>
//#include <cuda_runtime.h>
//#include "device_launch_parameters.h"
//
//using namespace std;
//
//struct Item {
//    string name;
//    int weight;
//    int value;
//};
//
//const int POPULATION_SIZE = 50000;
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
////101010
////1010100
//
//__global__ void calculateFitnessKernel(const char* d_solution, int* d_totalWeight, const Item* d_items, int* d_fitness, const int itemCount) {
//    int tid = blockIdx.x * blockDim.x + threadIdx.x;
//    if (tid < POPULATION_SIZE) {
//        int totalValue = 0;
//        int totalWeight = 0;
//        for (int i = 0; i < itemCount; ++i) {
//            if (d_solution[tid * itemCount + i]) {
//                totalValue += d_items[i].value;
//                totalWeight += d_items[i].weight;
//            }
//        }
//        if (totalWeight > KNAPSACK_CAPACITY) {
//            d_fitness[tid] = 0;
//        }
//        else {
//            d_fitness[tid] = totalValue;
//        }
//        d_totalWeight[tid] = totalWeight;
//    }
//}
//
//void calculateFitness(const vector<char>& solution, int* d_totalWeight, const vector<Item>& items, int* d_fitness, const int itemCount) {
//    char* d_solution;
//    Item* d_items;
//
//    cudaMalloc((void**)&d_solution, POPULATION_SIZE * itemCount * sizeof(char));
//    cudaMalloc((void**)&d_items, itemCount * sizeof(Item));
//
//    cudaMemcpy(d_solution, solution.data(), POPULATION_SIZE * itemCount * sizeof(char), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_items, items.data(), itemCount * sizeof(Item), cudaMemcpyHostToDevice);
//
//    int threadsPerBlock = 256;
//    int blocksPerGrid = (POPULATION_SIZE + threadsPerBlock - 1) / threadsPerBlock;
//
//    calculateFitnessKernel << <blocksPerGrid, threadsPerBlock >> > (d_solution, d_totalWeight, d_items, d_fitness, itemCount);
//
//    cudaDeviceSynchronize();  // Ensure all threads have finished
//
//    cudaMemcpy(d_totalWeight, d_totalWeight, POPULATION_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
//    cudaMemcpy(d_fitness, d_fitness, POPULATION_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
//
//    cudaFree(d_solution);
//    cudaFree(d_items);
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
//int main() {
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
//    vector<vector<char>> population(POPULATION_SIZE);
//
//    int* d_fitness;
//    int* d_totalWeight;
//
//    // Allocate device memory for d_fitness and d_totalWeight
//    cudaMalloc((void**)&d_fitness, POPULATION_SIZE * sizeof(int));
//    cudaMalloc((void**)&d_totalWeight, POPULATION_SIZE * sizeof(int));
//
//    for (int i = 0; i < POPULATION_SIZE; ++i) {
//        population[i] = generateRandomSolution(itemCount);
//    }
//
//    int bestFitness = 0;
//    vector<char> bestSolution;
//    int bestWeight = 0;
//
//    clock_t start_time = clock();
//
//    for (int generation = 0; generation < GENERATION_COUNT; ++generation) {
//        vector<vector<char>> newPopulation;
//        int fitness[POPULATION_SIZE];
//        int totalWeight[POPULATION_SIZE];
//
//        for (int i = 0; i < POPULATION_SIZE; ++i) {
//            calculateFitness(population[i], d_totalWeight, items, d_fitness, itemCount);
//
//            cudaMemcpy(&fitness[i], d_fitness, sizeof(int), cudaMemcpyDeviceToHost);
//            cudaMemcpy(&totalWeight[i], d_totalWeight, sizeof(int), cudaMemcpyDeviceToHost);
//
//            if (fitness[i] > bestFitness) {
//                bestFitness = fitness[i];
//                bestSolution = population[i];
//                bestWeight = totalWeight[i];
//            }
//
//            newPopulation.push_back(population[i]);
//        }
//
//        population = newPopulation;
//
//        cout << "Generation " << generation << ": Best value = " << bestFitness << ", Best weight = " << bestWeight << endl;
//    }
//
//    clock_t end_time = clock();
//    double elapsed_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;
//
//    cout << "Best solution: ";
//    for (size_t i = 0; i < items.size(); ++i) {
//        if (bestSolution[i]) {
//            cout << items[i].name << " ";
//        }
//    }
//
//    cout << endl;
//    cout << "Elapsed time: " << elapsed_time << " seconds" << endl;
//
//    cudaFree(d_fitness);
//    cudaFree(d_totalWeight);
//
//    return 0;
//}
