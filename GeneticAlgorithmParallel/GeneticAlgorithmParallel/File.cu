//#include <iostream>
//#include <vector>
//#include <algorithm>
//#include <ctime>
//#include <cstdlib>
//#include <string>
//#include <cuda_runtime.h>
//#include "device_launch_parameters.h"
////SERIAL WITH GENERATED ITEMS
//using namespace std;
//
//struct Item {
//    string name;
//    int weight;
//    int value;
//};
//
//
////Step 1 : Initialize
//const int POPULATION_SIZE = 5000;
//const int GENERATION_COUNT = 100;
//const double MUTATION_RATE = 0.1;
//const int KNAPSACK_CAPACITY = 3000;
//
//bool compareItems(const Item& item1, const Item& item2) {
//    return (static_cast<double>(item1.value) / item1.weight) > (static_cast<double>(item2.value) / item2.weight);
//}
//
//
////Step 1 : Generate solutions 
////Generates a solution based on number of items
////if itemCount = 5, solution = 10110
//vector<bool> generateRandomSolution(size_t itemCount) {
//    vector<bool> solution(itemCount);
//    for (size_t i = 0; i < itemCount; ++i) {
//        solution[i] = rand() % 2;
//    }
//    return solution;
//}
//
//__global__ void calculateFitnessKernel(const bool* d_solution, int* d_totalWeight, const Item* d_items, int* d_fitness, const int itemCount) {
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
//// Host function to launch the CUDA kernel
//void calculateFitness(const vector<char>& solution, int* d_totalWeight, const vector<Item>& items, int* d_fitness, const int itemCount) {
//    // Allocate device memory for solution, items, totalWeight, and fitness
//    char* d_solution;
//    Item* d_items;
//
//    cudaMalloc((void**)&d_solution, POPULATION_SIZE * itemCount * sizeof(char));
//    cudaMalloc((void**)&d_items, itemCount * sizeof(Item));
//
//    // Copy the solution data to device memory
//    cudaMemcpy(d_solution, solution.data(), POPULATION_SIZE * itemCount * sizeof(char), cudaMemcpyHostToDevice);
//    // Copy the items data to device memory
//    cudaMemcpy(d_items, items.data(), itemCount * sizeof(Item), cudaMemcpyHostToDevice);
//
//    // Allocate device memory for totalWeight and fitness
//    int* d_totalWeight;
//    cudaMalloc((void**)&d_totalWeight, POPULATION_SIZE * sizeof(int));
//
//    // Launch the CUDA kernel
//    calculateFitnessKernel << <blocksPerGrid, threadsPerBlock >> > (d_solution, d_totalWeight, d_items, d_fitness, itemCount);
//
//    // Copy results (totalWeight and fitness) back to host memory
//    cudaMemcpy(d_totalWeight, d_totalWeight, POPULATION_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
//    cudaMemcpy(d_fitness, d_fitness, POPULATION_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
//
//    // Free device memory
//    cudaFree(d_solution);
//    cudaFree(d_items);
//    cudaFree(d_totalWeight);
//}
//
//vector<bool> crossover(const vector<bool>& parent1, const vector<bool>& parent2) {
//    vector<bool> child(parent1.size());
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
////Step 3 : Mutate
//// loop through the bits of the child, determine if it gets flipped
//void mutate(vector<bool>& solution) {
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
//    size_t itemCount = 77; // Define the number of items (adjust as needed)
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
//    vector<vector<bool>> population(POPULATION_SIZE);
//
//    // Initialize device arrays for fitness calculation
//    int* d_fitness;
//    int* d_totalWeight;
//    cudaMalloc((void**)&d_fitness, POPULATION_SIZE * sizeof(int));
//    cudaMalloc((void**)&d_totalWeight, POPULATION_SIZE * sizeof(int));
//
//    for (int i = 0; i < POPULATION_SIZE; ++i) {
//        population[i] = generateRandomSolution(itemCount);
//    }
//
//    int bestFitness = 0;
//    vector<bool> bestSolution;
//    int bestWeight = 0;
//
//    clock_t start_time = clock(); // Record the starting time
//
//    for (int generation = 0; generation < GENERATION_COUNT; ++generation) {
//        vector<vector<bool>> newPopulation;
//
//        // Calculate fitness for each individual in the population using CUDA
//        for (int i = 0; i < POPULATION_SIZE; ++i) {
//            calculateFitness(population[i], d_totalWeight, items, d_fitness, itemCount);
//
//            // Copy fitness and totalWeight back from device to host
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
//    clock_t end_time = clock(); // Record the ending time
//
//    double elapsed_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC; // Calculate elapsed time
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
//    // Free device memory
//    cudaFree(d_fitness);
//    cudaFree(d_totalWeight);
//
//    return 0;
//}
