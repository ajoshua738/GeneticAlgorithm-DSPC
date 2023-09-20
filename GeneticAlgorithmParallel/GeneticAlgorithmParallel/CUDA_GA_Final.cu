#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <string>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <curand_kernel.h>
//SERIAL WITH GENERATED ITEMS
using namespace std;

struct Item {
    string name;
    int weight;
    int value;
};

// Step 1: Initialize
const int POPULATION_SIZE = 5000;
const int GENERATION_COUNT = 100;
const double MUTATION_RATE = 0.1;
const int KNAPSACK_CAPACITY = 3000;

bool compareItems(const Item& item1, const Item& item2) {
    return (static_cast<double>(item1.value) / item1.weight) > (static_cast<double>(item2.value) / item2.weight);
}

__global__ void geneticAlgorithmKernel(char* d_population, char* d_child, size_t itemCount, int* d_totalWeight, int* d_fitness, const Item* d_items, double mutationRate, int* d_bestFitness, int* d_bestWeight, char* d_bestSolution) {
    unsigned int threadID = blockDim.x * blockIdx.x + threadIdx.x;

    // Generate a seed based on threadID and current time
    unsigned int seed = threadID + 1;

    // Initialize the random number generator for this thread
    curandState state;
    curand_init(seed, threadID, 0, &state);

    if (threadID < POPULATION_SIZE) {

        // Generate parent indices
        // Generate a random index within the range [0, POPULATION_SIZE)

        int parent1Index = curand(&state) % POPULATION_SIZE;
        int parent2Index = curand(&state) % POPULATION_SIZE;

        // Generate crossover point
        int crossoverPoint = curand(&state) % (itemCount - 1);

        // Calculate the starting index for the child
        int childStartIndex = threadID * itemCount;


        // Assign values to d_child
        for (int i = 0; i < crossoverPoint; ++i) {
            d_child[childStartIndex + i] = d_population[parent1Index * itemCount + i];
        }
        for (int i = crossoverPoint; i < itemCount; ++i) {
            d_child[childStartIndex + i] = d_population[parent2Index * itemCount + i];
        }

        // Mutate the child: Loop through bits and flip them with a certain probability
        for (int i = 0; i < itemCount; ++i) {
            if (curand(&state) / static_cast<double>(RAND_MAX) < mutationRate) {
                d_child[childStartIndex + i] = !d_child[childStartIndex + i];
            }
        }


        // Calculate fitness for the child
        int totalValue = 0;
        int totalWeight = 0;

        for (int i = 0; i < itemCount; ++i) {
            if (d_child[childStartIndex + i]) {
                totalValue += d_items[i].value;
                totalWeight += d_items[i].weight;
            }
        }

        if (totalWeight > KNAPSACK_CAPACITY) {
            d_fitness[threadID] = 0;
        }
        else {
            d_fitness[threadID] = totalValue;
            // Compare the fitness value of the child to the bestFitness variable
            if (totalValue > *d_bestFitness) {
                // If the fitness value of the child is better than the bestFitness variable, update the bestFitness variable
                *d_bestFitness = totalValue;
                *d_bestWeight = totalWeight;

                for (int i = 0; i < itemCount; ++i) {
                    d_bestSolution[i] = d_child[childStartIndex + i];
                }
            }
        }
        d_totalWeight[threadID] = totalWeight;

       
    }




}




// Step 1: Generate solutions
vector<char> generateRandomSolution(size_t itemCount) {
    vector<char> solution(itemCount);
    for (size_t i = 0; i < itemCount; ++i) {
        solution[i] = rand() % 2;
    }
    return solution;
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

    vector<vector<char>> population(POPULATION_SIZE);

    // Step 1: Generate random solutions
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        population[i] = generateRandomSolution(itemCount);
    }

    int bestFitness = 0;
    int numOfItems = 0;
    int bestWeight = 0;

    // Create a vector to store the selected items in the best solution
    vector<Item> selectedItems;


    //Device memory pointers
    char* d_population, * d_child;
    int* d_fitness, * d_totalWeight;
    Item* d_items;
    int* d_bestFitness;
    int* d_bestWeight;

    char* d_bestSolution;



    //allocating device memory
    cudaMalloc((void**)&d_population, sizeof(char) * POPULATION_SIZE * itemCount);
    cudaMalloc((void**)&d_child, sizeof(char) * POPULATION_SIZE * itemCount);
    cudaMalloc((void**)&d_fitness, POPULATION_SIZE * sizeof(int));
    cudaMalloc((void**)&d_totalWeight, POPULATION_SIZE * sizeof(int));
    cudaMalloc((void**)&d_items, itemCount * sizeof(Item));
    cudaMalloc((void**)&d_bestFitness, sizeof(int));
    cudaMalloc((void**)&d_bestWeight, sizeof(int));
    cudaMalloc((void**)&d_bestSolution, itemCount * sizeof(char));
    //copying data from host to device
    cudaMemcpy(d_population, population.data(), sizeof(char) * POPULATION_SIZE * itemCount, cudaMemcpyHostToDevice);
    cudaMemcpy(d_items, items.data(), itemCount * sizeof(Item), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bestFitness, &bestFitness, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bestWeight, &bestWeight, sizeof(int), cudaMemcpyHostToDevice);



    clock_t start_time = clock(); // Record the starting time

    for (int generation = 0; generation < GENERATION_COUNT; ++generation) {
        vector<vector<char>> newPopulation;

        //calling kernel

        geneticAlgorithmKernel << <1, 128 >> > (d_population, d_child, itemCount, d_totalWeight, d_fitness, d_items, MUTATION_RATE, d_bestFitness,d_bestWeight, d_bestSolution); //perform crossover


        // Copy fitness and weight for the best solution from GPU to CPU
        cudaMemcpy(&bestFitness, d_bestFitness, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&bestWeight, d_bestWeight, sizeof(int), cudaMemcpyDeviceToHost);



        cout << "Generation " << generation << ": Best value = " << bestFitness << ", Best weight = " << bestWeight << endl;


        char* bestSolution = new char[itemCount];
        cudaMemcpy(bestSolution, d_bestSolution, itemCount * sizeof(char), cudaMemcpyDeviceToHost);

        
        // Update the selected items only for the final generation
        if (generation == GENERATION_COUNT - 1) {
            
            cout << "Selected items in the final best solution: \n";
            for (size_t i = 0; i < itemCount; ++i) {

                if (bestSolution[i]) {
                    numOfItems++;
                    selectedItems.push_back(items[i]);
                    cout << items[i].name << " \n";
                }
            }
            cout << "Number of items : " << numOfItems << " \n";
            cout << endl;
        }
       

        // Clean up
        delete[] bestSolution;
    }

    clock_t end_time = clock(); // Record the ending time

    double elapsed_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC; // Calculate elapsed time


    // Print all selected items in the final best solution
   /* cout << "Selected items in the final best solution: ";
    for (const Item& item : selectedItems) {
        cout << item.name << " ";
    }
    cout << endl;*/

    cout << endl;
    cout << "Elapsed time: " << elapsed_time << " seconds" << endl;

    cudaFree(d_population);
    cudaFree(d_child);
    cudaFree(d_totalWeight);
    cudaFree(d_fitness);
    cudaFree(d_items);
    cudaFree(d_bestFitness);
    cudaFree(d_bestSolution);
    cudaFree(d_bestWeight);
    return 0;
}
