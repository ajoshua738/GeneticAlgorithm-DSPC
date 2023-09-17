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

// Step 1: Initialize
const int POPULATION_SIZE = 5000;
const int GENERATION_COUNT = 100;
const double MUTATION_RATE = 0.1;
const int KNAPSACK_CAPACITY = 3000;

bool compareItems(const Item& item1, const Item& item2) {
    return (static_cast<double>(item1.value) / item1.weight) > (static_cast<double>(item2.value) / item2.weight);
}

// Step 1: Generate solutions
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

// Step 3: Mutate
// Loop through the bits of the child, determine if it gets flipped
void mutate(vector<char>& solution) {
    for (size_t i = 0; i < solution.size(); ++i) {
        if (static_cast<double>(rand()) / RAND_MAX < MUTATION_RATE) {
            solution[i] = ~solution[i];
        }
    }
}



int main(int argc, char* argv[]) {
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
    vector<char> bestSolution;
    int bestWeight = 0;

    int rank, size;
    double begin = 0.0, end = 0.0;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    begin = MPI_Wtime(); // Record the starting time

    for (int generation = rank; generation < GENERATION_COUNT; generation+=size) {
        vector<vector<char>> newPopulation;

        // Step 2: Takes 2 random solutions from the population
        for (int i = 0; i < POPULATION_SIZE; ++i) {
            int parent1Index = rand() % POPULATION_SIZE;
            int parent2Index = rand() % POPULATION_SIZE;

            vector<char> child = crossover(population[parent1Index], population[parent2Index]);

            // Step 3: Mutate
            mutate(child);

            // Step 4: Calculate fitness of child
            int totalWeight;
            int fitness = calculateFitness(child, totalWeight, items);

            if (fitness > bestFitness) {
                bestFitness = fitness;
                bestSolution = child;
                bestWeight = totalWeight;
            }

            newPopulation.push_back(child);
        }

        population = newPopulation;

        cout << "Generation " << generation << ": Best value = " << bestFitness << ", Best weight = " << bestWeight << endl;
    }

    end = MPI_Wtime(); // Record the ending time

    //Display Finally Result
    if (rank == 0)
    {
       cout << "Best solution: ";
        for (size_t i = 0; i < items.size(); ++i) {
            if (bestSolution[i]) {
                cout << items[i].name << " ";
            }
        }

        cout << endl;
        cout << "Elapsed time: " << end - begin << " seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}