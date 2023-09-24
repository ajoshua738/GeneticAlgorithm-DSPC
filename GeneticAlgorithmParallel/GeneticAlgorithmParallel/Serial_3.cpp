#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <string>
//SERIAL WITH GENERATED ITEMS
using namespace std;

struct Item {
    string name;
    int weight;
    int value;
};

// Step 1: Initialize
const int POPULATION_SIZE = 50000;
const int GENERATION_COUNT = 150;
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
        //cout << static_cast<int>(solution[i]) << endl;
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
            solution[i] = !solution[i];
        }
    }
}



int main() {
    //srand(static_cast<unsigned int>(time(nullptr)));
    srand(0);
   
    int numOfItems = 0;

    size_t itemCount = 80; // Define the number of items (adjust as needed)

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
        //cout << "solution " << i << endl;
        population[i] = generateRandomSolution(itemCount);
    }

    int bestFitness = 0;
    vector<char> bestSolution;
    int bestWeight = 0;

    

    clock_t start_time = clock(); // Record the starting time

    for (int generation = 0; generation < GENERATION_COUNT; ++generation) {
        vector<vector<char>> newPopulation;

        // Step 2: Takes 2 random solutions from the population
        for (int i = 0; i < POPULATION_SIZE; ++i) {
          
            int parent1Index = rand() % POPULATION_SIZE;
            int parent2Index = rand() % POPULATION_SIZE;
         /*   cout << "parent 1 index: " << parent1Index << endl;
            cout << "parent 2 index: " << parent2Index << endl;*/
           
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

        cout << "Generation " << generation +1 << ": Best value = " << bestFitness << ", Best weight = " << bestWeight << endl;
    }

    clock_t end_time = clock(); // Record the ending time

    double elapsed_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC; // Calculate elapsed time

    cout << "\nBest solution: ";
    for (size_t i = 0; i < items.size(); ++i) {
        if (bestSolution[i]) {
            cout << 1 << "";
        }
        else {
            cout << 0 << "";
        }
    }

    cout << "\n\nItems Selected: \n";
    int itemsInRow = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        if (bestSolution[i]) {
            numOfItems++;

            // Output the item name and add a newline if 10 items have been displayed
            cout << items[i].name << " ";
            itemsInRow++;

            // Check if we've displayed 10 items in the current row
            if (itemsInRow >= 10) {
                cout << "\n";
                itemsInRow = 0; // Reset the count for the next row
            }
        }
    }

    cout << "\n\nNumber of items : " << numOfItems << " \n";
    cout << endl;

    cout << endl;
    cout << "Elapsed time: " << elapsed_time << " seconds" << endl;

    return 0;
}
