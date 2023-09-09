//#include <iostream>
//#include <vector>
//#include <algorithm>
//#include <ctime>
//#include <cstdlib>
//#include <string>
////SERIAL WITH PREDEFINED ITEMS
//using namespace std;
//
//struct Item {
//    string name;
//    int value;
//    int weight;
//};
//
//vector<Item> items = {
//    {"Laptop", 500, 2200},
//    {"Headphones", 150, 160},
//    {"Coffee Mug", 60, 350},
//    {"Notepad", 40, 333},
//    {"Water Bottle", 30, 192},
//    {"Mints", 5, 25},
//    {"Socks", 10, 38},
//    {"Tissues", 15, 80},
//    {"Phone", 500, 200},
//    {"Baseball Cap", 100, 70}
//    // Add more items here
//};
//
//const int POPULATION_SIZE = 5000;
//const int GENERATION_COUNT = 100;
//const double MUTATION_RATE = 0.1;
//
//const int KNAPSACK_CAPACITY = 3000;
//
//bool compareItems(const Item& item1, const Item& item2) {
//    return (static_cast<double>(item1.value) / item1.weight) > (static_cast<double>(item2.value) / item2.weight);
//}
//
//vector<bool> generateRandomSolution() {
//    vector<bool> solution(items.size());
//    for (size_t i = 0; i < items.size(); ++i) {
//        solution[i] = rand() % 2;
//    }
//    return solution;
//}
//
//int calculateFitness(const vector<bool>& solution, int& totalWeight) {
//    int totalValue = 0;
//    totalWeight = 0;
//    for (size_t i = 0; i < items.size(); ++i) {
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
//    sort(items.begin(), items.end(), compareItems);
//
//    vector<vector<bool>> population(POPULATION_SIZE);
//
//    for (int i = 0; i < POPULATION_SIZE; ++i) {
//        population[i] = generateRandomSolution();
//    }
//
//    int bestFitness = 0;
//    vector<bool> bestSolution;
//    int bestWeight = 0;
//
//
//    clock_t start_time = clock(); // Record the starting time
//
//    for (int generation = 0; generation < GENERATION_COUNT; ++generation) {
//        vector<vector<bool>> newPopulation;
//
//        for (int i = 0; i < POPULATION_SIZE; ++i) {
//            int parent1Index = rand() % POPULATION_SIZE;
//            int parent2Index = rand() % POPULATION_SIZE;
//
//            vector<bool> child = crossover(population[parent1Index], population[parent2Index]);
//
//            mutate(child);
//
//            int totalWeight;
//            int fitness = calculateFitness(child, totalWeight);
//
//            if (fitness > bestFitness) {
//                bestFitness = fitness;
//                bestSolution = child;
//                bestWeight = totalWeight;
//            }
//
//            newPopulation.push_back(child);
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
//
//    cout << "Elapsed time: " << elapsed_time << " seconds" << endl;
//
//    return 0;
//}
