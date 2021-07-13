using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using MathNet.Numerics.LinearAlgebra;
using Random = UnityEngine.Random;

public class GeneticAlgo : MonoBehaviour
{
    [Header("References")]
    public CarController controller;

    [Header("Controls")]
    public int initialPopulation = 85;
    [Range(-1f, 1f)]
    public float mutationRate = 0.055f;

    [Header("Crossover Controls")]
    public int bestAgentSelection = 8;
    public int worstAgentSelection = 3;
    public int numberToCrossover = 39;

    private List<int> genePool = new List<int>();
    private int naturallySelected;
    private NeuralNet[] population;

    [Header("Public View")]
    public int currentGeneration;
    public int currentGenome = 0;

    private void Start()
    {
       CreatePopulation(); 
    }

    private void CreatePopulation(){
        population = new NeuralNet[initialPopulation];
        FillWithRandom(population, 0);
        ResetToCurrentGenome();
    }

    private void ResetToCurrentGenome(){
        controller.ResetWithNet(population[currentGenome]);
    }

    private void FillWithRandom(NeuralNet[] newPopulation, int startIndex ){
        while(startIndex < initialPopulation){
            newPopulation[startIndex] = new NeuralNet();
            newPopulation[startIndex].Initialize(controller.Layers, controller.Neurons);
            startIndex = startIndex + 1;
        }
    }

    public void Death(float fitness, NeuralNet net){
        if(currentGenome < population.Length - 1){
            population[currentGenome].fitness = fitness;
            currentGenome = currentGenome + 1;
            ResetToCurrentGenome();
        }
        else{
            RePopulate();
        }
    }

    private void RePopulate(){
        genePool.Clear();
        currentGeneration = currentGeneration + 1;
        naturallySelected = 0;

        SortPopulation();

        NeuralNet[] newPopulation = NaturallySelected();

        CrossOver(newPopulation);
        Mutate(newPopulation);

        FillWithRandom(newPopulation, naturallySelected);

        population = newPopulation;

        currentGenome = 0;
        ResetToCurrentGenome();
    }

    private NeuralNet[] NaturallySelected(){

        NeuralNet[] newPopulation = new NeuralNet[initialPopulation];

        for(int i = 0; i < bestAgentSelection; i++){
            
            newPopulation[naturallySelected] = population[i].InitializeCopy(controller.Layers, controller.Neurons);
            newPopulation[naturallySelected].fitness = 0;
            naturallySelected = naturallySelected + 1;

            int f = Mathf.RoundToInt(population[i].fitness * 10);

            for(int c = 0; c < f; i++){
                genePool.Add(i);
            }
        } 

        for(int i = 0; i < worstAgentSelection; i++){
            int last = population.Length - 1;
            last = last - i;

            int f = Mathf.RoundToInt(population[last].fitness*10);
            for(int c = 0; c < f; c++){
                genePool.Add(last);
            }
        }

        return newPopulation;
    }

    private void CrossOver(NeuralNet[] newPopulation){

        for(int i = 0; i < numberToCrossover; i = i + 2){
            int P1 = i;
            int P2 = i + 1;
            if(genePool.Count > 1){
                for(int j = 0; j < 100; j++){
                    P1 = genePool[Random.Range(0, genePool.Count)];
                    P2 = genePool[Random.Range(0, genePool.Count)];

                    if(P1 != P2) break;
                }
            }

            NeuralNet Child1 = new NeuralNet();
            NeuralNet Child2 = new NeuralNet();

            Child1.Initialize(controller.Layers, controller.Neurons);
            Child2.Initialize(controller.Layers, controller.Neurons);

            Child1.fitness = 0;
            Child2.fitness = 0;

            for(int w = 0; w < Child1.weights.Count; w++){
                if(Random.Range(0f, 1f) < 0.5){
                    Child1.weights[w] = population[P1].weights[w];
                    Child2.weights[w] = population[P2].weights[w];
                }
                else{
                    Child1.weights[w] = population[P2].weights[w];
                    Child2.weights[w] = population[P1].weights[w];
                }
            }

            for(int b = 0; b < Child1.biases.Count; b++){
                if(Random.Range(0f, 1f) < 0.5){
                    Child1.biases[b] = population[P1].biases[b];
                    Child2.biases[b] = population[P2].biases[b];
                }
                else{
                    Child1.biases[b] = population[P2].biases[b];
                    Child2.biases[b] = population[P1].biases[b];
                }
            }

            newPopulation[naturallySelected] = Child1;
            naturallySelected = naturallySelected + 1;

            newPopulation[naturallySelected] = Child2;
            naturallySelected = naturallySelected + 1;
        }

    }

    private void Mutate(NeuralNet[] newPopulation){
        for(int i = 0; i < naturallySelected; i++){
            for(int c = 0; c < newPopulation[i].weights.Count; c++){
                if(Random.Range(0f, 1f) < mutationRate){
                    newPopulation[i].weights[c] = RandomiseMatrix(newPopulation[i].weights[c]);
                }
            }
        }
    }

    private Matrix<float> RandomiseMatrix(Matrix<float> mutant){

        int rand = Random.Range(1, (mutant.RowCount * mutant.ColumnCount) / 7);

        Matrix<float> mutated = mutant;

        for(int count = 0; count < rand; count++){
            int randColumn = Random.Range(0, mutated.ColumnCount);
            int randRow = Random.Range(0, mutated.RowCount);

            mutated[randRow, randColumn] = Mathf.Clamp(mutated[randRow, randColumn] + Random.Range(-1f, 1f), -1f, 1f);
        }

        return mutated;
    }

    private void SortPopulation(){
        for(int i = 0; i < population.Length; i++){
            for(int j = i+1; j < population.Length; j++){
                if(population[i].fitness < population[j].fitness){
                    NeuralNet temp = population[j];
                    population[j] = population[i];
                    population[i] = temp;
                }
            }
        }
    }
}
