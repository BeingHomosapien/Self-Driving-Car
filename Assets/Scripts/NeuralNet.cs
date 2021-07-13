using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using MathNet.Numerics.LinearAlgebra;
using Random = UnityEngine.Random;

public class NeuralNet : MonoBehaviour
{
    // Neural Net Structure
    public Matrix<float> inputLayer = Matrix<float>.Build.Dense(1, 3);
    public List<Matrix<float>> hiddenLayers = new List<Matrix<float>>(); 
    public Matrix<float> outputLayer = Matrix<float>.Build.Dense(1, 2);

    // Weights ad Biases
    public List<Matrix<float>> weights = new List<Matrix<float>>();
    public List<float> biases = new List<float>();

    // Fitness
    public float fitness;

    public void Initialize(int hiddenLayerCount, int hiddenNeuronCount){
        inputLayer.Clear();
        hiddenLayers.Clear();
        outputLayer.Clear();
        weights.Clear();
        biases.Clear();

        for(int i = 0; i < hiddenLayerCount; i++){
            // Layers
            Matrix<float> layer = Matrix<float>.Build.Dense(1, hiddenNeuronCount);
            hiddenLayers.Add(layer);

            // Biases
            biases.Add(Random.Range(-1f, 1f));

            // Weights
            if(i == 0){
                Matrix<float> inputWeights = Matrix<float>.Build.Dense(3, hiddenNeuronCount);
                weights.Add(inputWeights);
            }
            else{
                Matrix<float> hiddenWeights = Matrix<float>.Build.Dense(hiddenNeuronCount, hiddenNeuronCount);
                weights.Add(hiddenWeights);
            }

        }

        // Last Hidden Lyaer to Output Layer
        biases.Add(Random.Range(-1f, 1f));
        Matrix<float> outputWeights = Matrix<float>.Build.Dense(hiddenNeuronCount, 2);
        weights.Add(outputWeights);

        // Randommize the Weights
        RandommizeWeights();
    }

    private void RandommizeWeights(){

        for(int i = 0; i < weights.Count; i++){
            for(int j = 0; j < weights[i].RowCount; j++){
                for(int k = 0; k < weights[i].ColumnCount; k++){
                    weights[i][j, k] = Random.Range(-1f, 1f);
                }
            }
        }

    }

    public NeuralNet InitializeCopy(int hiddenLayerCount, int hiddenNeuronCount){

        NeuralNet net = new NeuralNet();

        List<Matrix<float>> newWeights = new List<Matrix<float>>();

        // Copying the Weights
        for(int i = 0; i < this.weights.Count; i++){
            Matrix<float> current =  Matrix<float>.Build.Dense(this.weights[i].RowCount, this.weights[i].ColumnCount);
            for(int j = 0; j < current.RowCount; j++){
                for(int k = 0; k < current.ColumnCount; k++){
                    current[j, k] = this.weights[i][j, k];
                }
            }

            newWeights.Add(current);
        }

        List<float> newBiases = new List<float>();
        
        // Copying the Biases
        for(int i = 0; i < this.biases.Count; i++){
            newBiases.Add(this.biases[i]);
        }

        net.weights = newWeights;
        net.biases = newBiases;

        net.InitializeHidden(hiddenLayerCount, hiddenNeuronCount);

        return net;
    }

    public void InitializeHidden(int hiddenLayerCount, int hiddenNeuronCount){
        inputLayer.Clear();
        hiddenLayers.Clear();
        outputLayer.Clear();

        for(int i = 0; i < hiddenLayerCount; i++){
            Matrix<float> layer = Matrix<float>.Build.Dense(1, hiddenNeuronCount);
            hiddenLayers.Add(layer);
        }
    }

    public (float, float) RunNetwork(float aSensor, float bSensor, float cSensor){

        // Inputs
        inputLayer[0, 0] = aSensor;
        inputLayer[0, 1] = bSensor;
        inputLayer[0, 2] = cSensor;

        // Acivation Layer
        inputLayer = inputLayer.PointwiseTanh();

        // Hidden Layers
        hiddenLayers[0] = ((inputLayer*weights[0]) + biases[0]).PointwiseTanh();

        for(int i = 1; i < hiddenLayers.Count; i++){
            hiddenLayers[i] = ((hiddenLayers[i-1] * weights[i]) + biases[i]).PointwiseTanh();
        }

        // Output Layer 
        outputLayer = ((hiddenLayers[hiddenLayers.Count - 1] * weights[weights.Count - 1]) + biases[biases.Count - 1]).PointwiseTanh();

        return (Sigmoid(outputLayer[0, 0]), outputLayer[0, 1]);
    }

    private float Sigmoid(float s){
        return (1/(1 + Mathf.Exp(-s)));
    }
}
