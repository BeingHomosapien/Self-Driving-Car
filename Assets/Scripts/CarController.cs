using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(NeuralNet))]

public class CarController : MonoBehaviour
{
    private Vector3 startPosition, startRotation;

    [Range(-1f, 1f)]
    public float acc, turn;

    [Header("Survival Time")]
    public float timeSinceStart = 0f;

    [Header("Fitness")]
    public float overallFitness;
    public float distanceMultiplier = 0.05f;
    public float avgSpeedMultiplier = 0.005f;
    public float sensorMultiplier = 0.01f;

    private Vector3 lastPostition;
    private float totalDistanceTravelled;
    private float avgSpeed;

    private float aSensor, bSensor, cSensor;

    // Network 
    private NeuralNet network;
    [Header("Network Options")]
    public int Layers = 5;
    public int Neurons = 10;

    private void Awake(){
        startPosition = transform.position;
        startRotation = transform.eulerAngles;
        
        network = GetComponent<NeuralNet>();
        // network.Initialize(Layers, Neurons);

        Reset();
    }

    public void Reset(){
        timeSinceStart = 0f;
        overallFitness = 0f;
        lastPostition = startPosition;
        totalDistanceTravelled = 0f;
        avgSpeed = 0f;
        transform.position = startPosition;
        transform.eulerAngles = startRotation;

        // network.Initialize(Layers, Neurons);
    }

    public void ResetWithNet(NeuralNet net){
        network = net;
        Reset();
    }

    private void Death(){
        GameObject.FindObjectOfType<GeneticAlgo>().Death(overallFitness, network);
    }

    private void OnCollisionEnter(Collision collision){
        Death();
    }

    public void FixedUpdate(){

        InputSensor();

        // Neural Net For values of acc and turn;
        (acc, turn) = network.RunNetwork(aSensor, bSensor, cSensor);

        MoveCar(acc, turn);

        timeSinceStart += Time.deltaTime;

        CalculateFitness();

        // acc = 0;
        // turn = 0;   
    }

    private void CalculateFitness(){
        totalDistanceTravelled += Vector3.Distance(transform.position, lastPostition);
        avgSpeed = totalDistanceTravelled / timeSinceStart;
        
        overallFitness = (totalDistanceTravelled*distanceMultiplier) + (avgSpeed*avgSpeedMultiplier) + (((aSensor + bSensor + cSensor)/3)*sensorMultiplier);

        if(timeSinceStart > 20 && overallFitness < 1000){
            Death();
        }

        if(overallFitness >= 20000){
            // Save the network
            Death();
        }
    }

    public void MoveCar(float a, float t){
        Vector3 inp;

        inp = Vector3.Lerp(Vector3.zero, new Vector3(0, 0, a*11.4f), 0.02f);
        inp = transform.TransformDirection(inp);

        transform.position += inp;
        transform.eulerAngles += new Vector3(0, t*90*0.02f, 0);

    }

    private void InputSensor(){
        Vector3 a = transform.forward + transform.right;
        Vector3 b = transform.forward; 
        Vector3 c = transform.forward - transform.right;

        Ray r = new Ray(transform.position, a);
        RaycastHit hit;
        if(Physics.Raycast(r, out hit)){
            aSensor = hit.distance/20;
            Debug.DrawLine(r.origin, hit.point, Color.red);
        }

        r.direction = b;
        if(Physics.Raycast(r, out hit)){
            bSensor = hit.distance/20;
            Debug.DrawLine(r.origin, hit.point, Color.red);
        }

        r.direction = c;
        if(Physics.Raycast(r, out hit)){
            cSensor = hit.distance/20;
            Debug.DrawLine(r.origin, hit.point, Color.red);
        }

    }
}
