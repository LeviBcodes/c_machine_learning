/*
Fundamental machine learning with C
using classes and structs
no libraries
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


// struct for the neuron
typedef struct Neuron {
    double *weights;
    double bias;
    double output;
} Neuron;

// struct for the layer
typedef struct Layer {
    Neuron *neurons;
    int num_neurons;
} Layer;

// struct for the network
typedef struct Network {
    Layer *layers;
    int num_layers;
} Network;

// struct for the training data
typedef struct Data {
    double *inputs;
    double *outputs;
} Data;

// struct for the training set
typedef struct TrainingSet {
    Data *data;
    int num_data;
} TrainingSet;

// function to create a neuron
Neuron create_neuron(int num_weights) {
    Neuron neuron;
    neuron.bias = 0.0;
    neuron.output = 0.0;
    neuron.weights = malloc(num_weights * sizeof(double));
    
    if (neuron.weights == NULL) {
        printf("Error: malloc failed\n");
        exit(1);
    }

    for (int i = 0; i < num_weights; i++) {
        // Initialize weights with random values between -1 and 1
        neuron.weights[i] = (2.0 * ((double)rand() / RAND_MAX)) - 1.0;
    }
    
    return neuron;
}

// function to create a layer
Layer create_layer(int num_neurons, int num_weights) {
    Layer layer;
    layer.num_neurons = num_neurons;
    layer.neurons = malloc(num_neurons * sizeof(Neuron));
    if (layer.neurons == NULL) {
        printf("Error: malloc failed\n");
        exit(1);
    }
    for (int i = 0; i < num_neurons; i++) {
        layer.neurons[i] = create_neuron(num_weights);
    }
    return layer;
}
// function to create a network
Network create_network(int num_layers, int *num_neurons, int *num_weights) {
    Network network;
    network.num_layers = num_layers;
    network.layers = malloc(num_layers * sizeof(Layer));
    if (network.layers == NULL) {
        printf("Error: malloc failed\n");
        exit(1);
    }
    for (int i = 0; i < num_layers; i++) {
        network.layers[i] = create_layer(num_neurons[i], num_weights[i]);
    }
    return network;
}

// function to create a data
Data create_data(int num_inputs, int num_outputs) {
    Data data;
    data.inputs = malloc(num_inputs * sizeof(double));
    data.outputs = malloc(num_outputs * sizeof(double));
    return data;
}

// function to create a training set
TrainingSet create_training_set(int num_data, int num_inputs, int num_outputs) {
    TrainingSet training_set;
    training_set.num_data = num_data;
    training_set.data = malloc(num_data * sizeof(Data));
    for (int i = 0; i < num_data; i++) {
        training_set.data[i] = create_data(num_inputs, num_outputs);
    }
    return training_set;
}


// function to generate a random double between 0 and 1
double random_double() {
    return (double)rand() / (double)RAND_MAX;
}


// function to set the weights of a neuron
void set_weights(Neuron *neuron, double *weights) {
    for (int i = 0; i < sizeof(neuron->weights); i++) {
        neuron->weights[i] = weights[i];
    }
}

// function to set the bias of a neuron
void set_bias(Neuron *neuron, double bias) {
    neuron->bias = bias;
}


// function to set the inputs of a data
void set_inputs(Data *data, double *inputs) {
    for (int i = 0; i < sizeof(data->inputs); i++) {
        data->inputs[i] = inputs[i];
    }
}


// function to set the outputs of a data
void set_outputs(Data *data, double *outputs) {
    for (int i = 0; i < sizeof(data->outputs); i++) {
        data->outputs[i] = outputs[i];
    }
}


// function to set the inputs of a training set
void set_training_inputs(TrainingSet *training_set, double *inputs, int index) {
    for (int i = 0; i < sizeof(training_set->data[index].inputs); i++) {
        training_set->data[index].inputs[i] = inputs[i];
    }
}


// function to set the outputs of a training set

void set_training_outputs(TrainingSet *training_set, double *outputs, int index) {
    for (int i = 0; i < sizeof(training_set->data[index].outputs); i++) {
        training_set->data[index].outputs[i] = outputs[i];
    }
}


// function to calculate the sigmoid of a value
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}


// function to calculate the derivative of the sigmoid of a value
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}


// function to calculate the output of a neuron
double neuron_output(Neuron *neuron, double *inputs) {
    double output = 0.0;
    for (int i = 0; i < sizeof(neuron->weights); i++) {
        output += neuron->weights[i] * inputs[i];
    }
    output += neuron->bias;
    output = sigmoid(output);
    return output;
}


// function to calculate the output of a layer
double *layer_output(Layer *layer, double *inputs) {
    double *outputs = malloc(layer->num_neurons * sizeof(double));
    for (int i = 0; i < layer->num_neurons; i++) {
        outputs[i] = neuron_output(&layer->neurons[i], inputs);
    }
    return outputs;
}


// function to calculate the output of a network
double *network_output(Network *network, double *inputs) {
    double *outputs = malloc(network->layers[network->num_layers - 1].num_neurons * sizeof(double));
    double *layer_inputs = malloc(network->layers[0].num_neurons * sizeof(double));
    for (int i = 0; i < network->layers[0].num_neurons; i++) {
        layer_inputs[i] = inputs[i];
    }
    for (int i = 0; i < network->num_layers; i++) {
        outputs = layer_output(&network->layers[i], layer_inputs);
        for (int j = 0; j < network->layers[i].num_neurons; j++) {
            layer_inputs[j] = outputs[j];
        }
    }
    return outputs;
}