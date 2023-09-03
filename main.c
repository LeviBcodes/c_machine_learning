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
    double* weights;
    double bias;
    double output;
} Neuron;

// struct for the layer
typedef struct Layer {
    Neuron* neurons;
    int num_neurons;
} Layer;

// struct for the network
typedef struct Network {
    Layer* layers;
    int num_layers;
} Network;

// struct for the training data
typedef struct Data {
    double* inputs;
    double* outputs;
} Data;

// struct for the training set
typedef struct TrainingSet {
    Data* data;
    int num_data;
} TrainingSet;

// Function to create a single neuron with random weights
Neuron create_neuron_with_random_weights(int num_weights) {
    Neuron neuron;
    neuron.bias = 0.0;
    neuron.output = 0.0;
    neuron.weights = calloc(num_weights, sizeof(double));

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

// function to free the memory of the neuron weights
void free_neuron_weights(Neuron* neuron) {
    free(neuron->weights);
}

// Function to create a layer with random-weight neurons
Layer create_layer_with_random_weights(int num_neurons, int num_weights) {
    Layer layer;
    layer.num_neurons = num_neurons;
    layer.neurons = calloc(num_neurons, sizeof(Neuron));

    if (layer.neurons == NULL) {
        printf("Error: malloc failed\n");
        exit(1);
    }

    for (int i = 0; i < num_neurons; i++) {
        layer.neurons[i] = create_neuron_with_random_weights(num_weights);
    }

    return layer;
}

// function to free the memory of the layer neurons
void free_layer_neurons(Layer* layer) {
    for (int i = 0; i < layer->num_neurons; i++) {
        free_neuron_weights(&layer->neurons[i]);
    }
    free(layer->neurons);
}

// Function to create a layer with random-weight neurons
Layer create_random_layer(int num_neurons, int num_weights) {
    Layer layer;
    layer.num_neurons = num_neurons;
    layer.neurons = calloc(num_neurons, sizeof(Neuron));

    if (layer.neurons == NULL) {
        printf("Error: malloc failed\n");
        exit(1);
    }

    for (int i = 0; i < num_neurons; i++) {
        layer.neurons[i] = create_neuron_with_random_weights(num_weights);
    }
    return layer;
}

// function to free the memory of the layer
void free_layer(Layer* layer) {
    free_layer_neurons(layer);
    free(layer->neurons);
}

// function to create a data
Data create_data(int num_inputs, int num_outputs) {
    Data data;
    data.inputs = calloc(num_inputs, sizeof(double));
    data.outputs = calloc(num_outputs, sizeof(double));

    if (data.inputs == NULL || data.outputs == NULL) {
        printf("Error: malloc failed\n");
        exit(1);
    }

    return data;
}
// function to free the memory of the data
void free_data(Data* data) {
    free(data->inputs);
    free(data->outputs);
}

// function to create a training set
TrainingSet create_training_set(int num_data, int num_inputs, int num_outputs) {
    TrainingSet training_set;
    training_set.num_data = num_data;
    training_set.data = calloc(num_data, sizeof(Data));
    if (training_set.data == NULL) {
        printf("Error: malloc failed\n");
        exit(1);
    }
    for (int i = 0; i < num_data; i++) {
        training_set.data[i] = create_data(num_inputs, num_outputs);
    }

    return training_set;
}

// function to free the memory of the training set
void free_training_set(TrainingSet* training_set) {
    for (int i = 0; i < training_set->num_data; i++) {
        free_data(&training_set->data[i]);
    }
    free(training_set->data);
}

// function to generate a random double between 0 and 1
double random_double() {
    return (double)rand() / (double)RAND_MAX;
}


// function to set the weights of a neuron
void set_weights(Neuron* neuron, double* weights) {
    for (int i = 0; i < sizeof(neuron->weights); i++) {
        neuron->weights[i] = weights[i];
    }
}

// function to set the bias of a neuron
void set_bias(Neuron* neuron, double bias) {
    neuron->bias = bias;
}


// function to set the inputs of a data
void set_inputs(Data* data, double* inputs) {
    for (int i = 0; i < sizeof(data->inputs); i++) {
        data->inputs[i] = inputs[i];
    }
}


// function to set the outputs of a data
void set_outputs(Data* data, double* outputs) {
    for (int i = 0; i < sizeof(data->outputs); i++) {
        data->outputs[i] = outputs[i];
    }
}


// function to set the inputs of a training set
void set_training_inputs(TrainingSet* training_set, double* inputs, int index) {
    for (int i = 0; i < sizeof(training_set->data[index].inputs); i++) {
        training_set->data[index].inputs[i] = inputs[i];
    }
}


// function to set the outputs of a training set

void set_training_outputs(TrainingSet* training_set, double* outputs, int index) {
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
double neuron_output(Neuron* neuron, double* inputs) {
    double output = 0.0;
    for (int i = 0; i < sizeof(neuron->weights); i++) {
        output += neuron->weights[i] * inputs[i];
    }
    output += neuron->bias;
    output = sigmoid(output);
    return output;
}


// function to calculate the output of a layer
double* layer_output(Layer* layer, double* inputs) {

    if(layer == NULL) {
        printf("Error: layer is NULL\n");
        exit(1);
    }
    double* outputs = calloc(layer->num_neurons, sizeof(double));
    for (int i = 0; i < layer->num_neurons; i++) {
        outputs[i] = neuron_output(&layer->neurons[i], inputs);
    }
    
    return outputs;
}


// function to calculate the output of a network
double* network_output(Network* network, double* inputs) {
    double* outputs = calloc(network->layers[network->num_layers - 1].num_neurons , sizeof(double));
    double* layer_inputs = calloc(network->layers[0].num_neurons, sizeof(double));
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

int main()
{

    srand(time(NULL));

    // Create a network
    Network network;
    network.num_layers = 3;
    network.layers = calloc(network.num_layers, sizeof(Layer));
    network.layers[0] = create_layer_with_random_weights(2, 2);
    network.layers[1] = create_layer_with_random_weights(2, 2);
    network.layers[2] = create_layer_with_random_weights(1, 2);


    // Create a training set
    TrainingSet training_set = create_training_set(4, 2, 1);
    set_training_inputs(&training_set, (double[]) { 0.0, 0.0 }, 0);
    set_training_outputs(&training_set, (double[]) { 0.0 }, 0);
    set_training_inputs(&training_set, (double[]) { 0.0, 1.0 }, 1);
    set_training_outputs(&training_set, (double[]) { 1.0 }, 1);
    set_training_inputs(&training_set, (double[]) { 1.0, 0.0 }, 2);
    set_training_outputs(&training_set, (double[]) { 1.0 }, 2);
    set_training_inputs(&training_set, (double[]) { 1.0, 1.0 }, 3);
    set_training_outputs(&training_set, (double[]) { 0.0 }, 3);


    // Train the network
    double* outputs;
    double* errors;
    double* layer_inputs;
    double* layer_outputs;
    double* layer_errors;
    double* layer_weights;
    double* layer_weights_deltas;
    double* layer_bias_deltas;
    double* layer_weights_deltas_sum;
    double* layer_bias_deltas_sum;
    double* layer_weights_deltas_avg;
    double* layer_bias_deltas_avg;
    double* layer_weights_deltas_sum_prev;
    double* layer_bias_deltas_sum_prev;
    double* layer_weights_deltas_avg_prev;
    double* layer_bias_deltas_avg_prev;

    double learning_rate = 0.1;
    double momentum = 0.9;
    int num_epochs = 10000;
    int num_data = training_set.num_data;
    int num_layers = network.num_layers;
    int num_neurons;
    int num_weights;
    int num_inputs;
    int num_outputs;
    int num_weights_deltas;
    int num_bias_deltas;
    int num_weights_deltas_sum;
    int num_bias_deltas_sum;
    int num_weights_deltas_avg;
    int num_bias_deltas_avg;
    int num_weights_deltas_sum_prev;
    int num_bias_deltas_sum_prev;
    int num_weights_deltas_avg_prev;
    int num_bias_deltas_avg_prev;

    // Initialize the weights deltas
    for (int i = 0; i < num_layers; i++) {
        num_neurons = network.layers[i].num_neurons;
        num_weights = sizeof(network.layers[i].neurons[0].weights) / sizeof(double);
        num_weights_deltas = num_neurons * num_weights;
        num_bias_deltas = num_neurons;
        num_weights_deltas_sum = num_weights_deltas;
        num_bias_deltas_sum = num_bias_deltas;
        num_weights_deltas_avg = num_weights_deltas;
        num_bias_deltas_avg = num_bias_deltas;
        num_weights_deltas_sum_prev = num_weights_deltas;
        num_bias_deltas_sum_prev = num_bias_deltas;
        num_weights_deltas_avg_prev = num_weights_deltas;
        num_bias_deltas_avg_prev = num_bias_deltas;
        layer_weights_deltas = calloc(num_weights_deltas, sizeof(double));
        layer_bias_deltas = calloc(num_bias_deltas, sizeof(double));
        layer_weights_deltas_sum = calloc(num_weights_deltas_sum, sizeof(double));
        layer_bias_deltas_sum = calloc(num_bias_deltas_sum, sizeof(double));
        layer_weights_deltas_avg = calloc(num_weights_deltas_avg, sizeof(double));
        layer_bias_deltas_avg = calloc(num_bias_deltas_avg, sizeof(double));
        layer_weights_deltas_sum_prev = calloc(num_weights_deltas_sum_prev, sizeof(double));
        layer_bias_deltas_sum_prev = calloc(num_bias_deltas_sum_prev, sizeof(double));
        layer_weights_deltas_avg_prev = calloc(num_weights_deltas_avg_prev, sizeof(double));
        layer_bias_deltas_avg_prev = calloc(num_bias_deltas_avg_prev, sizeof(double));
        for (int j = 0; j < num_weights_deltas; j++) {
            layer_weights_deltas[j] = 0.0;
        }
        for (int j = 0; j < num_bias_deltas; j++) {
            layer_bias_deltas[j] = 0.0;
        }
        for (int j = 0; j < num_weights_deltas_sum; j++) {
            layer_weights_deltas_sum[j] = 0.0;
        }
        for (int j = 0; j < num_bias_deltas_sum; j++) {
            layer_bias_deltas_sum[j] = 0.0;
        }
        for (int j = 0; j < num_weights_deltas_avg; j++) {
            layer_weights_deltas_avg[j] = 0.0;
        }

        for (int j = 0; j < num_bias_deltas_avg; j++) {
            layer_bias_deltas_avg[j] = 0.0;
        }
        for (int j = 0; j < num_weights_deltas_sum_prev; j++) {
            layer_weights_deltas_sum_prev[j] = 0.0;
        }
        for (int j = 0; j < num_bias_deltas_sum_prev; j++) {
            layer_bias_deltas_sum_prev[j] = 0.0;
        }
        for (int j = 0; j < num_weights_deltas_avg_prev; j++) {
            layer_weights_deltas_avg_prev[j] = 0.0;
        }
        for (int j = 0; j < num_bias_deltas_avg_prev; j++) {
            layer_bias_deltas_avg_prev[j] = 0.0;
        }
        network.layers[i].neurons[0].weights = layer_weights_deltas;
        network.layers[i].neurons[0].weights = layer_bias_deltas;
        network.layers[i].neurons[0].weights = layer_weights_deltas_sum;
        network.layers[i].neurons[0].weights = layer_bias_deltas_sum;
        network.layers[i].neurons[0].weights = layer_weights_deltas_avg;
        network.layers[i].neurons[0].weights = layer_bias_deltas_avg;
        network.layers[i].neurons[0].weights = layer_weights_deltas_sum_prev;
        network.layers[i].neurons[0].weights = layer_bias_deltas_sum_prev;
        network.layers[i].neurons[0].weights = layer_weights_deltas_avg_prev;
        network.layers[i].neurons[0].weights = layer_bias_deltas_avg_prev;

        if (i != num_layers - 1) {
            free(layer_weights_deltas);
            free(layer_bias_deltas);
            free(layer_weights_deltas_sum);
            free(layer_bias_deltas_sum);
            free(layer_weights_deltas_avg);
            free(layer_bias_deltas_avg);
            free(layer_weights_deltas_sum_prev);
            free(layer_bias_deltas_sum_prev);
            free(layer_weights_deltas_avg_prev);
            free(layer_bias_deltas_avg_prev);
        }
    }

    // Train the network
    for (int i = 0; i < num_epochs; i++) {
        for (int j = 0; j < num_data; j++) {
            num_inputs = sizeof(training_set.data[j].inputs) / sizeof(double);
            num_outputs = sizeof(training_set.data[j].outputs) / sizeof(double);
            outputs = network_output(&network, training_set.data[j].inputs);
            errors = malloc(num_outputs * sizeof(double));
            for (int k = 0; k < num_outputs; k++) {
                errors[k] = training_set.data[j].outputs[k] - outputs[k];
            }
            for (int k = num_layers - 1; k >= 0; k--) {
                num_neurons = network.layers[k].num_neurons;
                num_weights = sizeof(network.layers[k].neurons[0].weights) / sizeof(double);
                num_weights_deltas = num_neurons * num_weights;
                num_bias_deltas = num_neurons;
                num_weights_deltas_sum = num_weights_deltas;
                num_bias_deltas_sum = num_bias_deltas;
                num_weights_deltas_avg = num_weights_deltas;
                num_bias_deltas_avg = num_bias_deltas;
                num_weights_deltas_sum_prev = num_weights_deltas;
                num_bias_deltas_sum_prev = num_bias_deltas;
                num_weights_deltas_avg_prev = num_weights_deltas;
                num_bias_deltas_avg_prev = num_bias_deltas;
                layer_inputs = calloc(num_inputs , sizeof(double));
                layer_outputs = calloc(num_neurons , sizeof(double));
                layer_errors = calloc(num_neurons , sizeof(double));
                layer_weights = calloc(num_weights , sizeof(double));
                layer_weights_deltas = calloc(num_weights_deltas , sizeof(double));
                layer_bias_deltas = calloc(num_bias_deltas , sizeof(double));
                layer_weights_deltas_sum = calloc(num_weights_deltas_sum , sizeof(double));
                layer_bias_deltas_sum = calloc(num_bias_deltas_sum , sizeof(double));
                layer_weights_deltas_avg = calloc(num_weights_deltas_avg , sizeof(double));
                layer_bias_deltas_avg = calloc(num_bias_deltas_avg , sizeof(double));
                layer_weights_deltas_sum_prev = calloc(num_weights_deltas_sum_prev , sizeof(double));
                layer_bias_deltas_sum_prev = calloc(num_bias_deltas_sum_prev , sizeof(double));
                layer_weights_deltas_avg_prev = calloc(num_weights_deltas_avg_prev , sizeof(double));
                layer_bias_deltas_avg_prev = calloc(num_bias_deltas_avg_prev, sizeof(double));
                for (int l = 0; l < num_inputs; l++) {
                    layer_inputs[l] = training_set.data[j].inputs[l];
                }
                for (int l = 0; l < num_neurons; l++) {
                    layer_outputs[l] = network.layers[k].neurons[l].output;
                }
                for (int l = 0; l < num_neurons; l++) {
                    layer_errors[l] = errors[l];
                }
                for (int l = 0; l < num_weights; l++) {
                    layer_weights[l] = network.layers[k].neurons[0].weights[l];
                }
                for (int l = 0; l < num_weights_deltas; l++) {
                    layer_weights_deltas[l] = network.layers[k].neurons[0].weights[l];
                }
                for (int l = 0; l < num_bias_deltas; l++) {
                    layer_bias_deltas[l] = network.layers[k].neurons[0].weights[l];
                }
                for (int l = 0; l < num_weights_deltas_sum; l++) {
                    layer_weights_deltas_sum[l] = network.layers[k].neurons[0].weights[l];
                }
                for (int l = 0; l < num_bias_deltas_sum; l++) {
                    layer_bias_deltas_sum[l] = network.layers[k].neurons[0].weights[l];
                }
                for (int l = 0; l < num_weights_deltas_avg; l++) {
                    layer_weights_deltas_avg[l] = network.layers[k].neurons[0].weights[l];
                }
                for (int l = 0; l < num_bias_deltas_avg; l++) {
                    layer_bias_deltas_avg[l] = network.layers[k].neurons[0].weights[l];
                }
                for (int l = 0; l < num_weights_deltas_sum_prev; l++) {
                    layer_weights_deltas_sum_prev[l] = network.layers[k].neurons[0].weights[l];
                }
                for (int l = 0; l < num_bias_deltas_sum_prev; l++) {
                    layer_bias_deltas_sum_prev[l] = network.layers[k].neurons[0].weights[l];
                }
                for (int l = 0; l < num_weights_deltas_avg_prev; l++) {
                    layer_weights_deltas_avg_prev[l] = network.layers[k].neurons[0].weights[l];
                }
                for (int l = 0; l < num_bias_deltas_avg_prev; l++) {
                    layer_bias_deltas_avg_prev[l] = network.layers[k].neurons[0].weights[l];
                }
                for (int l = 0; l < num_neurons; l++) {
                    for (int m = 0; m < num_weights; m++) {
                        layer_weights_deltas[l * num_weights + m] = layer_inputs[m] * layer_errors[l] * sigmoid_derivative(layer_outputs[l]);
                    }
                }
                for (int l = 0; l < num_neurons; l++) {
                    layer_bias_deltas[l] = layer_errors[l] * sigmoid_derivative(layer_outputs[l]);
                }
                for (int l = 0; l < num_weights_deltas; l++) {
                    layer_weights_deltas_sum[l] = layer_weights_deltas[l] + layer_weights_deltas_sum_prev[l] * momentum;
                }
                for (int l = 0; l < num_bias_deltas; l++) {
                    layer_bias_deltas_sum[l] = layer_bias_deltas[l] + layer_bias_deltas_sum_prev[l] * momentum;
                }
                for (int l = 0; l < num_weights_deltas; l++) {
                    layer_weights_deltas_avg[l] = layer_weights_deltas_sum[l] / num_data;
                }
                for (int l = 0; l < num_bias_deltas; l++) {
                    layer_bias_deltas_avg[l] = layer_bias_deltas_sum[l] / num_data;
                }
                for (int l = 0; l < num_weights_deltas; l++) {
                    layer_weights[l] += layer_weights_deltas_avg[l] * learning_rate;
                }
                for (int l = 0; l < num_bias_deltas; l++) {
                    network.layers[k].neurons[l].bias += layer_bias_deltas_avg[l] * learning_rate;
                }
                for (int l = 0; l < num_weights_deltas; l++) {
                    layer_weights_deltas_sum_prev[l] = layer_weights_deltas_sum[l];
                }
                for (int l = 0; l < num_bias_deltas; l++) {
                    layer_bias_deltas_sum_prev[l] = layer_bias_deltas_sum[l];
                }
                for (int l = 0; l < num_weights_deltas; l++) {
                    layer_weights_deltas_avg_prev[l] = layer_weights_deltas_avg[l];
                }
                for (int l = 0; l < num_bias_deltas; l++) {
                    layer_bias_deltas_avg_prev[l] = layer_bias_deltas_avg[l];
                }
                for (int l = 0; l < num_weights; l++) {
                    network.layers[k].neurons[0].weights[l] = layer_weights[l];
                }
            }

            if (i != num_epochs - 1) {
                free(outputs);
                free(errors);
                free(layer_inputs);
                free(layer_outputs);
                free(layer_errors);
                free(layer_weights);
                free(layer_weights_deltas);
                free(layer_bias_deltas);
                free(layer_weights_deltas_sum);
                free(layer_bias_deltas_sum);
                free(layer_weights_deltas_avg);
                free(layer_bias_deltas_avg);
                free(layer_weights_deltas_sum_prev);
                free(layer_bias_deltas_sum_prev);
                free(layer_weights_deltas_avg_prev);
                free(layer_bias_deltas_avg_prev);
            }
        }

    }

    // Test the network
    outputs = network_output(&network, (double[]) { 0.0, 0.0 });
    printf("0.0, 0.0 -> %f\n", outputs[0]);
    outputs = network_output(&network, (double[]) { 0.0, 1.0 });
    printf("0.0, 1.0 -> %f\n", outputs[0]);
    outputs = network_output(&network, (double[]) { 1.0, 0.0 });
    printf("1.0, 0.0 -> %f\n", outputs[0]);
    outputs = network_output(&network, (double[]) { 1.0, 1.0 });
    printf("1.0, 1.0 -> %f\n", outputs[0]);

    return 0;
}