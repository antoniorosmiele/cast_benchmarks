/* Already trained DNN to recognize 0-9 digits from MNIST */

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cmath>
#include <ctime>

///////////////////////////
//APPROSIMATED DATA ARRAYS
///////////////////////////
#define WEIGHT1 0
#define WEIGHT2 1
#define WEIGHT3 2
#define INPUT 3


// Define constants
constexpr size_t INPUT_SIZE = 28 * 28; // 28x28 MNIST images
constexpr size_t HIDDEN_LAYER_SIZE = 10; // Number of units in each hidden layer
constexpr size_t OUTPUT_SIZE = 1;  // Single output unit
constexpr size_t MAX_SAMPLES = 10000; // Maximum number of samples
constexpr size_t NUM_EPOCHS = 10; // Number of epochs
constexpr float LEARNING_RATE = 0.01f;
constexpr size_t NUM_HIDDEN_LAYERS = 3; // Number of hidden layers

// Define dataset storage
struct Sample {
    float image[INPUT_SIZE];
    float target;
};

Sample dataset[MAX_SAMPLES];
size_t dataset_size = 0;

struct{
float input_weights[HIDDEN_LAYER_SIZE][INPUT_SIZE];
float input_biases[HIDDEN_LAYER_SIZE];
} weight1;

struct{
float hidden_weights[NUM_HIDDEN_LAYERS][HIDDEN_LAYER_SIZE][HIDDEN_LAYER_SIZE];
float hidden_biases[NUM_HIDDEN_LAYERS][HIDDEN_LAYER_SIZE];
} weight2;

struct{
float output_weights[OUTPUT_SIZE][HIDDEN_LAYER_SIZE];
float output_biases[OUTPUT_SIZE];
} weight3;

float input_layer[INPUT_SIZE];
float hidden_layer[NUM_HIDDEN_LAYERS][HIDDEN_LAYER_SIZE];
float output_layer[OUTPUT_SIZE];

void load_dataset(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file) {
        std::cerr << "Could not read dataset!" << std::endl;
        return;
    }

    std::string tmp;
    char c;
    bool is_label = true;
    size_t sample_index = 0, pixel_index = 0;

    while (file.get(c)) {
        if (c == ',' || c == ';') {
            int val = std::stoi(tmp);
            tmp.clear();

            if (is_label) {
                dataset[sample_index].target = static_cast<float>(val);
                is_label = false;
                pixel_index = 0;
            } else {
                dataset[sample_index].image[pixel_index++] = val / 255.0f;
            }

            if (c == ';') {
                is_label = true;
                sample_index++;
                if (sample_index >= MAX_SAMPLES) break;
            }
        } else {
            tmp += c;
        }
    }
    dataset_size = sample_index;
}

void initialize_weights() {
    for (size_t i = 0; i < HIDDEN_LAYER_SIZE; ++i) {
        for (size_t j = 0; j < INPUT_SIZE; ++j) {
            weight1.input_weights[i][j] = static_cast<float>(rand()) / RAND_MAX * 0.01f;
        }
        weight1.input_biases[i] = static_cast<float>(rand()) / RAND_MAX * 0.01f;
    }

    for (size_t layer = 0; layer < NUM_HIDDEN_LAYERS; ++layer) {
        for (size_t i = 0; i < HIDDEN_LAYER_SIZE; ++i) {
            for (size_t j = 0; j < HIDDEN_LAYER_SIZE; ++j) {
                weight2.hidden_weights[layer][i][j] = static_cast<float>(rand()) / RAND_MAX * 0.01f;
            }
            weight2.hidden_biases[layer][i] = static_cast<float>(rand()) / RAND_MAX * 0.01f;
        }
    }

    for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
        for (size_t j = 0; j < HIDDEN_LAYER_SIZE; ++j) {
            weight3.output_weights[i][j] = static_cast<float>(rand()) / RAND_MAX * 0.01f;
        }
        weight3.output_biases[i] = static_cast<float>(rand()) / RAND_MAX * 0.01f;
    }
}

float relu(float x) {
    return x > 0 ? x : 0;
}

float relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

float forward(const float input[INPUT_SIZE]) {
    for (size_t i = 0; i < HIDDEN_LAYER_SIZE; ++i) {
        hidden_layer[0][i] = weight1.input_biases[i];
        for (size_t j = 0; j < INPUT_SIZE; ++j) {
            hidden_layer[0][i] += input[j] * weight1.input_weights[i][j];
        }
        hidden_layer[0][i] = relu(hidden_layer[0][i]);
    }

    for (size_t layer = 1; layer < NUM_HIDDEN_LAYERS; ++layer) {
        for (size_t i = 0; i < HIDDEN_LAYER_SIZE; ++i) {
            hidden_layer[layer][i] = weight2.hidden_biases[layer][i];
            for (size_t j = 0; j < HIDDEN_LAYER_SIZE; ++j) {
                hidden_layer[layer][i] += hidden_layer[layer - 1][j] * weight2.hidden_weights[layer - 1][j][i];
            }
            hidden_layer[layer][i] = relu(hidden_layer[layer][i]);
        }
    }

    for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
        output_layer[i] = weight3.output_biases[i];
        for (size_t j = 0; j < HIDDEN_LAYER_SIZE; ++j) {
            output_layer[i] += hidden_layer[NUM_HIDDEN_LAYERS - 1][j] * weight3.output_weights[i][j];
        }
    }

    return output_layer[0];
}

void backward(const float input[INPUT_SIZE], float target, float lr) {
    float prediction = forward(input);
    float error_signal = 2.0f * (prediction - target); // Derivative of MSE loss

    float hidden_error[HIDDEN_LAYER_SIZE] = {};
    for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
        for (size_t j = 0; j < HIDDEN_LAYER_SIZE; ++j) {
            float grad = error_signal * hidden_layer[NUM_HIDDEN_LAYERS - 1][j];
            weight3.output_weights[i][j] -= lr * grad;
            hidden_error[j] += error_signal * weight3.output_weights[i][j];
        }
        weight3.output_biases[i] -= lr * error_signal;
    }

    for (size_t layer = NUM_HIDDEN_LAYERS - 1; layer > 0; --layer) {
        float layer_error[HIDDEN_LAYER_SIZE] = {};
        for (size_t i = 0; i < HIDDEN_LAYER_SIZE; ++i) {
            hidden_error[i] *= relu_derivative(hidden_layer[layer][i]);
            for (size_t j = 0; j < HIDDEN_LAYER_SIZE; ++j) {
                float grad = hidden_error[i] * hidden_layer[layer - 1][j];
                weight2.hidden_weights[layer - 1][j][i] -= lr * grad;
            }
            weight2.hidden_biases[layer][i] -= lr * hidden_error[i];
        }
    }

    for (size_t i = 0; i < HIDDEN_LAYER_SIZE; ++i) {
        hidden_error[i] *= relu_derivative(hidden_layer[0][i]);
        for (size_t j = 0; j < INPUT_SIZE; ++j) {
            float grad = hidden_error[i] * input[j];
            weight1.input_weights[i][j] -= lr * grad;
        }
        weight1.input_biases[i] -= lr * hidden_error[i];
    }
}

float classify(float prediction) {
    return prediction > 0.5f ? 1.0f : 0.0f;
}

void train(size_t epochs, float lr) {
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        for (size_t i = 0; i < dataset_size; ++i) {
            const Sample& sample = dataset[i];
            float prediction = forward(sample.image);
            float loss = (prediction - sample.target) * (prediction - sample.target);
            total_loss += loss;
            backward(sample.image, sample.target, lr);
        }
        std::cout << "Epoch " << epoch + 1 << "/" << epochs
                  << " - Loss: " << total_loss / dataset_size << std::endl;
    }
}

float evaluate() {
    size_t correct = 0;
    for (size_t i = 0; i < dataset_size; ++i) {
        const Sample& sample = dataset[i];
        float prediction = forward(sample.image);
        correct += classify(prediction) == sample.target ? 1 : 0;
    }
    return static_cast<float>(correct) / dataset_size;
}

void save_model(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Could not save model!" << std::endl;
        return;
    }

    file.write(reinterpret_cast<char*>(weight1.input_weights), sizeof(weight1.input_weights));
    file.write(reinterpret_cast<char*>(weight1.input_biases), sizeof(weight1.input_biases));
    file.write(reinterpret_cast<char*>(weight2.hidden_weights), sizeof(weight2.hidden_weights));
    file.write(reinterpret_cast<char*>(weight2.hidden_biases), sizeof(weight2.hidden_biases));
    file.write(reinterpret_cast<char*>(weight3.output_weights), sizeof(weight3.output_weights));
    file.write(reinterpret_cast<char*>(weight3.output_biases), sizeof(weight3.output_biases));

    std::cout << "Model saved to " << filename << std::endl;
}

void load_model(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Could not load model!" << std::endl;
        return;
    }

    file.read(reinterpret_cast<char*>(weight1.input_weights), sizeof(weight1.input_weights));
    file.read(reinterpret_cast<char*>(weight1.input_biases), sizeof(weight1.input_biases));
    file.read(reinterpret_cast<char*>(weight2.hidden_weights), sizeof(weight2.hidden_weights));
    file.read(reinterpret_cast<char*>(weight2.hidden_biases), sizeof(weight2.hidden_biases));
    file.read(reinterpret_cast<char*>(weight3.output_weights), sizeof(weight3.output_weights));
    file.read(reinterpret_cast<char*>(weight3.output_biases), sizeof(weight3.output_biases));

    std::cout << "Model loaded from " << filename << std::endl;
}

int main(int argc, char** argv) {
    if(argc!=3){
        std::cout << "USAGE: " << argv[0] << " dataset.txt output.txt" << std::endl<< std::endl;
        exit(1);
    }
    srand(static_cast<unsigned>(time(0))); // Seed for random number generator

    // Initialize network weights
    //initialize_weights();

    load_dataset(argv[1]);

    //train(NUM_EPOCHS, LEARNING_RATE);

    
    //save_model("model.bin");

    load_model("model.bin");
    
    std::ofstream f(argv[2]);
    f << evaluate() << std::endl;
    f.close();

    return 0;
}





