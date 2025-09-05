/* see mnist folder */

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cmath>
#include <ctime>

#include <vector>
#include <cstdint>
#include <cassert>
#include<stdexcept>

///////////////////////////
//APPROSIMATED DATA ARRAYS
///////////////////////////
#define WEIGHT1 0
#define WEIGHT2 1
#define WEIGHT3 2
#define INPUT 3

// Define constants
constexpr size_t INPUT_SIZE = 28 * 28; 
constexpr size_t IMAGE_SIZE = 28 * 28; 
constexpr size_t OUTPUT_SIZE = 10;  // Number of classes
constexpr size_t NUM_TRAIN_IMAGES = 60000;
constexpr size_t NUM_TEST_IMAGES = 10000;
constexpr size_t MAX_SAMPLES = 10000;

constexpr size_t HIDDEN_LAYER_SIZE = 100; // Number of units in each hidden layer
constexpr size_t NUM_EPOCHS = 25; // Number of epochs
constexpr float LEARNING_RATE = 0.0003f;
constexpr size_t NUM_HIDDEN_LAYERS = 3; // Number of hidden layers

struct Sample {
    float image[IMAGE_SIZE]; // Flattened 28x28 image stored as a simple array
    float target;            // The label (0-9)
};

Sample dataset[NUM_TRAIN_IMAGES]; // Fixed-size array for the dataset
size_t dataset_size = 0;

float minLoss;
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
    srand(static_cast<unsigned>(time(0))); // Seed for random number generator
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

void softmax(float output[OUTPUT_SIZE]) {
    float max_val = output[0];
    for (size_t i = 1; i < OUTPUT_SIZE; ++i) {
        if (output[i] > max_val) max_val = output[i];
    }

    float sum = 0.0f;
    for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
        output[i] = exp(output[i] - max_val); // Numerical stability
        sum += output[i];
    }

    for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
        output[i] /= sum;
    }
}

void forward(const float input[INPUT_SIZE]) {
    //float min, max;
    
    // Input to first hidden layer
    //min = max = hidden_layer[0][0];
    for (size_t i = 0; i < HIDDEN_LAYER_SIZE; ++i) {
        hidden_layer[0][i] = weight1.input_biases[i];
        for (size_t j = 0; j < INPUT_SIZE; ++j) {
            hidden_layer[0][i] += input[j] * weight1.input_weights[i][j];
        }
        hidden_layer[0][i] = relu(hidden_layer[0][i]);
        if(hidden_layer[0][i]>3.0 || hidden_layer[0][i]<0.0)hidden_layer[0][i] = 0;
        //if(hidden_layer[0][i]>max) max = hidden_layer[0][i];
        //if(hidden_layer[0][i]<max) min = hidden_layer[0][i];
    }
    //printf("%f %f\n", min, max);

    // Hidden layers
    for (size_t layer = 1; layer < NUM_HIDDEN_LAYERS; ++layer) {
        //min = max = hidden_layer[layer][0];
        for (size_t i = 0; i < HIDDEN_LAYER_SIZE; ++i) {
            hidden_layer[layer][i] = weight2.hidden_biases[layer][i];
            for (size_t j = 0; j < HIDDEN_LAYER_SIZE; ++j) {
                hidden_layer[layer][i] += hidden_layer[layer - 1][j] * weight2.hidden_weights[layer - 1][j][i];
            }
            hidden_layer[layer][i] = relu(hidden_layer[layer][i]);
            if(hidden_layer[layer][i]>3.0 || hidden_layer[layer][i]<0.0)hidden_layer[layer][i] = 0;
            //if(hidden_layer[layer][i]>max) max = hidden_layer[layer][i];
            //if(hidden_layer[layer][i]<max) min = hidden_layer[layer][i];
        }
        //printf("%f %f\n", min, max);
    }

    // Last hidden layer to output
    for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
        output_layer[i] = weight3.output_biases[i];
        for (size_t j = 0; j < HIDDEN_LAYER_SIZE; ++j) {
            output_layer[i] += hidden_layer[NUM_HIDDEN_LAYERS - 1][j] * weight3.output_weights[i][j];
        }
        if(output_layer[i]>3.0 || output_layer[i]<0.0)output_layer[i] = 0;
        //if(output_layer[i]>max) max = output_layer[i];
        //if(output_layer[i]<max) min = output_layer[i];
    }
    //printf("%f %f\n", min, max);
}


void backward(const float input[INPUT_SIZE], int target_class, float lr) {
    forward(input); // Ensure forward pass is done

    float output_error[OUTPUT_SIZE] = {0};

    // Compute output layer error signals
    for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
        float target = (i == target_class) ? 1.0f : 0.0f; // One-hot encoding
        output_error[i] = 2.0f * (output_layer[i] - target);
    }

    // Output layer to hidden layer error propagation
    float hidden_error[NUM_HIDDEN_LAYERS][HIDDEN_LAYER_SIZE] = {};
    for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
        for (size_t j = 0; j < HIDDEN_LAYER_SIZE; ++j) {
            float grad = output_error[i] * hidden_layer[NUM_HIDDEN_LAYERS - 1][j];
            weight3.output_weights[i][j] -= lr * grad;
            hidden_error[NUM_HIDDEN_LAYERS - 1][j] += output_error[i] * weight3.output_weights[i][j];
        }
        weight3.output_biases[i] -= lr * output_error[i];
    }


    // Backpropagation through hidden layers
    for (int layer = NUM_HIDDEN_LAYERS - 1; layer > 0; --layer) {
        for (size_t i = 0; i < HIDDEN_LAYER_SIZE; ++i) {
            hidden_error[layer][i] *= relu_derivative(hidden_layer[layer][i]);
            for (size_t j = 0; j < HIDDEN_LAYER_SIZE; ++j) {
                float grad = hidden_error[layer][i] * hidden_layer[layer - 1][j];
                weight2.hidden_weights[layer - 1][j][i] -= lr * grad;
                hidden_error[layer - 1][j] += hidden_error[layer][i] * weight2.hidden_weights[layer - 1][j][i];
            }
            weight2.hidden_biases[layer][i] -= lr * hidden_error[layer][i];
        }
    }

    // Input layer to first hidden layer
    for (size_t i = 0; i < HIDDEN_LAYER_SIZE; ++i) {
        hidden_error[0][i] *= relu_derivative(hidden_layer[0][i]);
        for (size_t j = 0; j < INPUT_SIZE; ++j) {
            float grad = hidden_error[0][i] * input[j];
            weight1.input_weights[i][j] -= lr * grad;
        }
        weight1.input_biases[i] -= lr * hidden_error[0][i];
    }
}



int classify() {
    int max_index = 0;
    float max_value = output_layer[0];
    for (size_t i = 1; i < OUTPUT_SIZE; ++i) {
        if (output_layer[i] > max_value) {
            max_value = output_layer[i];
            max_index = i;
        }
    }
    return max_index; // Return the index of the class with the highest output
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

    //std::cout << "Model saved to " << filename << std::endl;
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

    //std::cout << "Model loaded from " << filename << std::endl;
}


void train(size_t epochs, float lr) {
    minLoss=20.0;
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        for (size_t i = 0; i < dataset_size; ++i) {
            const Sample& sample = dataset[i];
            forward(sample.image); // Forward pass
            backward(sample.image, static_cast<int>(sample.target), lr); // Backward pass

            // Compute loss for the correct class
            int target_class = static_cast<int>(sample.target);
            float target_value = 1.0f; // One-hot encoded target
            float loss = (output_layer[target_class] - target_value) * (output_layer[target_class] - target_value);
            total_loss += loss;
        }
        std::cout << "Epoch " << epoch + 1 << "/" << epochs
                  << " - Loss: " << total_loss / dataset_size << std::endl;
            if(total_loss/dataset_size<minLoss&&total_loss/dataset_size>0.0001){
            	    save_model("model.bin");
            	    minLoss=total_loss/dataset_size;
            }
    }
}

void evaluate() {
    size_t correct = 0;
    for (size_t i = 0; i < dataset_size; ++i) {
        const Sample& sample = dataset[i];
        forward(sample.image);
        int predicted_class = classify();
        correct += (predicted_class == static_cast<int>(sample.target)) ? 1 : 0;
    }
    std::cout << "Accuracy: " << static_cast<float>(correct) / dataset_size << std::endl;
}

// Function to read MNIST dataset in IDX format
std::vector<uint8_t> read_idx(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    file.seekg(0, std::ios::end);
    size_t filesize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(filesize);
    file.read(reinterpret_cast<char*>(buffer.data()), filesize);
    return buffer;
}

void load_dataset2(const std::string& dataset_path) {

    const std::string train_images_path = dataset_path + "/train-images.idx3-ubyte";
    const std::string train_labels_path = dataset_path + "/train-labels.idx1-ubyte";

    // Read data files
    auto images_data = read_idx(train_images_path);
    auto labels_data = read_idx(train_labels_path);

    // Parse header for images
    uint32_t num_images = (images_data[4] << 24) | (images_data[5] << 16) | (images_data[6] << 8) | images_data[7];
    uint32_t rows = (images_data[8] << 24) | (images_data[9] << 16) | (images_data[10] << 8) | images_data[11];
    uint32_t cols = (images_data[12] << 24) | (images_data[13] << 16) | (images_data[14] << 8) | images_data[15];

    assert(rows == 28 && cols == 28 && "MNIST images must be 28x28 pixels.");
    assert(num_images <= NUM_TRAIN_IMAGES && "Dataset size exceeds allocated limit.");

    // Parse header for labels
    uint32_t num_labels = (labels_data[4] << 24) | (labels_data[5] << 16) | (labels_data[6] << 8) | labels_data[7];
    assert(num_labels == num_images && "Mismatch between number of images and labels.");

    // Load samples into dataset
    for (size_t i = 0; i < num_images; ++i) {
        Sample& sample = dataset[i];

        // Normalize image pixel values to [0, 1]
        for (size_t j = 0; j < IMAGE_SIZE; ++j) {
            sample.image[j] = images_data[16 + i * IMAGE_SIZE + j] / 255.0f;
        }

        // Store label as the target value
        sample.target = static_cast<float>(labels_data[8 + i]);
    }

    dataset_size = num_images;
    std::cout << "Loaded " << dataset_size << " samples from the MNIST dataset." << std::endl;
}

void load_dataset3(char* fn) {
    FILE* fp;
    fp = fopen(fn, "rb");
    fread(&dataset[0].target, sizeof(float), 1, fp);
    fread(dataset[0].image, sizeof(float), IMAGE_SIZE, fp);
    fclose(fp);
    dataset_size = 1;
}


int main(int argc, char** argv) {
    if(argc!=3){
        std::cout << "USAGE: " << argv[0] << " dataset.txt output.txt" << std::endl<< std::endl;
        exit(1);
    }
    
    load_model("model.bin");

    load_dataset3(argv[1]);

    const Sample& sample = dataset[0];
    forward(sample.image);
    std::ofstream f(argv[2]);
    f << classify() << std::endl;
    f.close();
    return 0;

    
/*  //code to exdtract single images from the dataset
    load_dataset2("data");


    int i;
    FILE *fp;
    FILE *fp2;

    fp = fopen("labels.txt", "w");
    for(i=0; i<40; i++){
        std::string fn = std::to_string(i) + ".dat";
        fprintf(fp, "%s %d\n", fn.c_str(), (int)dataset[i].target);
        fp2 = fopen(fn.c_str(), "wb");
        fwrite(&dataset[i].target, sizeof(float), 1, fp2);
        fwrite(dataset[i].image, sizeof(float), IMAGE_SIZE, fp2);
        fclose(fp2);
    }
    fclose(fp);*/


    //srand(static_cast<unsigned>(time(0))); // Seed for random number generator
    
    //initialize_weights();
    
    //train(NUM_EPOCHS, LEARNING_RATE);

    //load_model("model.bin");

    //std::cout << "Evaluating model on training data..." << std::endl;
    
    //evaluate();

    //return 0;
}

