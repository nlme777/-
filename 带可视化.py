import numpy as np
import pickle
import os
import matplotlib.pyplot as plt


def load_cifar10_batch(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data['data'], data['labels']


def load_cifar10(root_dir):
    train_data = []
    train_labels = []
    for i in range(1, 6):
        file_path = os.path.join(root_dir, f'data_batch_{i}')
        data, labels = load_cifar10_batch(file_path)
        train_data.append(data)
        train_labels.append(labels)
    train_data = np.vstack(train_data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    train_labels = np.hstack(train_labels)

    test_file_path = os.path.join(root_dir, 'test_batch')
    test_data, test_labels = load_cifar10_batch(test_file_path)
    test_data = test_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    return train_data, train_labels, test_data, test_labels


def normalize_data(data):
    return data.astype(np.float32) / 255.0


def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]


def preprocess_data(train_data, train_labels, test_data, test_labels):
    train_data = normalize_data(train_data)
    test_data = normalize_data(test_data)
    train_labels = one_hot_encode(train_labels)
    test_labels = one_hot_encode(test_labels)
    return train_data, train_labels, test_data, test_labels


class Layer:
    def __init__(self, input_dim, output_dim, activation='relu'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.bias = np.zeros((1, output_dim))
        self.input = None
        self.output = None

    def forward(self, input_data):
        # 如果是第一层且输入是图像，需要展平
        if len(input_data.shape) > 2:
            input_data = input_data.reshape(input_data.shape[0], -1)
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.bias
        if self.activation == 'relu':
            self.output = np.maximum(0, self.output)
        elif self.activation == 'softmax':
            exp_output = np.exp(self.output - np.max(self.output, axis=1, keepdims=True))
            self.output = exp_output / np.sum(exp_output, axis=1, keepdims=True)
        return self.output

    def backward(self, output_error, learning_rate, regularization_lambda):
        if self.activation == 'relu':
            output_error[self.output <= 0] = 0
        elif self.activation == 'softmax':
            pass  # No need to modify for softmax

        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        bias_error = np.sum(output_error, axis=0, keepdims=True)

        # Apply regularization
        weights_error += regularization_lambda * self.weights

        # Update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * bias_error

        return input_error


class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, activation='relu'):
        self.layers = [
            Layer(input_dim, hidden_dim, activation),
            Layer(hidden_dim, hidden_dim, activation),
            Layer(hidden_dim, output_dim, 'softmax')
        ]

    def forward(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, output_error, learning_rate, regularization_lambda):
        for layer in reversed(self.layers):
            output_error = layer.backward(output_error, learning_rate, regularization_lambda)

    def compute_loss(self, predictions, labels, regularization_lambda):
        loss = -np.mean(np.sum(labels * np.log(predictions + 1e-9), axis=1))
        for layer in self.layers:
            loss += 0.5 * regularization_lambda * np.sum(np.square(layer.weights))
        return loss


def train_model(model, train_data, train_labels, val_data, val_labels, learning_rate, epochs, batch_size,
                regularization_lambda):
    num_samples = train_data.shape[0]
    best_val_accuracy = 0
    best_weights = None

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        # Shuffle data
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        train_data = train_data[indices]
        train_labels = train_labels[indices]

        epoch_train_loss = 0
        # Mini-batch training
        for i in range(0, num_samples, batch_size):
            batch_data = train_data[i:i + batch_size]
            batch_labels = train_labels[i:i + batch_size]

            # Forward pass
            predictions = model.forward(batch_data)

            # Compute loss
            loss = model.compute_loss(predictions, batch_labels, regularization_lambda)
            epoch_train_loss += loss
            print(f'Epoch {epoch + 1}, Batch {i // batch_size + 1}, Loss: {loss}')

            # Backward pass
            output_error = predictions - batch_labels
            model.backward(output_error, learning_rate, regularization_lambda)

        epoch_train_loss /= (num_samples // batch_size)
        train_losses.append(epoch_train_loss)

        # Validation
        val_predictions = model.forward(val_data)
        val_loss = model.compute_loss(val_predictions, val_labels, regularization_lambda)
        val_losses.append(val_loss)
        val_accuracy = np.mean(np.argmax(val_predictions, axis=1) == np.argmax(val_labels, axis=1))
        val_accuracies.append(val_accuracy)
        print(f'Epoch {epoch + 1}, Validation Accuracy: {val_accuracy}')

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_weights = [layer.weights.copy() for layer in model.layers]

            # Learning rate decay
        learning_rate *= 0.95

        # Load best weights
    for i, layer in enumerate(model.layers):
        layer.weights = best_weights[i]

        # Plot training and validation loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    return model


def test_model(model, test_data, test_labels):
    predictions = model.forward(test_data)
    accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(test_labels, axis=1))
    print(f'Test Accuracy: {accuracy}')
    return accuracy


def hyperparameter_search(train_data, train_labels, val_data, val_labels):
    learning_rates = [0.01, 0.001]
    hidden_dims = [128, 256]
    regularization_lambdas = [0.01, 0.001]

    best_hyperparameters = None
    best_val_accuracy = 0

    for lr in learning_rates:
        for hd in hidden_dims:
            for rl in regularization_lambdas:
                print(f'Training with learning_rate={lr}, hidden_dim={hd}, regularization_lambda={rl}')
                model = NeuralNetwork(input_dim=32 * 32 * 3, hidden_dim=hd, output_dim=10)
                model = train_model(model, train_data, train_labels, val_data, val_labels, lr, epochs=10, batch_size=64,
                                    regularization_lambda=rl)
                val_accuracy = test_model(model, val_data, val_labels)
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_hyperparameters = (lr, hd, rl)

    print(
        f'Best Hyperparameters: learning_rate={best_hyperparameters[0]}, hidden_dim={best_hyperparameters[1]}, regularization_lambda={best_hyperparameters[2]}')
    return best_hyperparameters


def visualize_model_weights(model):
    for i, layer in enumerate(model.layers):
        weights = layer.weights
        plt.figure(figsize=(8, 6))
        plt.imshow(weights, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title(f'Weights  of Layer {i + 1}')
        plt.xlabel('Output  Neurons')
        plt.ylabel('Input  Neurons')
        plt.show()


if __name__ == '__main__':
    # Load and preprocess CIFAR-10 dataset
    train_data, train_labels, test_data, test_labels = load_cifar10('cifar-10-python/cifar-10-batches-py')
    train_data, train_labels, test_data, test_labels = preprocess_data(train_data, train_labels, test_data, test_labels)

    # Split into training and validation sets
    val_size = int(0.2 * train_data.shape[0])
    val_data, val_labels = train_data[:val_size], train_labels[:val_size]
    train_data, train_labels = train_data[val_size:], train_labels[val_size:]

    # Hyperparameter search
    best_hyperparameters = hyperparameter_search(train_data, train_labels, val_data, val_labels)

    # Train the best model with the best hyperparameters
    best_lr, best_hd, best_rl = best_hyperparameters
    print(
        f'Training the best model with learning_rate={best_lr}, hidden_dim={best_hd}, regularization_lambda={best_rl}')
    best_model = NeuralNetwork(input_dim=32 * 32 * 3, hidden_dim=best_hd, output_dim=10)
    best_model = train_model(best_model, train_data, train_labels, val_data, val_labels, best_lr, epochs=30,
                             batch_size=64, regularization_lambda=best_rl)

    # Test the best model
    print("Testing the best model on the test set...")
    test_accuracy = test_model(best_model, test_data, test_labels)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Visualize model weights
    visualize_model_weights(best_model)