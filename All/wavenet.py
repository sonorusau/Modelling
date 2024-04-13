import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def prep_data(specs_train, specs_test):
    specs_train_flatten = []
    specs_test_flatten = []
    for i in range(specs_train.shape[0]): 
        specs_train_flatten.append(specs_train[i].flatten())
    for i in range(specs_test.shape[0]):
        specs_test_flatten.append(specs_test[i].flatten())
    specs_train_flatten = np.array(specs_train_flatten)
    specs_test_flatten = np.array(specs_test_flatten)

        # Add a new dimension for 'channels' at the end of the existing dimensions
    specs_train_expanded = np.expand_dims(specs_train_flatten, axis=-1)
    specs_test_expanded = np.expand_dims(specs_test_flatten, axis=-1)

    return specs_train_expanded, specs_test_expanded


class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate):
        super(ResidualBlock, self).__init__()
        self.dilated_conv_tanh = layers.Conv1D(filters=filters, kernel_size=kernel_size,
                                               dilation_rate=dilation_rate, padding='same', activation='tanh')
        self.dilated_conv_sigmoid = layers.Conv1D(filters=filters, kernel_size=kernel_size,
                                                  dilation_rate=dilation_rate, padding='same', activation='sigmoid')
        self.multiply = layers.Multiply()
        self.skip_connection_conv = layers.Conv1D(filters=filters, kernel_size=1, activation='relu')

    def call(self, inputs):
        tanh_out = self.dilated_conv_tanh(inputs)
        sigmoid_out = self.dilated_conv_sigmoid(inputs)
        combined = self.multiply([tanh_out, sigmoid_out])
        skip_out = self.skip_connection_conv(combined)
        out = layers.Add()([skip_out, inputs])  # Residual connection
        return out, skip_out

class WaveNetModel(tf.keras.Model):
    def __init__(self, num_blocks=6, num_residual_channels=32, num_skip_channels=32, num_output_classes=2):
        super(WaveNetModel, self).__init__()
        self.initial_conv = tf.keras.layers.Conv1D(filters=num_residual_channels, kernel_size=1, activation='relu')

        # Stacking residual blocks
        self.residual_blocks = []
        for i in range(num_blocks):
            dilation_rate = 2 ** i
            self.residual_blocks.append(ResidualBlock(filters=num_residual_channels, kernel_size=3, dilation_rate=dilation_rate))

        self.final_layers = [
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1D(filters=num_skip_channels, kernel_size=1, activation='relu')
        ]

        # Final Dense layer with softmax activation
        self.final_layer = tf.keras.layers.Dense(num_output_classes, activation='softmax')

    def call(self, inputs):
        x = self.initial_conv(inputs)
        skip_connections = []

        for block in self.residual_blocks:
            x, skip = block(x)
            skip_connections.append(skip)

        # Summing skip connections
        x = tf.keras.layers.Add()(skip_connections)

        for layer in self.final_layers:
            x = layer(x)

        # Apply the final Dense layer
        x = tf.keras.layers.Flatten()(x)
        x = self.final_layer(x)
        return x

def wavenet_train(specs_train, specs_test, outcomes_train, outcomes_test):

    #wavenet with spectrograms
    # Create the model instance
    wavenet_model = WaveNetModel()

    # Build the model with the input shape
    wavenet_model.build(input_shape=(None, specs_train.shape[1],  specs_train.shape[2]))  

    # Compile the model with optimizer and loss
    wavenet_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0005),
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
    # Train the model
    wavenet_model.fit(specs_train, outcomes_train, epochs=3, batch_size=3, validation_data=(specs_test, outcomes_test))

    return wavenet_model