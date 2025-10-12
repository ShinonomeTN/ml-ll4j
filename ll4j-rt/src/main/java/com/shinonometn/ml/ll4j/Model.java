package com.shinonometn.ml.ll4j;

import java.util.LinkedList;

/**
 * Model is a cache for each classification job
 */
public class Model {
    final Layer[] layers;

    Model(final Layer[] layers) {
        this.layers = layers;
    }

    /**
     * Get model input size
     */
    public int getInputSize() {
        return layers[0].getInputSize();
    }

    /**
     * Get model output size
     */
    public int getOutputSize() {
        return layers[layers.length - 1].getOutputSize();
    }

    //================================================================

    /**
     * Parse model from a list of string representing layer weights
     */
    public static Model parseLayers(final String[] model) throws MinRtException {
        final LinkedList<Layer> layers = new LinkedList<>();

        for (final String s : model) {
            // Ignore empty layer
            // layer data is type + size(vararg) + data(vararg)
            if (s.length() < 2) continue;

            final String[] tokens = s.trim().split(" ");

            int pos = 0; // Position of reader

            // Read the type first
            final String type = tokens[pos++];

            // Get input count, the second field is always input count
            final int inputCount = Integer.parseInt(tokens[pos++]); // Input Count
            if (!layers.isEmpty()) { // If is not the first layer, check the input size
                final int inputSize = layers.getLast().getOutputSize();
                if (inputSize != inputCount) throw new LayerInputMismatchException(String.format(
                        "Wrong input size for layer %s@%02d (expected %d, got %d)",
                        type, layers.size(), inputCount, inputSize
                ));
            }

            switch (type) {
                // Dense Layer
                // Output count at 2
                case "D": {
                    final int outputCount = Integer.parseInt(tokens[pos++]);

                    // Parse data
                    final double[] data = new double[tokens.length - pos];
                    for (int i = 0; i < data.length; i++)
                        data[i] = Double.parseDouble(tokens[pos++]);

                    layers.add(Layers.dense(inputCount, outputCount, data));
                    break;
                }

                // For "LeakyRelu Layer"
                case "L": {
                    // Input is same as output
                    layers.add(Layers.leakyRelu(inputCount));
                    break;
                }

                // For "Judge Layer"
                case "J": {
                    // It only output a single number
                    layers.add(Layers.judge(inputCount));
                    break;
                }

                default: {
                    throw new UnsupportedLayerTypeException(type);
                }
            }
        }

        return new Model(layers.toArray(new Layer[0]));
    }

    /**
     * Do classification with a parsed model
     */
    public double[] classification(double[] input) throws MinRtException {
        final int inputSize = getInputSize();
        if (input.length != inputSize) throw new MinRtException(String.format(
                "Wrong input size for this model, expected %d, got %d", inputSize, input.length
        ));

        double[] ref = input;
        for (Layer layer : layers) {
            final double[] next = new double[layer.getOutputSize()];
            layer.function.apply(ref, layer.data, next);
            ref = next;
        }
        return ref;
    }
}
