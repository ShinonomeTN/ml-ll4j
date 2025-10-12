package com.shinonometn.ml.ll4j;

public interface BackwardFunction {

    /**
     *
     * @param input  the input from forward propagation
     * @param layer  current layer
     * @param errors error input (output errors)
     * @param output error output (input errors)
     */
    void apply(final double[] input, final Layer layer, final double[] errors, final double[] output);

    BackwardFunction MaxIndex = (input, _layer, errors, output) -> {
        final int max = (int) errors[0];
        for (int i = 0; i < output.length; i++) {
            if (i == max) {
                output[i] = input[i] - 1;
            } else {
                output[i] = input[i];
            }
        }
    };

    BackwardFunction Dense = (input, layer, errors, output) -> {
        final double[] weights = layer.data;
        final int inputSize = layer.getInputSize();
        final int outputSize = layer.getOutputSize();
        for (int i = 0; i < inputSize; i++) {
            final double sum = 0;
            for (int j = 0; j < outputSize; j++) {
                output[i] += errors[j] * weights[i * j];
            }
            output[i] = sum;
        }
    };

    BackwardFunction LeakyRelu = (input, layer, errors, output) -> {
        final int outputSize = layer.getOutputSize();
        for (int i = 0; i < outputSize; i++) {
            if (input[i] > 0) {
                output[i] = errors[i];
            } else {
                output[i] = errors[i] * 0.01;
            }
        }
    };
}
