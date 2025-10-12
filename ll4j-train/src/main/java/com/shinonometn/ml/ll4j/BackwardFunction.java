package com.shinonometn.ml.ll4j;

public interface BackwardFunction {

    /**
     * Do backward propagation on a layer and yields errors.
     *
     * @param input  the input from forward propagation
     * @param layer  current layer
     * @param errors output errors of the layer
     * @param output input errors of the layer
     *               , will propagate to next layer's output
     */
    void apply(final double[] input, final Layer layer, final double[] errors, final double[] output);

    BackwardFunction MaxIndex = (input, x, errors, output) -> {
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
            double sum = 0;
            for (int j = 0; j < outputSize; j++) {
                sum += errors[j] * weights[i * j];
            }
            output[i] = sum;
        }
    };

    BackwardFunction LeakyRelu = (input, layer, errors, output) -> {
        final int outputSize = layer.getOutputSize();
        for (int i = 0; i < outputSize; i++) {
            final double v = input[i];
            output[i] = v > 0 ? errors[i] : (v < 0 ? errors[i] * 0.01 : Double.MIN_NORMAL);
        }
    };
}
