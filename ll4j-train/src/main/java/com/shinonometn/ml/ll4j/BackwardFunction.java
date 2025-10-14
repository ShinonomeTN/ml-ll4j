package com.shinonometn.ml.ll4j;

import java.util.stream.IntStream;

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

    BackwardFunction Dense = new BackwardFunction() {
        @Override
        public void apply(double[] input, Layer layer, double[] errors, double[] output) {
            final int inputSize = layer.getInputSize();

            IntStream.range(0, inputSize).forEach(idxI -> forEachInput(
                    /*        Input index  */ idxI,
                    /* Layer size and data */ layer.getInputSize(), layer.getOutputSize(), layer.data,
                    /*   Lower layer error */ errors,
                    /*        Error output */ output
            ));
        }

        private void forEachInput(
                final int idxI,
                final int iSize, final int oSize, final double[] weights,
                final double[] errors,
                final double[] output
        ) {
            double err = 0;
            for (int idxO = 0; idxO < oSize; idxO++) {
                err += errors[idxO] * weights[(idxO * iSize) + idxI];
            }
            output[idxI] = err;
        }
    };

    BackwardFunction LeakyRelu = (input, layer, errors, output) -> {
        final int outputSize = layer.getOutputSize();
        for (int i = 0; i < outputSize; i++) {
            final double v = input[i];
            if (v > 0) {
                output[i] = errors[i];
            } else if (v < 0) {
                output[i] = errors[i] * 0.01;
            } else {
                // if it is exactly zero, just let it be a very small value
                output[i] = Double.MIN_NORMAL;
            }
        }
    };
}
