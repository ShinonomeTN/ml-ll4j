package com.shinonometn.ml.ll4j;

import java.util.stream.IntStream;

public interface ForwardFunction {
    /**
     * Apply the function
     *
     * @param input   Input array
     * @param weights Weights array
     * @param output  Output array
     */
    void apply(final double[] input, final double[] weights, final double[] output);

    //================================================================

    /**
     * A dense layer do the following operation:
     * output[j] = sum(input[i] * weights[i][j]) for i in [0, inputSize)
     * where weights is a 2D array with shape [inputSize, outputSize]
     * but here we use a 1D array to represent it, so the index is calculated as:
     * weights[i][j] = weights[i * outputSize + j]
     */
    ForwardFunction Dense = new ForwardFunction() {
        @Override
        public void apply(double[] input, double[] weights, double[] output) {
            final int inputSize = input.length;
            final int outputSize = output.length;

            IntStream.range(0, outputSize).parallel().forEach(idxO -> forEachOutput(
                    idxO, inputSize, outputSize, input, weights, output
            ));
        }

        private void forEachOutput(
                final int idxO,
                final int iSize, final int oSize,
                final double[] input, final double[] weights,
                final double[] output
        ) {
            double sum = 0;
            for (int idxI = 0; idxI < iSize; idxI++) {
                sum += input[idxI] * weights[(idxI * oSize) + idxO];
            }
            output[idxO] = sum;
        }
    };

    //================================================================
    /**
     * What a LeakyRelu do is just check if each value is greater than 0
     */
    ForwardFunction LeakyRelu = (input, trans, output) -> {
        final int inputSize = input.length;

        for (int i = 0; i < inputSize; i++) {
            final double v = input[i];
            if (v > 0) {
                output[i] = v;
            } else if (v < 0) {
                output[i] = v * 0.01;
            } else {
                output[i] = Double.MIN_NORMAL;
            }
        }
    };

    //================================================================

    ForwardFunction MaxIndex = (input, trans, output) -> {
        final int inputSize = input.length;
        int maxIdx = 0;
        for (int i = 0; i < inputSize; i++) {
            final double v = input[i];

            if (Double.isNaN(v)) throw new RuntimeException(
                    "input[" + i + "] is NaN! Plz reduce learning rate!"
            );

            if (v > input[maxIdx]) maxIdx = i;
        }
        output[0] = maxIdx;
    };

    /**
     * The implementation is moved to MaxIndex
     * <p>
     * What a Judge do is select the greatest value of the result.
     * I modify it to be a function that checks if the result is NaN and copy it to output
     */
    ForwardFunction Judge = MaxIndex;
}
