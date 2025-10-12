package com.shinonometn.ml.ll4j;

import java.util.stream.IntStream;

public interface ForwardFunction {
    void apply(final double[] input, final double[] weights, final double[] output);

    //================================================================

    ForwardFunction Dense = (input, weights, output) -> {
        final int inputSize = input.length;
        final int outputSize = output.length;

        // For each output position, get the weights for each input,
        // multiples it and set the result.
        IntStream.range(0, outputSize).parallel().forEach(idxO -> {
            double sum = 0;
            for (int idxI = 0; idxI < inputSize; idxI++) {
                sum += input[idxI] * weights[(idxI * outputSize) + idxO];
            }
            output[idxO] = sum;
        });
    };

    //================================================================
    /**
     * What a LeakyRelu do is just check if each value is greater than 0
     */
    ForwardFunction LeakyRelu = (input, trans, output) -> {
        final int inputSize = input.length;

        for (int i = 0; i < inputSize; i++) {
            final double v = input[i];
            // I added a check for the "eq 0" scenario
            output[i] = v > 0 ? v : (v < 0 ? input[i] * 0.01 : Double.MIN_NORMAL);
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
     * What a Judge do is select the greatest value of the result.
     * I modify it to be a function that just check if the result is NaN and copy it to output
     */
    ForwardFunction Judge = MaxIndex;
}
