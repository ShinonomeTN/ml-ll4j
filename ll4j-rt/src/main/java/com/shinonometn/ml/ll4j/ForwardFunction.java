package com.shinonometn.ml.ll4j;

public interface ForwardFunction {
    void apply(final double[] input, final double[] transform, final double[] output);

    //================================================================

    ForwardFunction Dense = (input, trans, output) -> {
        final int inputSize = input.length;
        final int outputSize = output.length;

        for (int idxO = 0; idxO < outputSize; idxO++) {
            double sum = 0;
            for (int idxI = 0; idxI < inputSize; idxI++) {
                sum += input[idxI] * trans[idxO + idxI * outputSize];
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
            output[i] = input[i] > 0 ? input[i] : input[i] * 0.01;
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
