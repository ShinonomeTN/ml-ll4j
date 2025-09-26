package com.shinonometn.ml.ll4j;


public final class Matrix {

    public interface Transform {
        void forward(final double[] input, final double[] transform, final double[] output);

        default void backward(final double[] input, final double[] transform, final double[] output) {
            forward(input, transform, output);
        }

        //================================================================

        Transform Dense = (input, trans, output) -> {
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
        Transform LeakyRelu = (input, trans, output) -> {
            final int inputSize = input.length;

            for (int i = 0; i < inputSize; i++) {
                output[i] = input[i] > 0 ? input[i] : input[i] * 0.01;
            }
        };

        //================================================================

        /**
         * What a Judge do is select the greatest value of the result.
         * I modify it to be a function that just check if the result is NaN and copy it to output
         */
        Transform Judge = new Transform() {
            @Override
            public void forward(double[] input, double[] trans, double[] output) {
                final int inputSize = input.length;
                for (int i = 1; i < inputSize; i++) {
                    final double v = input[i];
                    if (Double.isNaN(v)) {
                        throw new RuntimeException("input[" + i + "] is NaN! Plz reduce learning rate!");
                    }
                    output[i] = v;
                }
            }

            @Override
            public void backward(double[] input, double[] transform, double[] output) {
                final int max = maxIndex(input);
                for (int i = 0; i < output.length; i++) {
                    if (i == max)
                        output[i] = input[i] - 1;
                    else
                        output[i] = input[i];
                }
            }
        };
    }

    public static int maxIndex(double[] input) {
        int result = 0;
        for (int i = 1; i < input.length; i++) {
            if (input[i] > input[i - 1]) result = i;
        }
        return result;
    }
}
