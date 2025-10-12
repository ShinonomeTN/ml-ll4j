package com.shinonometn.ml.ll4j;

import java.util.Arrays;

/**
 * Layer function is a function that accept layer weights, and processing iteration result
 */
public final class AdjustFunctions {
    private AdjustFunctions() {
    }

    public static final ForwardFunction DenseAdjust = (input, trans, output) -> {
        final int inputSize = input.length;
        final int outputSize = output.length;

        for (int idxO = 0; idxO < outputSize; idxO++) {
            double sum = 0;
            for (int idxI = 0; idxI < inputSize; idxI++) {
                sum += input[idxI] * trans[idxO * idxI];
            }
            output[idxO] = sum;
        }
    };

    /** An empty update function. */
    static final LayerAdjust.Updater Noop = (input, layer, error, learningRate) -> {};

    static final LayerAdjust.Updater DenseUpdate = (inputs, layer, errors, learningRate) -> {
        final int inputSize = layer.getInputSize();
        final int outputSize = layer.getOutputSize();
        final double[] weights = layer.data;

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights[i * j] -= learningRate * errors[j] * inputs[i];
            }
        }
    };
    //================================================================

    /* LeakyRelu has no weight to update */
    static final LayerAdjust.Updater LeakyReluUpdate = Noop;

    //================================================================

    /* Judge has no weight to update */
    static final LayerAdjust.Updater MaxIndexUpdate = Noop;
    static final LayerAdjust.Updater JudgeUpdate = MaxIndexUpdate;

    //================================================================
    static final NRandom random = new NRandom(System.nanoTime());


    // According to the source code, it has been used as initializer
    public static Layer fillWithGaussianRandom(final Layer layer) {
        final double[] input = layer.data;
        final int inputSize = input.length;
        for (int i = 0; i < inputSize; i++)
            input[i] = random.nextGaussian(0, 1.0 / Math.sqrt(inputSize));

        return layer;
    }

    public static double[] fillWithZero(final double[] data) {
        Arrays.fill(data, 0.0);
        return data;
    }

    /**
     * It seems that it haven't been use.
     */
    public static Layer fillWithRandom(final Layer layer, final double rv) {
        final int inputSize = layer.getInputSize();
        final int outputSize = layer.getOutputSize();
        final double[] data = layer.data;
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                data[i * j] = random.nextGaussian(1 - rv, 1 + rv);
            }
        }
        return layer;
    }
}
