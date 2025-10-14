package com.shinonometn.ml.ll4j;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Layer function is a function that accept layer weights, and processing iteration result
 */
public final class AdjustFunctions {
    private AdjustFunctions() {
    }

    /**
     * An empty update function.
     */
    static final LayerAdjust.Updater Noop = (input, layer, error, learningRate) -> {
    };

    static final LayerAdjust.Updater DenseUpdate = new LayerAdjust.Updater() {

        @Override
        public void apply(double[] inputs, Layer layer, double[] errors, double lr) {
            final int inputSize = layer.getInputSize();

            IntStream.range(0, inputSize).parallel().forEach(idxI -> forEachInput(
                    /*       Connection Index */ idxI,
                    /*        Layer size info */ layer.getOutputSize(),
                    /*           Layer inputs */ inputs,
                    /* Weights to be adjusted */ layer.data,
                    /*                 Errors */ errors,
                    /*          Learning rate */ lr
            ));
        }

        private void forEachInput(
                final int idxI,
                final int oSize,
                final double[] input, final double[] weights,
                final double[] errors, final double lr
        ) {
            for (int idxO = 0; idxO < oSize; idxO++) {
                final double delta = lr * errors[idxO] * input[idxI];
                weights[(idxI * oSize) + idxO] -= delta;
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
        fillWithGaussianRandom(input);
        return layer;
    }

    /**
     * Fill an array with GaussianRandom
     */
    public static double[] fillWithGaussianRandom(final double[] array) {
        final int inputSize = array.length;
        for (int i = 0; i < inputSize; i++) {
            array[i] = random.nextGaussian(0, 1.0 / Math.sqrt(inputSize));
        }
        return array;
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
        fillWithRandom(inputSize, outputSize, data, rv);
        return layer;
    }

    public static double[] fillWithRandom(final int inputSize, final int outputSize, final double[] array, final double rv) {
        if (inputSize * outputSize != array.length) throw new IllegalArgumentException("input * output != array.size");

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                array[i * j] = random.nextGaussian(1 - rv, 1 + rv);
            }
        }

        return array;
    }
}
