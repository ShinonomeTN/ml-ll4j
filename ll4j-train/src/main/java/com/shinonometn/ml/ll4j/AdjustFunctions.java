package com.shinonometn.ml.ll4j;

/**
 * Layer function is a function that accept layer weights, and processing iteration result
 */
public final class AdjustFunctions {
    private AdjustFunctions() {
    }

    /** An empty update function. */
    static final LayerAdjust.Updater Noop = (input, layer, error, learningRate) -> {};

    static final LayerAdjust.Updater DenseUpdate = (layer, inputs, errors, learningRate) -> {
        final int inputSize = layer.getInputSize();
        final double[] layerData = layer.data;
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < layer.getOutputSize(); j++) {
                double delta = learningRate * errors[j] * inputs[i];
                layerData[i * j] -= delta;
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
