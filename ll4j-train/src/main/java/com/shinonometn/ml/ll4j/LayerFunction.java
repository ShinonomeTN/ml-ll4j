package com.shinonometn.ml.ll4j;

/**
 * Layer function is a function that accept layer weights, and processing iteration result
 */
public final class LayerFunction {
    private LayerFunction() {}

    @FunctionalInterface
    interface Update {
        void apply(final double[] input, final Layer layer, final double[] error, final double learningRate);
    }

    static final Update DenseUpdate = (inputs, layer, errors, learningRate) -> {
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

    static final LayerFunction.Update LeakyReluUpdate = (inputs, layer, errors, learningRate) -> {
        /* LeakyRelu has no weight to update */
    };
    //================================================================

    static final LayerFunction.Update JudgeUpdate = (inputs, layer, errors, learningRate) -> {
        /* Judge has no weight to update */
    };

    //================================================================
    static final NRandom random = new NRandom(System.nanoTime());

    // According to the source code, it has been used as initializer
    static void fillWithGaussianRandom(final Layer layer) {
        final double[] input = layer.data;
        final int inputSize = input.length;
        for (int i = 0; i < inputSize; i++)
            input[i] = random.nextGaussian(0, 1.0/ Math.sqrt(inputSize));
    }

    /** It seems that it haven't been use. */
    static void fillWithRandom(final Layer layer, final double rv) {
        final int inputSize = layer.getInputSize();
        final int outputSize = layer.getOutputSize();
        final double[] data = layer.data;
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                data[i * j] = random.nextGaussian(1 - rv, 1 + rv);
            }
        }
    }
}
