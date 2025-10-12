package com.shinonometn.ml.ll4j;

import static com.shinonometn.ml.ll4j.Layers.*;

public class LayerAdjust {
    /**
     * Layer to be adjusted
     */
    final Layer layer;

    public int getOutputSize() {
        return layer.getOutputSize();
    }

    public int getInputSize() {
        return layer.getInputSize();
    }

    /**
     * Backward propagation function
     */
    final BackwardFunction function;

    /**
     * The layer update method
     */
    final Updater updater;

    LayerAdjust(Layer layer, Updater updater, BackwardFunction function) {
        this.layer = layer;

        this.updater = updater;
        this.function = function;
    }

    @FunctionalInterface
    public interface Updater {

        /**
         * Apply update to a layer
         *
         * @param layer        The layer that containing weights to be updated
         * @param input        Input of this layer, that says, the output of the upper layer.
         * @param error        Errors that to be adjusts to
         * @param learningRate How aggressive the layer update should be
         */
        void apply(final double[] input, final Layer layer, final double[] error, final double learningRate);
    }

    //================================================================

    static LayerAdjust createAdjuster(Layer layer) {
        final Updater updater;
        final BackwardFunction function;
        switch (layer.type) {
            case TYPE_DENSE:
                updater = AdjustFunctions.DenseUpdate;
                function = BackwardFunction.Dense;
                break;
            case TYPE_JUDGE:
                updater = AdjustFunctions.JudgeUpdate;
                function = BackwardFunction.MaxIndex;
                break;
            case TYPE_LEAKY_RELU:
                updater = AdjustFunctions.LeakyReluUpdate;
                function = BackwardFunction.LeakyRelu;
                break;
            default:
                throw new IllegalArgumentException(String.format("Unknown layer type %s", layer.type));
        }
        return new LayerAdjust(layer, updater, function);
    }
}
