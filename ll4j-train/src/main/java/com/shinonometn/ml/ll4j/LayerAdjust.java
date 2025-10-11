package com.shinonometn.ml.ll4j;

import static com.shinonometn.ml.ll4j.Layers.*;

public class LayerAdjust {
    final Layer layer;

    final double[] outputState;

    final LayerFunction.Update function;

    LayerAdjust(Layer layer, LayerFunction.Update update) {
        this.layer = layer;
        outputState = new double[layer.getOutputSize()];
        this.function = update;
    }

    //================================================================

    static LayerAdjust createAdjuster(Layer layer) {
        final LayerFunction.Update updater;
        switch (layer.type) {
            case TYPE_DENSE:
                updater = LayerFunction.DenseUpdate;
                break;
            case TYPE_JUDGE:
                updater = LayerFunction.JudgeUpdate;
                break;
            case TYPE_LEAKY_RELU:
                updater = LayerFunction.LeakyReluUpdate;
                break;
            default:
                throw new IllegalArgumentException(String.format("Unknown layer type %s", layer.type));
        }
        return new LayerAdjust(layer, updater);
    }
}
