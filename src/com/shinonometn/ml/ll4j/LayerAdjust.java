package com.shinonometn.ml.ll4j;

public class LayerAdjust {
    final Layer layer;

    final double[] outputState;

    final LayerFunction.Update function;

    LayerAdjust(Layer layer, LayerFunction.Update update) {
        this.layer = layer;
        outputState = new double[layer.getOutputSize()];
        this.function = update;
    }
}
