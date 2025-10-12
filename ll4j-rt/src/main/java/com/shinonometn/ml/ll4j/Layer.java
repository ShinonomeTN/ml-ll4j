package com.shinonometn.ml.ll4j;

/**
 * Layer is a part of the Model
 */
public class Layer {
    public final String type;
    final int[] meta;
    final double[] data;
    final ForwardFunction function;

    Layer(final String type, final int[] meta, final double[] data, final ForwardFunction function) {
        this.type = type;
        this.data = data;
        this.meta = meta;
        this.function = function;
    }

    int getInputSize() {
        return meta[0];
    }

    int getOutputSize() {
        return meta[1];
    }
}
