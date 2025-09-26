package com.shinonometn.ml.ll4j;

public final class Layers {
    private Layers() {
    }

    final static String TYPE_DENSE = "D";

    public static Layer dense(final int input, final int output) {
        final double[] data = new double[input * output];
        return dense(input, output, data);
    }

    public static Layer dense(final int input, final int output, final double[] weights) {
        if (weights.length != (input * output)) throw new IllegalArgumentException(String.format(
                "Size of the weight array is not equals to the capacity. (%d * %d != %d)",
                input, output, weights.length
        ));
        return new Layer(TYPE_DENSE, new int[]{input, output}, weights, Matrix.Transform.Dense);
    }

    final static String TYPE_JUDGE = "J";

    public static Layer judge(final int size) {
        return new Layer(TYPE_JUDGE, new int[]{size, size}, new double[0], Matrix.Transform.Judge);
    }

    final static String TYPE_LEAKY_RELU = "L";

    public static Layer leakyRelu(final int size) {
        return new Layer(TYPE_LEAKY_RELU, new int[]{size, size}, new double[0], Matrix.Transform.LeakyRelu);
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
