package com.shinonometn.ml.ll4j;

public final class Layers {
    private Layers() {
    }

    //================================================================

    public final static String TYPE_DENSE = "D";

    public static Layer dense(final int input, final int output) {
        final double[] data = new double[input * output];
        return dense(input, output, data);
    }

    public static Layer dense(final int input, final int output, final double[] weights) {
        return dense(input, output, weights, ForwardFunction.Dense);
    }

    public static Layer dense(final int input, final int output, final double[] weights, final ForwardFunction function) {
        if (weights.length != (input * output)) throw new IllegalArgumentException(String.format(
                "Size of the weight array is not equals to the capacity. (%d * %d != %d)",
                input, output, weights.length
        ));
        return new Layer(TYPE_DENSE, new int[]{input, output}, weights, function);
    }

    public static Layer dense(final int input, final int output, final ForwardFunction function) {
        return dense(input, output, new double[input * output], function);
    }

    //================================================================

    public final static String TYPE_JUDGE = "J";

    /** Same as maxIndex */
    public static Layer judge(final int size) {
        return new Layer(TYPE_JUDGE, new int[]{size, 1}, new double[0], ForwardFunction.Judge);
    }

    /** Select the max value's index as output */
    public static Layer maxIndex(final int size) {
        return new Layer(TYPE_JUDGE, new int[]{size, 1}, new double[0], ForwardFunction.MaxIndex);
    }

    //================================================================

    public final static String TYPE_LEAKY_RELU = "L";

    public static Layer leakyRelu(final int size) {
        return new Layer(TYPE_LEAKY_RELU, new int[]{size, size}, new double[0], ForwardFunction.LeakyRelu);
    }

}
