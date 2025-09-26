package com.shinonometn.ml.ll4j;

/**
 * Represent an iteration result
 */
public class Iteration {
    final double[] result;

    Iteration(double[] result) {
        this.result = result;
    }

    Iteration copy() {
        final double[] copy = new double[result.length];
        System.arraycopy(result, 0, copy, 0, result.length);
        return new Iteration(copy);
    }
}
