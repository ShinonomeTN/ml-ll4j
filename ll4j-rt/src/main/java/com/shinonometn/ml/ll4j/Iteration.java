package com.shinonometn.ml.ll4j;

/**
 * Represent an iteration result
 */
public class Iteration {

    /** The reference to a matrix */
    final double[] result;

    Iteration(double[] result) {
        this.result = result;
    }

    /** Check if it is referring the same array */
    boolean isResultRefSame(Iteration other) {
        return this.result == other.result;
    }

    Iteration copy() {
        final double[] copy = new double[result.length];
        System.arraycopy(result, 0, copy, 0, result.length);
        return new Iteration(copy);
    }
}
