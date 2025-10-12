package com.shinonometn.ml.ll4j;


public final class Matrix {
    private Matrix() {}

    public static int maxIndex(double[] input) {
        int result = 0;
        for (int i = 1; i < input.length; i++) {
            if (input[i] > input[i - 1]) result = i;
        }
        return result;
    }
}
