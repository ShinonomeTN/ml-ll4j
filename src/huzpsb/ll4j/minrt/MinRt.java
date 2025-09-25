// MinRt is a minimal runtime for LL4J deep learning framework
// While MinRt is as small as possible, it is SLOW and INEFFICIENT.
// Use MinRt only when you need to run LL4J models in a memory-constrained environment.
// Copyright (c) 2024 huzpsb [admin<at>huzpsb<dot>eu<dot>org]
// Licensed under the WTFPL license. You may remove this notice at will.

package huzpsb.ll4j.minrt;

import java.util.LinkedList;

public final class MinRt {
    private MinRt() {
        /* All computational method should be static. We don't do OOP here. */
    }

    /**
     * Model is a cache for each classification job
     */
    public static class Model {
        final Layer[] layers;

        Model(Layer[] layers) {
            this.layers = layers;
        }

        public int getInputSize() {
            return layers[0].getInputSize();
        }
    }

    /**
     * Layer is a part of the Model
     */
    static class Layer {
        final String type;
        final int[] meta;
        final double[] data;
        final LayerFunction function;

        public Layer(String type, int[] meta, double[] data, LayerFunction function) {
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

    public static Model parseModel(String[] model) throws MinRtException {
        final LinkedList<Layer> layers = new LinkedList<>();

        for (final String s : model) {
            // Ignore empty layer
            // layer data is type + size(vararg) + data(vararg)
            if (s.length() < 2) continue;

            final String[] tokens = s.split(" ");

            int pos = 0; // Position of reader

            // Read the type first
            final String type = tokens[pos++];

            // Get input count, the second field is always input count
            final int inputCount = Integer.parseInt(tokens[pos++]); // Input Count
            if (!layers.isEmpty()) { // If is not the first layer, check the input size
                final int inputSize = layers.getLast().getOutputSize();
                if (inputSize != inputCount) throw new LayerInputMismatchException(String.format(
                        "Wrong input size for layer %s@%02d (expected %d, got %d)",
                        type, layers.size(), inputCount, inputSize
                ));
            }

            switch (type) {
                // Dense Layer
                // Output count at 2
                case "D": {
                    final int outputCount = Integer.parseInt(tokens[pos++]);
                    final int[] meta = new int[]{inputCount, outputCount};

                    final double[] data = new double[tokens.length - pos];
                    for (int i = 0; i < data.length; i++)
                        data[i] = Double.parseDouble(tokens[pos++]);

                    layers.add(new Layer(type, meta, data, DenseLayerFunction));
                    break;
                }

                // For "LeakyRelu Layer"
                case "L": {
                    final int[] meta = new int[]{inputCount, inputCount}; // Input is same as output
                    final double[] data = new double[0];
                    layers.add(new Layer(type, meta, data, LeakyReluFunction));
                    break;
                }
                // For "Judge Layer"
                case "J": {
                    final int[] meta = new int[]{inputCount, 1}; // It only output a single number
                    layers.add(new Layer(type, meta, new double[1], JudgeLayerFunction));
                    break;
                }

                default: {
                    throw new UnsupportedLayerTypeException(type);
                }
            }
        }

        return new Model(layers.toArray(new Layer[0]));
    }

    static class Iteration {
        final double[] result;

        Iteration(final double[] result) {
            this.result = result;
        }
    }

    @FunctionalInterface
    interface LayerFunction {
        Iteration apply(final Iteration last, final Layer layer);
    }

    static final LayerFunction DenseLayerFunction = (last, layer) -> {
        final double[] input = last.result;
        final int inputSize = input.length;

        final double[] matrix = layer.data;
        final int outputSize = layer.getOutputSize();

        final double[] result = new double[outputSize];

        for (int idxO = 0; idxO < outputSize; idxO++) {
            double sum = 0;
            for (int idxI = 0; idxI < inputSize; idxI++) {
                sum += input[idxI] * matrix[idxO + idxI * outputSize];
            }
            result[idxO] = sum;
        }
        return new Iteration(result);
    };

    /**
     * What a LeakyRelu layer do is just check if each value is greater than 0
     */
    static final LayerFunction LeakyReluFunction = (last, layer) -> {
        final double[] input = last.result;
        final int inputSize = input.length;

        for (int i = 0; i < inputSize; i++) {
            input[i] = input[i] > 0 ? input[i] : input[i] * 0.01;
        }

        return last;
    };

    /**
     * What a Judge layer do is select the greatest value of the result.
     */
    static final LayerFunction JudgeLayerFunction = (last, layer) -> {
        final double[] input = last.result;
        final int inputSize = input.length;

        int idx = 0;
        for (int i = 1; i < inputSize; i++) {
            if (input[i] > input[idx]) idx = i;
        }
        return new Iteration(new double[]{idx});
    };

    public static double[] classification(double[] input, Model model) throws MinRtException {
        final int inputSize = model.getInputSize();
        if (input.length != inputSize) throw new MinRtException(String.format(
                "Wrong input size, expected %d, got %d", inputSize, input.length
        ));

        Iteration iter = new Iteration(input);
        for (Layer layer : model.layers) {
            iter = layer.function.apply(iter, layer);
        }
        return iter.result;
    }

    public static class MinRtException extends Exception {
        public MinRtException() {
            super();
        }

        public MinRtException(String message) {
            super(message);
        }
    }

    public static final class UnsupportedLayerTypeException extends MinRtException {
        public UnsupportedLayerTypeException(String message) {
            super(message);
        }
    }

    public static final class LayerInputMismatchException extends MinRtException {
        public LayerInputMismatchException(String message) {
            super(message);
        }
    }

    public static int doAi(double[] input, String[] script) {
        double[] current = new double[input.length];
        System.arraycopy(input, 0, current, 0, input.length);

        for (String str : script) {
            if (str.length() < 2) {
                continue;
            }
            String[] tokens = str.split(" ");
            switch (tokens[0]) {
                case "D":
                    int ic = Integer.parseInt(tokens[1]);
                    int oc = Integer.parseInt(tokens[2]);
                    if (current.length != ic) {
                        throw new RuntimeException("Wrong input size for Dense layer (expected " + ic + ", got " + current.length + ")");
                    }
                    double[] tmp = new double[oc];
                    for (int oIdx = 0; oIdx < oc; oIdx++) {
                        double sum = 0;
                        for (int iIdx = 0; iIdx < ic; iIdx++) {
                            sum += current[iIdx] * Double.parseDouble(tokens[3 + oIdx + iIdx * oc]);
                        }
                        tmp[oIdx] = sum;
                    }
                    current = tmp;
                    break;
                case "L":
                    int n = Integer.parseInt(tokens[1]);
                    if (current.length != n) {
                        throw new RuntimeException("Wrong input size for LeakyRelu layer (expected " + n + ", got " + current.length + ")");
                    }
                    for (int i = 0; i < n; i++) {
                        current[i] = current[i] > 0 ? current[i] : current[i] * 0.01;
                    }
                    break;
                case "J":
                    int m = Integer.parseInt(tokens[1]);
                    if (current.length != m) {
                        throw new RuntimeException("Wrong input size for Judge layer (expected " + m + ", got " + current.length + ")");
                    }
                    int idx = 0;
                    for (int i = 1; i < m; i++) {
                        if (current[i] > current[idx]) {
                            idx = i;
                        }
                    }
                    return idx;
                default:
                    throw new RuntimeException("Unknown layer type");
            }
        }
        throw new RuntimeException("No output layer");
    }
}
