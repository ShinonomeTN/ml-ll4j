package com.shinonometn.ml.ll4j;

import java.util.LinkedList;
import java.util.Objects;

public class ModelTrainer {
    final LayerAdjust[] tweakers;
    final double[] input;
    final Model model;

    ModelTrainer(LayerAdjust[] tweakers, Model model) {
        if (!Objects.equals(tweakers[tweakers.length - 1].layer.type, Layers.TYPE_JUDGE))
            throw new IllegalArgumentException("Output layer is not JudgeLayer");

        this.tweakers = tweakers;
        this.model = model;
        this.input = new double[model.getInputSize()];
    }

    public void setInput(final double[] input) {
        final int inputSize = input.length;
        if (inputSize != this.input.length) throw new IllegalArgumentException("input size != input size");
        System.arraycopy(input, 0, this.input, 0, inputSize);
    }

    public Model toModel() {
        final Layer[] layers = new Layer[tweakers.length];
        for (int i = 0; i < layers.length; i++) {
            layers[i] = tweakers[i].layer;
        }
        return new Model(layers);
    }

    //================================================================

    public static class AdjustResult {
        public final boolean correct;

        public AdjustResult(boolean correct) {
            this.correct = correct;
        }
    }

    void adjust(final DataSet.LabeledEntry entry, final double[] learningRate) {
        final double[] input = entry.values;
        final int inputSize = input.length;
        if (inputSize != this.input.length) throw new IllegalArgumentException(
                "Data size not equals to the model input"
        );
        // Set input
        final LinkedList<Iteration> results = new LinkedList<>();
        results.push(new Iteration(input));

        for (final LayerAdjust tweaker : tweakers) {
            final Iteration current = new Iteration(tweaker.outputState);
            final Layer layer = tweaker.layer;
            layer.function.forward(results.getLast().result, layer.data, current.result);
            results.add(current);
        }

        final int actualLabel = entry.label;
        final int predictedLabel = Matrix.maxIndex(results.getLast().result);
        final boolean isCorrect = (actualLabel == predictedLabel);

        final LinkedList<Iteration> errors = new LinkedList<>();
        for (int i = tweakers.length - 1; i >= 0; i--) {
            final Iteration current = results.pop();
            final LayerAdjust tweaker = tweakers[i];
            final Layer layer = tweaker.layer;
            errors.add(new Iteration(new double[layer.getInputSize()]));
            layer.function.backward(current.result, layer.data, errors.getLast().result);
        }
    }

    //================================================================

    static LayerAdjust[] adjusterForLayers(Layer[] layers) {
        final LayerAdjust[] adjusts = new LayerAdjust[layers.length];
        for (int i = 0; i < adjusts.length; i++) {
            adjusts[i] = LayerAdjust.createAdjuster(layers[i]);
        }
        return adjusts;
    }

    public static ModelTrainer create(Layer... layers) {
        final Model model = new Model(layers);
        final LayerAdjust[] adjusts = adjusterForLayers(layers);
        return new ModelTrainer(adjusts, model);
    }

    public static ModelTrainer fromModel(final Model model) {
        final LayerAdjust[] adjusts = adjusterForLayers(model.layers);
        return new ModelTrainer(adjusts, model);
    }
}
