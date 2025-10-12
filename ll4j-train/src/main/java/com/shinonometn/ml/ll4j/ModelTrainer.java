package com.shinonometn.ml.ll4j;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;

import static com.shinonometn.ml.ll4j.Layers.*;

public class ModelTrainer {
    final LayerAdjust[] adjusters;
    final Model model;

    private final HashMap<Object, double[]> outputCache = new HashMap<>();
    private final HashMap<Object, double[]> errorCache = new HashMap<>();


    private final Object InputKey;

    /**
     * get model input array cache, it's not null
     */
    private double[] getInput() {
        return outputCache.get(InputKey);
    }


    private final Object AnswerKey;

    /**
     * get model output array cache, it's not null
     */
    private double[] getAnswer() {
        return outputCache.get(AnswerKey);
    }

    ModelTrainer(LayerAdjust[] adjusters, Model model) {

        this.adjusters = adjusters;
        this.model = model;

        // Create input
        final double[] input = new double[model.getInputSize()];
        this.InputKey = input;
        outputCache.put(InputKey, input);

        // Create answer
        final double[] output = new double[model.getOutputSize()];
        this.AnswerKey = output;
        outputCache.put(AnswerKey, output);
    }

    void setLabeledData(final DataSet.Entry dataEntry) {
        final double[] input = dataEntry.values;
        final int inputSize = input.length;
        final double[] modelInput = getInput();
        if (inputSize != getInput().length) throw new IllegalArgumentException(String.format(
                "Input data size does not equal to the model input size: %d, expected: %d",
                inputSize, modelInput.length
        ));

        final int outputSize = dataEntry.getLabelLength();
        final double[] answer = getAnswer();
        if (outputSize != answer.length) throw new IllegalArgumentException(String.format(
                "Label dimensions does not equals to the model output: %d, expected: %d",
                outputSize, answer.length
        ));

        // Set input
        System.arraycopy(input, 0, modelInput, 0, inputSize);

        // Set the answer
        dataEntry.toValues(answer);
    }

    public Model toModel() {
        final Layer[] layers = new Layer[adjusters.length];
        for (int i = 0; i < layers.length; i++) {
            layers[i] = adjusters[i].layer;
        }
        return new Model(layers);
    }

    //================================================================

    private int correctCount = 0;

    public int getCorrectCount() {
        return correctCount;
    }

    private int wrongCount = 0;

    public int getWrongCount() {
        return wrongCount;
    }

    public int getIterationCount() {
        return correctCount + wrongCount;
    }

    public void resetCounters() {
        correctCount = 0;
        wrongCount = 0;
    }

    //================================================================
    abstract static class Step {

        /**
         * The reference to this step's result cache
         */
        abstract double[] getValues();

        protected ModelTrainer ctx;

        /**
         * A network tweaking step.
         * <p>
         * It's a handy reference to each element in network training.
         */
        protected Step(final ModelTrainer ctx) {
            this.ctx = ctx;
        }

        /**
         * The network input.
         */
        static final class Input extends Step {
            Input(final ModelTrainer trainer) {
                super(trainer);
            }

            @Override
            double[] getValues() {
                return ctx.getInput();
            }
        }

        /**
         * Weight & bias adjustment steps.
         */
        static final class Adjust extends Step {
            /**
             * The corresponding tweaker of the result.
             */
            final LayerAdjust tweaker;

            /**
             * The reference to tweaker's error value cache.
             * Related to the current layer, it's the input error.
             */
            double[] getErrors() {
                return ctx.errorCache.computeIfAbsent(
                        /*     key = */ tweaker,
                        /* factory = */k -> AdjustFunctions
                                .fillWithZero(new double[tweaker.layer.getInputSize()])
                );
            }

            @Override
            double[] getValues() {
                return ctx.outputCache.computeIfAbsent(
                        /*     key = */ tweaker,
                        /* factory = */ k -> AdjustFunctions
                                .fillWithZero(new double[tweaker.layer.getOutputSize()])
                );
            }


            Adjust(final ModelTrainer trainer, final LayerAdjust tweaker) {
                super(trainer);
                this.tweaker = tweaker;
            }
        }
    }

    public void adjust(final DataSet.Entry entry) {
        adjust(entry, DefaultLearningRate);
    }

    /**
     * Run a single adjust iteration, with a learning rate.
     *
     * @param entry        A data entry, with the sample data and a correct label
     * @param learningRate learning rate of this network
     */
    public void adjust(final DataSet.Entry entry, double learningRate) {
        // Check the learning rate
        if (Double.isNaN(learningRate) || learningRate <= 0.0) {
            learningRate = DefaultLearningRate;
        }

        // Set data
        setLabeledData(entry);

        // Set input
        // The first layer is always an input layer
        final LinkedList<Step> steps = new LinkedList<>();
        steps.push(new Step.Input(this));

        /*
         * Forward propagation
         * ===
         * Each `tweaker` holds a layer. Layers are holding weights and the forward propagation function.
         *
         * Each result the layer produces are stored in the `results` list for convenience, since each
         * layer's output is the input for next layer.
         *
         * And the latest layer's output is the final result. We need those indeterminate results for the
         * network updating.
         */
        for (final LayerAdjust adjuster : adjusters) {
            final Layer currentLayer = adjuster.layer;

            final Step currentStep = new Step.Adjust(this, adjuster);
            final double[] input = steps.getFirst().getValues();
            final double[] output = currentStep.getValues();

            currentLayer.function.apply(
                    /* Outputs of the latest layer  */ input,
                    /* Weights of the current layer */ currentLayer.data,
                    /* Destination of the outputs   */ output
            );

            steps.push(currentStep);
        }

        // log the correct count
        final Step outputStep = steps.getFirst();
        final double[] outputResults = outputStep.getValues();
        final double[] expectedResults = getAnswer();
        final boolean isCorrect = Arrays.equals(expectedResults, outputResults);
        if (isCorrect) correctCount++;
        else wrongCount++;

        // Set the correct answer to the network
        // In back propagation, the answer becomes the input
        System.arraycopy(expectedResults, 0, outputResults, 0, outputResults.length);

        /*
         * Backward propagation
         * ===
         * The backward propagation is just reverse the steps of the origin calculation
         * (though some step needs a different algorithm). Use the benefit of stack, to
         * do calculation and update together.
         */
        double[] upperError = expectedResults;
        Step i = steps.pop();
        while ((i instanceof Step.Adjust) && !steps.isEmpty()) {
            final Step.Adjust currentStep = (Step.Adjust) i;
            final Layer currentlayer = currentStep.tweaker.layer;
            final double[] lowerError = currentStep.getErrors();

            final Step nextStep = steps.getFirst();
            final double[] input = nextStep.getValues();

            currentStep.tweaker.function.apply(
                    /*     input = */ input,
                    /*     layer = */ currentlayer,
                    /*    errors = */ upperError,
                    /*    output = */ lowerError
            );

            currentStep.tweaker.updater.apply(
                    /*        input = */ input,
                    /*        layer = */ currentlayer,
                    /*       errors = */ upperError,
                    /* learningRate = */ learningRate
            );

            // Current layer's input error is the previous layer's output error
            upperError = lowerError;
            i = steps.pop();
        }
    }
    //================================================================

    /**
     * Save the model to file
     */
    public void writeModelToFile(String path) throws IOException {
        try (final PrintWriter writer = new PrintWriter(path)) {
            for (final LayerAdjust adjuster : adjusters) {
                final Layer layer = adjuster.layer;
                switch (layer.type) {
                    case TYPE_DENSE: {
                        writer.printf("D %d %d ", layer.getInputSize(), layer.getOutputSize());
                        for (int i = 0; i < layer.getInputSize(); i++) {
                            for (int k = 0; k < layer.getOutputSize(); k++) {
                                writer.print(layer.data[i * k]);
                                writer.print(" ");
                            }
                        }
                        break;
                    }

                    case TYPE_LEAKY_RELU: {
                        writer.printf("L %d%n", layer.getInputSize());
                        break;
                    }

                    case TYPE_JUDGE: {
                        writer.printf("J %d%n", layer.getInputSize());
                        break;
                    }
                }
                writer.println();
                writer.flush();
            }
        }
    }

    //================================================================

    /**
     * Create a ModelTrainer from a set of layers
     */
    public static ModelTrainer create(Layer... layers) {
        final Model model = new Model(layers);
        final LayerAdjust[] adjusters = createAdjustersForLayers(layers);
        return new ModelTrainer(adjusters, model);
    }

    /**
     * Create a ModelTrainer on a Model
     */
    public static ModelTrainer on(final Model model) {
        final LayerAdjust[] adjusts = createAdjustersForLayers(model.layers);
        return new ModelTrainer(adjusts, model);
    }

    // Helper to create adjuster for each layers
    static LayerAdjust[] createAdjustersForLayers(Layer[] layers) {
        final LayerAdjust[] adjusts = new LayerAdjust[layers.length];
        for (int i = 0; i < adjusts.length; i++) {
            adjusts[i] = LayerAdjust.createAdjuster(layers[i]);
        }
        return adjusts;
    }

    //================================================================
    /**
     * The default learning rate.
     * <p>
     * Why `8e-7`(0.0000008)? I have no idea. This value is from the origin LL4J codes.
     */
    public static final double DefaultLearningRate = 8e-7;
}
