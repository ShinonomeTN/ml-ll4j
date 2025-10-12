package huzpsb.ll4j.model;

import huzpsb.ll4j.data.DataEntry;
import huzpsb.ll4j.data.DataSet;
import huzpsb.ll4j.layer.AbstractLayer;
import huzpsb.ll4j.layer.JudgeLayer;

import java.io.PrintWriter;

public class Model {
    public final AbstractLayer[] layers;

    public Model(AbstractLayer... layers) {
        this.layers = layers;
    }

    public void trainOn(DataSet dataSet) {
        int t = 0, f = 0;
        if (!(layers[layers.length - 1] instanceof JudgeLayer)) {
            throw new RuntimeException("Last layer is not output layer");
        }
        final JudgeLayer judgeLayer = (JudgeLayer) layers[layers.length - 1];

        for (DataEntry dataEntry : dataSet.split) {
            layers[0].input = dataEntry.values;

            // Catten Linger:
            // Forward propergation started.
            for (int i = 0; i < layers.length; i++) {
                // Invoke the forward method
                //
                // Each `forward()` method sets output value to the `output` array.
                layers[i].forward();

                if (i < layers.length - 1) {
                    // Then, it copies the `output` array to the `input` array in
                    // the next layer.
                    //
                    // Why the `layers.length - 1` up there? The latest layer in array is
                    // the output layer, there is no more layer in follows.
                    layers[i + 1].input = layers[i].output;
                }
            }

            // Catten Linger:
            // Here is just for logging the correct rate.
            final int predict = judgeLayer.result;
            final int actual = dataEntry.type;
            if (predict == actual) t++;
            else f++;

            // Set the actual value to the network output, prepare for the back propagation.
            // In the back propagation, the input becomes the output.
            //
            // It seems that no matter is the result correct or wrong, the back propagation
            // is still be process.
            judgeLayer.result = actual;

            // Catten Linger:
            // Backward propagation started.
            for (int i = layers.length - 1; i >= 0; i--) {
                /*
                 * Invoke the backward method
                 *
                 * All `backward()` methods will invoke the `makeInputError()` at first,
                 * just for ensure the `input_error` field is not null
                 * if `input_error` is null, it will create a new array has the same size
                 * as the input(inputSize).
                 *
                 * The `input_error`, `output_error` and `input` are used in this function.
                 * The process takes `output_error`, apply weights on it, and set it to
                 * `input_error`.
                 */
                layers[i].backward();

                // Invoke the update method with a learning rate.
                // Why 0.0000008? idk.
                //
                // In general, the update function takes the
                layers[i].update(8e-7);

                // Only propagate errors to internal layers.
                // Why the `i > 0`? Because there is no layer before the latest
                // layer, nowhere to propagate.
                // The input layer is a virtual layer in this implementation.
                if (i > 0)
                    // Each layer's output error is the lower layer's input error.
                    layers[i - 1].output_error = layers[i].input_error;
            }
        }
        System.out.println("t: " + t + ", f: " + f);
    }

    public void testOn(DataSet dataSet) {
        int t = 0, f = 0;
        for (DataEntry dataEntry : dataSet.split) {
            layers[0].input = dataEntry.values;
            for (int i = 0; i < layers.length; i++) {
                layers[i].forward();
                if (i < layers.length - 1) {
                    layers[i + 1].input = layers[i].output;
                }
            }
            JudgeLayer judgeLayer = (JudgeLayer) layers[layers.length - 1];
            int predict = judgeLayer.result;
            int actual = dataEntry.type;
            if (predict == actual) {
                t++;
            } else {
                f++;
            }
        }
        System.out.println("t: " + t + ", f: " + f);
    }

    public void save(String path) {
        try {
            StringBuilder sb = new StringBuilder();
            for (AbstractLayer layer : layers) {
                layer.serialize(sb);
            }
            PrintWriter pw = new PrintWriter(path);
            pw.print(sb);
            pw.close();
        } catch (Exception ex) {
            throw new RuntimeException(ex);
        }
    }
}
