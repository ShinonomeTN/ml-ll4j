package com.shinonometn.ml.ll4j.demo;

import com.shinonometn.ml.ll4j.Layer;
import com.shinonometn.ml.ll4j.Layers;
import com.shinonometn.ml.ll4j.MinRtException;
import com.shinonometn.ml.ll4j.Model;
import com.shinonometn.utils.Loaders;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

public class VisualizeDenseLayer {
    private final static String MODEL_PATH = "digits/test.model";

    public static void main(String[] args) throws IOException, MinRtException {
        final Model model = Model.parseLayers(Loaders.loadModelString(MODEL_PATH));
        final Path outputFolder = Paths.get("./layer_visualized");
        if (outputFolder.toFile().mkdirs()) System.out.println(
                "Created output folder: " + outputFolder.toAbsolutePath()
        );
        dumpModelToGreyScaleImageFile(model, 0.5, outputFolder);
    }

    public static void dumpModelToGreyScaleImageFile(
            final Model model, final double rowRatio, final Path destination
    ) throws IOException {
        final List<Layer> denseLayers = model.getLayers().stream()
                .filter(l -> Objects.equals(l.type, Layers.TYPE_DENSE))
                .collect(Collectors.toList());

        int layerIndex = 0;
        for (Layer layer : denseLayers) {
            final double[] data = layer.getData();

            final int size = layer.getInputSize();
            final int total = layer.getOutputSize();

            final int rows = (int) Math.floor(((Math.sqrt(size) * 2) * rowRatio));

            // find the weight's max and min in layer
            double wMax = data[0];
            double wMin = data[0];
            for (double datum : data) {
                if (wMax < datum) wMax = datum;
                if (wMin > datum) wMin = datum;
            }
            // normalize
            final double base = 0 - wMin;
            final double wAdj = wMax + base;
            System.out.printf("Layer %d, max: %f, min: %f, base: %f, maxAdj: %f%n", layerIndex, wMax, wMin, base, wAdj);

            // For each connection output
            for (int idxO = 0; idxO < total; idxO++) {
                // start positon of input
                final int offset = idxO * size;

                // Get row and col count
                int cols = size / rows;
                final int remains = size % rows;
                if (remains != 0) cols += 1;

                final BufferedImage image = new BufferedImage(rows, cols, BufferedImage.TYPE_BYTE_GRAY);

                // for each connection input
                for (int x = 0; x < cols; x++) {
                    for (int y = 0; y < rows; y++) {
                        final int index = offset + (y * rows + x);
                        if (index >= data.length) continue;

                        // Reverse the color
                        final int color = (int) ((1 - (data[index] - base) / wAdj) * 255);
                        image.setRGB(x, y, color << 16 | color << 8 | color);
                    }
                }
                final File target = destination.resolve(String.format("L%03d_N%03d.png", layerIndex, idxO)).toFile();
                ImageIO.write(image, "png", target);
            }

            layerIndex++;
        }
    }
}
