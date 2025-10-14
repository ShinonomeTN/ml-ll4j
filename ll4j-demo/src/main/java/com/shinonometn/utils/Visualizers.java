package com.shinonometn.utils;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;

public final class Visualizers {
    private Visualizers() {
    }

    //================================================================

    public static void dumpSampleToGreyScaleImageFile(
            final SampleVisualizingParams samples,
            final Path destination
    ) throws IOException {
        final BufferedImage img = convertSampleToGreyScaleImage(samples);
        ImageIO.write(img, "png", destination.toFile());
    }

    public static BufferedImage convertSampleToGreyScaleImage(
            final SampleVisualizingParams samples
    ) {
        final int width = samples.width;
        final int height = samples.height;
        final double[] data = samples.samples;
        if (width == 0 || height == 0) throw new IllegalArgumentException("width or height cannot be zero");
        if (width * height > data.length) throw new IllegalArgumentException(
                "Given dimension is not matched to the sample"
        );

        // scan the max and min value of the sample
        double min = data[0];
        double max = data[0];
        for (int i = 1; i < data.length; i++) {
            if (data[i] > max) max = data[i];
            if (data[i] < min) min = data[i];
        }
        // normalized to zero base
        final double base = 0 - min;
        max -= base;
        final BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        samples.rowFormat.fill(img, data, width, height, base, max);
        return img;
    }
}
