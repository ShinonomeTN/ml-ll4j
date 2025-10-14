package com.shinonometn.utils;

import java.awt.image.BufferedImage;

public final class SampleVisualizingParams {
    final int width;
    final int height;
    final RowFormat rowFormat;
    final double[] samples;

    public enum RowFormat {
        RowCol {
            @Override
            void fill(BufferedImage c, final double[] data, int rows, int cols, double base, double max) {
                for (int x = 0; x < cols; x++) {
                    for (int y = 0; y < rows; y++) {
                        final double sample = data[x * rows + y];
                        final int rgb = (int) (((sample - base) / max) * 255);
                        c.setRGB(x, y, rgb << 16 | rgb << 8 | rgb);
                    }
                }
            }
        },
        ColRow {
            @Override
            void fill(BufferedImage c, final double[] data, int rows, int cols, double base, double max) {
                for (int y = 0; y < rows; y++) {
                    for (int x = 0; x < cols; x++) {
                        final double sample = data[x + (y * cols)];
                        final int rgb = (int) (((sample - base) / max) * 255);
                        c.setRGB(x, y, rgb << 16 | rgb << 8 | rgb);
                    }
                }
            }
        };

        abstract void fill(
                final BufferedImage c,
                final double[] data ,final int rows, final int cols,
                final double base, final double max
        );
    }

    private SampleVisualizingParams(int width, int height, double[] samples, RowFormat rowFormat) {
        this.width = width;
        this.height = height;
        this.rowFormat = rowFormat;
        this.samples = samples;
    }

    public static SampleVisualizingParams of(int width, int height, double[] samples, RowFormat rowFormat) {
        return new SampleVisualizingParams(width, height, samples, rowFormat);
    }

    public static SampleVisualizingParams colFirst(int width, int height, double[] samples) {
        return of(width, height, samples, RowFormat.RowCol);
    }

    public static SampleVisualizingParams rowFirst(int width, int height, double[] samples) {
        return of(width, height, samples, RowFormat.ColRow);
    }
}
