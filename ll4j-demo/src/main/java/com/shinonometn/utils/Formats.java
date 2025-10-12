package com.shinonometn.utils;

public class Formats {
    public static String millisDuration(long ms) {
        return String.format("%d:%02d:%02d.%03d", (ms / 1000) / 3600, ((ms / 1000) % 3600) / 60, ((ms / 1000) % 60), ms % 1000);
    }
}
