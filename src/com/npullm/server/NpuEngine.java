package com.npullm.server;

public class NpuEngine {
    static { System.loadLibrary("npubridge"); }
    public native int nativeInit(String modelPath, String backend,
            String cacheDir, String dispatchDir, int maxTokens);
    public native String nativeGenerate(String prompt);
    public native void nativeClose();
    public native String nativeLastError();
}
