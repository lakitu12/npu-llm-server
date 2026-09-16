// ShellEngine: runs inference in a shell-namespace child process.
// The LiteRT-LM engine must load the MTK NeuronAdapter (libneuronusdk_adapter.mtk.so),
// which lives in /vendor/lib64 but is NOT in any App-visible public library list.
// A shell-namespace process (started via `adb shell` / run-as wrapper) CAN load it;
// an App-namespace process cannot. So the App never inits the engine itself:
// it spawns `lm_shell` (or litert_lm_advanced_main) as a child via ProcessBuilder
// with the shell linker namespace inherited... except: a plain App-spawned child
// still runs in the App SELinux context + linker namespace.
//
// Proven working path (verified 2026-09-16 on dash/mt6991):
//   * `adb shell` -> shell SELinux context + shell linker namespace
//   * LD_LIBRARY_PATH=/data/local/tmp/npullm ./lm_shell mt6991.litertlm npu ...
//   * NPU delegate loads, Neuron SDK 8.2.26, prefill_128, init ~1.5s, gen ~0.5s
//
// This class implements the App side of the "shell daemon" architecture:
//   1. App pushes lm_shell + liblitert-lm.so + dispatch so + model to
//      /data/local/tmp/npullm/ on first boot (world-readable tmp dir).
//   2. App starts a tiny `sh` loop that keeps a persistent lm_shell-all daemon?
//      NO — lm_shell is one-shot. Instead the App keeps its own HTTP server
//      and per-request execs lm_shell with a per-request session. Engine init
//      is ~1.5s on NPU, acceptable per request; sessions are stateless across
//      requests (each request = fresh engine + single-turn generate).
//   3. Once `adb shell` bootstraps the long-lived daemon (lm_daemon), the App
//      proxies HTTP -> local socket. See ShellDaemon for the protocol.
//
// Until the daemon bootstrap lands, ShellEngine.execOneShot() is the functional
// path: ~2s overhead per request (engine init), NPU-accelerated generate.
package com.npullm.server;

import android.util.Log;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

public class ShellEngine {
    private static final String TAG = "ShellEngine";

    private final File workDir;      // /data/local/tmp/npullm (world-accessible)
    private final File modelFile;    // workDir/mt6991.litertlm
    private final File dispatchDir;  // workDir/dispatch/
    private final File lmShell;      // workDir/lm_shell
    private final File libDir;       // workDir (liblitert-lm.so lives here)
    private volatile String lastError = "";

    public ShellEngine(File workDir, File modelFile, File dispatchDir, File lmShell) {
        this.workDir = workDir;
        this.modelFile = modelFile;
        this.dispatchDir = dispatchDir;
        this.lmShell = lmShell;
        this.libDir = workDir;
    }

    public String lastError() { return lastError; }

    public boolean usable() {
        return lmShell.exists() && lmShell.canExecute()
                && modelFile.exists() && dispatchDir.exists()
                && new File(libDir, "liblitert-lm.so").exists();
    }

    private static String readAll(InputStream in) throws Exception {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        byte[] buf = new byte[8192];
        int n;
        while ((n = in.read(buf)) > 0) bos.write(buf, 0, n);
        return new String(bos.toByteArray(), StandardCharsets.UTF_8);
    }

    /** One-shot NPU generate. Returns reply text, or null on failure (see lastError()). */
    public synchronized String generate(String prompt, int maxTokens) {
        lastError = "";
        try {
            List<String> cmd = new ArrayList<String>();
            cmd.add(lmShell.getAbsolutePath());
            cmd.add(modelFile.getAbsolutePath());
            cmd.add("npu");
            cmd.add(dispatchDir.getAbsolutePath());
            cmd.add(workDir.getAbsolutePath()); // cache dir
            cmd.add(String.valueOf(maxTokens > 0 ? maxTokens : 1280));
            cmd.add(prompt);
            ProcessBuilder pb = new ProcessBuilder(cmd);
            pb.directory(workDir);
            pb.environment().put("LD_LIBRARY_PATH", libDir.getAbsolutePath());
            pb.redirectErrorStream(false);
            Process p = pb.start();
            // feed nothing on stdin; close it so child never blocks on stdin
            try { p.getOutputStream().close(); } catch (Exception ignored) {}
            String stdout = readAll(p.getInputStream());
            String stderr = readAll(p.getErrorStream());
            int rc = p.waitFor();
            if (rc != 0) {
                lastError = "lm_shell rc=" + rc + " err=" + tail(stderr, 2000);
                Log.w(TAG, lastError);
                return null;
            }
            // stdout = reply text (+ trailing newline); stderr has [init/generate] timings
            Log.i(TAG, "shell timings: " + tail(stderr.replace("\n", " | "), 500));
            return stdout.trim();
        } catch (Exception e) {
            lastError = String.valueOf(e.getMessage());
            Log.w(TAG, "shell generate failed", e);
            return null;
        }
    }

    private static String tail(String s, int n) {
        if (s == null) return "";
        return s.length() <= n ? s : s.substring(s.length() - n);
    }
}
