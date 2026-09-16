package com.npullm.server;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Intent;
import android.os.Build;
import android.os.IBinder;
import android.util.Log;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.ByteArrayOutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicReference;

public class LlmService extends Service {
    private static final String TAG = "NpuLlmSvc";
    public static final int PORT = 18080;
    static final AtomicReference<String> STATUS = new AtomicReference<String>("starting");
    static final AtomicReference<String> LAST_ERROR = new AtomicReference<String>("");

    private NpuEngine engine;
    private ServerSocket server;
    private ExecutorService pool = Executors.newCachedThreadPool();
    private volatile boolean running;
    private volatile String backend = "npu";
    private Thread acceptThread;

    @Override
    public void onCreate() {
        super.onCreate();
        startFg();
        running = true;
        acceptThread = new Thread(new Runnable() {
            @Override public void run() { bootAndServe(); }
        });
        acceptThread.setDaemon(true);
        acceptThread.start();
    }

    private void startFg() {
        String ch = "npu_llm";
        NotificationManager nm = (NotificationManager) getSystemService(NOTIFICATION_SERVICE);
        if (Build.VERSION.SDK_INT >= 26 && nm.getNotificationChannel(ch) == null) {
            nm.createNotificationChannel(new NotificationChannel(ch, "NPU LLM", NotificationManager.IMPORTANCE_LOW));
        }
        Notification.Builder b = Build.VERSION.SDK_INT >= 26
                ? new Notification.Builder(this, ch) : new Notification.Builder(this);
        b.setContentTitle("NPU LLM Server").setContentText("port 18080")
                .setSmallIcon(android.R.drawable.stat_sys_data_bluetooth);
        startForeground(1, b.build());
    }

    private volatile boolean engineReady = false;

    private void bootAndServe() {
        // Bind the HTTP port FIRST so /health is always reachable, then
        // init the (slow, fallible) engine on a worker thread.
        try {
            server = new ServerSocket(PORT);
            Log.i(TAG, "listening on " + PORT);
        } catch (Exception e) {
            Log.e(TAG, "bind failed", e);
            STATUS.set("error: bind " + e.getMessage());
            return;
        }
        Thread init = new Thread(new Runnable() {
            @Override public void run() { initEngine(); }
        });
        init.setDaemon(true);
        init.start();
        try {
            while (running) {
                final Socket s = server.accept();
                pool.execute(new Runnable() {
                    @Override public void run() { handle(s); }
                });
            }
        } catch (Exception e) {
            if (running) { Log.e(TAG, "serve failed", e); STATUS.set("error: " + e.getMessage()); }
        }
    }

    private void initEngine() {
        try {
            STATUS.set("loading model...");
            engine = new NpuEngine();
            File files = getFilesDir();
            File model = new File(files, "model.litertlm");
            STATUS.set("copying model...");
            copyAsset("model.litertlm", model);
            File dispatchDir = new File(files, "dispatch");
            dispatchDir.mkdirs();
            copyAssetTo("libLiteRtDispatch_MediaTek.so", new File(dispatchDir, "libLiteRtDispatch_MediaTek.so"));
            File cacheDir = new File(getCacheDir(), "litertlm");
            cacheDir.mkdirs();
            // NPU first, fall back to CPU so the HTTP API is always usable.
            // NOTE: each nativeInit runs on the single native worker; a failed
            // NPU init leaves no engine behind, so retrying with CPU is safe.
            // A model compiled for a different SoC (e.g. mt6993 on mt6991)
            // fails NPU init -> falls back to CPU only if the model has no
            // NPU backend constraint. NPU-constrained models fail both.
            boolean npuOk = false;
            try {
                engine.nativeInit(model.getAbsolutePath(), "npu",
                        cacheDir.getAbsolutePath(), dispatchDir.getAbsolutePath(), 1280);
                backend = "npu";
                npuOk = true;
            } catch (RuntimeException e) {
                Log.w(TAG, "npu init failed, fallback cpu: " + e.getMessage());
                LAST_ERROR.set("npu init failed: " + e.getMessage());
            }
            if (!npuOk) {
                engine.nativeInit(model.getAbsolutePath(), "cpu",
                        cacheDir.getAbsolutePath(), "", 1280);
                backend = "cpu";
            }
            STATUS.set("ready (" + backend + ")");
            engineReady = true;
            Log.i(TAG, "engine ready backend=" + backend);
        } catch (Exception e) {
            Log.e(TAG, "boot failed", e);
            LAST_ERROR.set(String.valueOf(e.getMessage()));
            STATUS.set("error: " + e.getMessage());
            return;
        }
    }

    private void copyAsset(String name, File dst) throws Exception {
        copyAssetTo(name, dst);
    }

    private void copyAssetTo(String name, File dst) throws Exception {
        // Re-copy when the bundled asset differs (model updates must propagate;
        // the old "exists => skip" check pinned a stale 1.1G mt6993 model).
        long assetLen = -1;
        try {
            android.content.res.AssetFileDescriptor afd = getAssets().openFd(name);
            assetLen = afd.getLength();
            afd.close();
        } catch (Exception ignored) {}
        if (dst.exists() && assetLen > 0 && dst.length() == assetLen) return;
        if (dst.exists()) dst.delete();
        STATUS.set("copying " + name + "...");
        InputStream in = getAssets().open(name);
        OutputStream out = new FileOutputStream(dst);
        byte[] buf = new byte[65536];
        int n;
        while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
        out.close(); in.close();
    }

    private static String readAll(InputStream in) throws Exception {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        byte[] buf = new byte[8192];
        int n;
        while ((n = in.read(buf)) > 0) bos.write(buf, 0, n);
        return new String(bos.toByteArray(), StandardCharsets.UTF_8);
    }

    private void handle(Socket s) {
        try {
            s.setSoTimeout(600000);
            InputStream in = s.getInputStream();
            OutputStream out = s.getOutputStream();
            // read headers
            ByteArrayOutputStream head = new ByteArrayOutputStream();
            int b, last4 = 0;
            while ((b = in.read()) != -1) {
                head.write(b);
                last4 = ((last4 << 8) | (b & 0xff)) & 0xffffffff;
                if (head.size() > 65536) break;
                if (head.size() >= 4 && last4 == 0x0d0a0d0a) break;
            }
            String hs = new String(head.toByteArray(), StandardCharsets.UTF_8);
            int eol = hs.indexOf("\r\n");
            if (eol < 0) { s.close(); return; }
            String[] req = hs.substring(0, eol).split(" ");
            if (req.length < 2) { s.close(); return; }
            String method = req[0], path = req[1];
            Map<String, String> headers = new HashMap<String, String>();
            for (String line : hs.substring(eol + 2).split("\r\n")) {
                int c = line.indexOf(':');
                if (c > 0) headers.put(line.substring(0, c).trim().toLowerCase(),
                        line.substring(c + 1).trim());
            }
            int len = 0;
            try { len = Integer.parseInt(headers.get("content-length")); } catch (Exception e) { len = 0; }
            byte[] body = new byte[len];
            int off = 0;
            while (off < len) { int r = in.read(body, off, len - off); if (r < 0) break; off += r; }
            String bodyStr = new String(body, 0, off, StandardCharsets.UTF_8);

            if (method.equals("GET") && path.equals("/v1/models")) {
                send(out, 200, "{\"object\":\"list\",\"data\":[{\"id\":\"npu-llm\",\"object\":\"model\",\"owned_by\":\"mt6991-npu\"}]}");
            } else if (method.equals("GET") && (path.equals("/") || path.equals("/health"))) {
                send(out, 200, "{\"status\":\"" + esc(STATUS.get()) + "\",\"backend\":\"" + esc(backend)
                        + "\",\"error\":\"" + esc(LAST_ERROR.get()) + "\"}");
            } else if (method.equals("POST") && path.equals("/v1/chat/completions")) {
                if (!engineReady) { send(out, 503, "{\"error\":\"engine not ready: " + esc(STATUS.get()) + "\"}"); s.close(); return; }
                String prompt = extractPrompt(bodyStr);
                boolean stream = bodyStr.contains("\"stream\"") && bodyStr.contains("true");
                String reply;
                try {
                    reply = engine.nativeGenerate(prompt);
                    if (reply == null) reply = "";
                } catch (RuntimeException e) {
                    send(out, 500, "{\"error\":\"" + esc(String.valueOf(e.getMessage())) + "\"}");
                    s.close(); return;
                }
                if (stream) {
                    String chunk = "{\"id\":\"chatcmpl-npu\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\""
                            + esc(reply) + "\"},\"index\":0,\"finish_reason\":null}]}\n";
                    String payload = "data: " + chunk + "\ndata: [DONE]\n";
                    String h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: "
                            + payload.getBytes(StandardCharsets.UTF_8).length + "\r\nConnection: close\r\n\r\n";
                    out.write(h.getBytes(StandardCharsets.UTF_8));
                    out.write(payload.getBytes(StandardCharsets.UTF_8));
                } else {
                    String payload = "{\"id\":\"chatcmpl-npu\",\"object\":\"chat.completion\",\"model\":\"npu-llm\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\""
                            + esc(reply) + "\"},\"finish_reason\":\"stop\"}]}";
                    send(out, 200, payload);
                }
            } else if (method.equals("POST") && path.equals("/v1/completions")) {
                if (!engineReady) { send(out, 503, "{\"error\":\"engine not ready: " + esc(STATUS.get()) + "\"}"); s.close(); return; }
                String prompt = extractPrompt(bodyStr);
                String reply;
                try { reply = engine.nativeGenerate(prompt); if (reply == null) reply = ""; }
                catch (RuntimeException e) { send(out, 500, "{\"error\":\"" + esc(String.valueOf(e.getMessage())) + "\"}"); s.close(); return; }
                send(out, 200, "{\"id\":\"cmpl-npu\",\"object\":\"text_completion\",\"model\":\"npu-llm\",\"choices\":[{\"text\":\""
                        + esc(reply) + "\",\"index\":0,\"finish_reason\":\"stop\"}]}");
            } else {
                send(out, 404, "{\"error\":\"not found: " + esc(path) + "\"}");
            }
            out.flush(); s.close();
        } catch (Exception e) {
            try { s.close(); } catch (Exception ignored) {}
        }
    }

    static String extractPrompt(String json) {
        // Prefer OpenAI messages[-1].content (string or content-parts array).
        int mi = json.lastIndexOf("\"messages\"");
        String scope = mi >= 0 ? json.substring(mi) : json;
        // content parts: {"type":"text","text":"..."}
        int ti = scope.lastIndexOf("\"text\"");
        if (ti >= 0) {
            int c = scope.indexOf(':', ti);
            String v = readJsonString(scope, c + 1);
            if (v != null) return v;
        }
        // plain "content": "..." or "prompt": "..."
        int ci = scope.lastIndexOf("\"content\"");
        if (ci < 0) ci = json.lastIndexOf("\"prompt\"");
        if (ci >= 0) {
            int c = json.indexOf(':', ci);
            String v = readJsonString(json, c + 1);
            if (v != null) return v;
        }
        return json.length() > 4000 ? json.substring(0, 4000) : json;
    }

    static String readJsonString(String s, int i) {
        while (i < s.length() && Character.isWhitespace(s.charAt(i))) i++;
        if (i >= s.length() || s.charAt(i) != '"') return null;
        StringBuilder sb = new StringBuilder();
        i++;
        while (i < s.length()) {
            char c = s.charAt(i);
            if (c == '\\' && i + 1 < s.length()) {
                char n = s.charAt(i + 1);
                if (n == 'n') sb.append('\n');
                else if (n == 't') sb.append('\t');
                else if (n == 'r') sb.append('\r');
                else if (n == 'u' && i + 5 < s.length()) {
                    try { sb.append((char) Integer.parseInt(s.substring(i + 2, i + 6), 16)); i += 6; continue; }
                    catch (Exception e) { sb.append(n); }
                } else sb.append(n);
                i += 2;
            } else if (c == '"') break;
            else { sb.append(c); i++; }
        }
        return sb.toString();
    }

    static String esc(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\").replace("\"", "\\\"")
                .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t");
    }

    private static void send(OutputStream out, int code, String payload) throws Exception {
        byte[] b = payload.getBytes(StandardCharsets.UTF_8);
        String h = "HTTP/1.1 " + code + (code == 200 ? " OK" : " ERR")
                + "\r\nContent-Type: application/json\r\nContent-Length: " + b.length
                + "\r\nConnection: close\r\n\r\n";
        out.write(h.getBytes(StandardCharsets.UTF_8));
        out.write(b);
    }

    @Override public IBinder onBind(Intent i) { return null; }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        return START_STICKY;
    }

    @Override
    public void onDestroy() {
        running = false;
        try { if (server != null) server.close(); } catch (Exception ignored) {}
        pool.shutdownNow();
        try { if (engine != null) engine.nativeClose(); } catch (Exception ignored) {}
        super.onDestroy();
    }
}
