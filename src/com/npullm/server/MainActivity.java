package com.npullm.server;

import android.app.Activity;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.Gravity;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

public class MainActivity extends Activity {
    private TextView status;
    private Handler h = new Handler(Looper.getMainLooper());
    private Runnable poller;

    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(32, 32, 32, 32);
        root.setGravity(Gravity.CENTER_HORIZONTAL);

        TextView title = new TextView(this);
        title.setText("NPU LLM Server (MT6991)");
        title.setTextSize(20);
        root.addView(title);

        status = new TextView(this);
        status.setText("starting...");
        status.setTextSize(14);
        root.addView(status);

        TextView hint = new TextView(this);
        hint.setText("API: http://127.0.0.1:18080\nGET  /health, /v1/models\nPOST /v1/chat/completions (OpenAI)\nPC: adb reverse tcp:18080 tcp:18080");
        hint.setTextSize(13);
        root.addView(hint);

        Button start = new Button(this);
        start.setText("Start service");
        start.setOnClickListener(new android.view.View.OnClickListener() {
            @Override public void onClick(android.view.View v) { startSvc(); }
        });
        root.addView(start);

        Button stop = new Button(this);
        stop.setText("Stop service");
        stop.setOnClickListener(new android.view.View.OnClickListener() {
            @Override public void onClick(android.view.View v) { stopService(new Intent(MainActivity.this, LlmService.class)); }
        });
        root.addView(stop);

        ScrollView sv = new ScrollView(this);
        sv.addView(root);
        setContentView(sv);
        startSvc();

        poller = new Runnable() {
            @Override public void run() {
                status.setText("status: " + LlmService.STATUS.get()
                        + "\nerr: " + LlmService.LAST_ERROR.get());
                h.postDelayed(this, 1000);
            }
        };
        h.post(poller);
    }

    private void startSvc() {
        Intent i = new Intent(this, LlmService.class);
        if (Build.VERSION.SDK_INT >= 26) startForegroundService(i);
        else startService(i);
    }

    @Override
    protected void onDestroy() {
        h.removeCallbacks(poller);
        super.onDestroy();
    }
}
