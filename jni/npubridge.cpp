// NpuBridge: JNI glue between Java service and LiteRT-LM C API (liblitert-lm.so).
// Loaded as libnpubridge.so. All engine work happens on a single dedicated
// worker thread to keep the LiteRT-LM engine thread-safe; Java blocks on
// condition variables via polling (simple + robust, no callbacks across JNI).
#include <jni.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <android/log.h>

#define LOG_TAG "NpuBridge"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ---- LiteRT-LM C API forward declarations (subset we use) ----
typedef struct LiteRtLmEngine LiteRtLmEngine;
typedef struct LiteRtLmEngineSettings LiteRtLmEngineSettings;
typedef struct LiteRtLmSession LiteRtLmSession;
typedef struct LiteRtLmSessionConfig LiteRtLmSessionConfig;
typedef struct LiteRtLmInputData LiteRtLmInputData;
typedef struct LiteRtLmResponses LiteRtLmResponses;
typedef struct LiteRtLmConversation LiteRtLmConversation;
typedef struct LiteRtLmConversationConfig LiteRtLmConversationConfig;
typedef struct LiteRtLmConversationOptionalArgs LiteRtLmConversationOptionalArgs;

typedef enum {
  kLiteRtLmInputDataTypeText = 0,
} LiteRtLmInputDataType;

static LiteRtLmEngineSettings* (*p_settings_create)(const char*, const char*, const char*, const char*);
static void (*p_settings_delete)(LiteRtLmEngineSettings*);
static void (*p_settings_set_cache_dir)(LiteRtLmEngineSettings*, const char*);
static void (*p_settings_set_dispatch_dir)(LiteRtLmEngineSettings*, const char*);
static void (*p_settings_set_max_tokens)(LiteRtLmEngineSettings*, int);
static LiteRtLmEngine* (*p_engine_create)(const LiteRtLmEngineSettings*);
static void (*p_engine_delete)(LiteRtLmEngine*);
static LiteRtLmSession* (*p_engine_create_session)(LiteRtLmEngine*, LiteRtLmSessionConfig*);
static void (*p_session_delete)(LiteRtLmSession*);
static LiteRtLmSessionConfig* (*p_session_config_create)(void);
static void (*p_session_config_delete)(LiteRtLmSessionConfig*);
static int (*p_session_config_set_max_output)(LiteRtLmSessionConfig*, int);
static LiteRtLmInputData* (*p_input_create)(LiteRtLmInputDataType, const void*, size_t);
static void (*p_input_delete)(LiteRtLmInputData*);
static LiteRtLmResponses* (*p_session_generate)(LiteRtLmSession*, const LiteRtLmInputData* const*, size_t);
static void (*p_responses_delete)(LiteRtLmResponses*);
static int (*p_responses_num)(const LiteRtLmResponses*);
static const char* (*p_responses_text_at)(const LiteRtLmResponses*, int);
static void (*p_session_cancel)(LiteRtLmSession*);

static void* g_lm_handle = NULL;

#define LOAD_SYM(var, name) do { \
  *(void**)(&p_##var) = dlsym(g_lm_handle, name); \
  if (!p_##var) { LOGE("missing symbol %s", name); return 0; } \
} while (0)

static int load_lm_lib(void) {
  if (g_lm_handle) return 1;
  g_lm_handle = dlopen("liblitert-lm.so", RTLD_NOW | RTLD_GLOBAL);
  if (!g_lm_handle) { LOGE("dlopen liblitert-lm.so failed: %s", dlerror()); return 0; }
  LOAD_SYM(settings_create, "litert_lm_engine_settings_create");
  LOAD_SYM(settings_delete, "litert_lm_engine_settings_delete");
  LOAD_SYM(settings_set_cache_dir, "litert_lm_engine_settings_set_cache_dir");
  LOAD_SYM(settings_set_dispatch_dir, "litert_lm_engine_settings_set_litert_dispatch_lib_dir");
  LOAD_SYM(settings_set_max_tokens, "litert_lm_engine_settings_set_max_num_tokens");
  LOAD_SYM(engine_create, "litert_lm_engine_create");
  LOAD_SYM(engine_delete, "litert_lm_engine_delete");
  LOAD_SYM(engine_create_session, "litert_lm_engine_create_session");
  LOAD_SYM(session_delete, "litert_lm_session_delete");
  LOAD_SYM(session_config_create, "litert_lm_session_config_create");
  LOAD_SYM(session_config_delete, "litert_lm_session_config_delete");
  LOAD_SYM(input_create, "litert_lm_input_data_create");
  LOAD_SYM(input_delete, "litert_lm_input_data_delete");
  LOAD_SYM(session_generate, "litert_lm_session_generate_content");
  LOAD_SYM(responses_delete, "litert_lm_responses_delete");
  LOAD_SYM(responses_num, "litert_lm_responses_get_num_candidates");
  LOAD_SYM(responses_text_at, "litert_lm_responses_get_response_text_at");
  // optional symbols (may be absent in some builds)
  *(void**)(&p_session_cancel) = dlsym(g_lm_handle, "litert_lm_session_cancel_process");
  *(void**)(&p_session_config_set_max_output) = dlsym(g_lm_handle, "litert_lm_session_config_set_max_output_tokens");
  LOGI("liblitert-lm.so symbols loaded");
  return 1;
}

// ---- Worker state ----
static pthread_t g_worker;
static int g_worker_started = 0;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_cond = PTHREAD_COND_INITIALIZER;

static LiteRtLmEngine* g_engine = NULL;
static LiteRtLmSession* g_session = NULL;
static char g_model_path[1024];
static char g_cache_dir[1024];
static char g_dispatch_dir[1024];
static char g_backend[32];
static int g_max_tokens = 1280;

// request slots
static int g_cmd = 0; // 0 idle, 1 init, 2 generate, 3 destroy
static int g_done = 0;
static int g_status = -1;
static char* g_prompt = NULL;   // malloc'd, worker frees
static char* g_result = NULL;   // malloc'd, java copies then frees
static char g_error[1024];

static void set_error(const char* msg) {
  strncpy(g_error, msg, sizeof(g_error) - 1);
  g_error[sizeof(g_error) - 1] = 0;
}

static void do_init(void) {
  if (g_engine) { g_status = 0; return; }
  if (!load_lm_lib()) { set_error("dlopen liblitert-lm.so failed"); g_status = -1; return; }
  LiteRtLmEngineSettings* s = p_settings_create(g_model_path, g_backend, NULL, NULL);
  if (!s) { set_error("settings_create failed"); g_status = -1; return; }
  if (g_cache_dir[0]) p_settings_set_cache_dir(s, g_cache_dir);
  if (g_dispatch_dir[0]) p_settings_set_dispatch_dir(s, g_dispatch_dir);
  p_settings_set_max_tokens(s, g_max_tokens);
  g_engine = p_engine_create(s);
  p_settings_delete(s);
  if (!g_engine) { set_error("engine_create failed (check model/backend/dispatch so)"); g_status = -1; return; }
  LiteRtLmSessionConfig* sc = p_session_config_create();
  if (sc && p_session_config_set_max_output) p_session_config_set_max_output(sc, 1024);
  g_session = p_engine_create_session(g_engine, sc);
  if (sc) p_session_config_delete(sc);
  if (!g_session) { p_engine_delete(g_engine); g_engine = NULL; set_error("create_session failed"); g_status = -1; return; }
  LOGI("engine init ok backend=%s", g_backend);
  g_status = 0;
}

static void do_generate(void) {
  if (!g_session) { set_error("engine not initialized"); g_status = -1; return; }
  LiteRtLmInputData* in = p_input_create(kLiteRtLmInputDataTypeText, g_prompt, strlen(g_prompt));
  if (!in) { set_error("input_create failed"); g_status = -1; return; }
  const LiteRtLmInputData* arr[1] = { in };
  LiteRtLmResponses* r = p_session_generate(g_session, arr, 1);
  p_input_delete(in);
  if (!r) { set_error("generate failed"); g_status = -1; return; }
  int n = p_responses_num(r);
  const char* t = (n > 0) ? p_responses_text_at(r, 0) : "";
  free(g_result); g_result = NULL;
  g_result = strdup(t ? t : "");
  p_responses_delete(r);
  g_status = 0;
}

static void do_destroy(void) {
  if (g_session) { p_session_delete(g_session); g_session = NULL; }
  if (g_engine) { p_engine_delete(g_engine); g_engine = NULL; }
  g_status = 0;
}

static void* worker_main(void* arg) {
  (void)arg;
  for (;;) {
    pthread_mutex_lock(&g_mu);
    while (g_cmd == 0) pthread_cond_wait(&g_cond, &g_mu);
    int cmd = g_cmd;
    pthread_mutex_unlock(&g_mu);
    if (cmd == 1) do_init();
    else if (cmd == 2) do_generate();
    else if (cmd == 3) do_destroy();
    pthread_mutex_lock(&g_mu);
    g_cmd = 0; g_done = 1;
    pthread_cond_signal(&g_cond);
    pthread_mutex_unlock(&g_mu);
    if (cmd == 3) break;
  }
  return NULL;
}

static void ensure_worker(void) {
  if (g_worker_started) return;
  pthread_create(&g_worker, NULL, worker_main, NULL);
  g_worker_started = 1;
}

// run cmd on worker and wait
static int run_cmd(int cmd) {
  ensure_worker();
  pthread_mutex_lock(&g_mu);
  g_cmd = cmd; g_done = 0; g_status = -1;
  pthread_cond_signal(&g_cond);
  while (!g_done) pthread_cond_wait(&g_cond, &g_mu);
  int st = g_status;
  pthread_mutex_unlock(&g_mu);
  return st;
}

extern "C" {

JNIEXPORT jint JNICALL
Java_com_npullm_server_NpuEngine_nativeInit(JNIEnv* env, jobject thiz,
    jstring jModel, jstring jBackend, jstring jCache, jstring jDispatch, jint maxTokens) {
  const char* m = env->GetStringUTFChars(jModel, 0);
  const char* b = env->GetStringUTFChars(jBackend, 0);
  const char* c = jCache ? env->GetStringUTFChars(jCache, 0) : NULL;
  const char* d = jDispatch ? env->GetStringUTFChars(jDispatch, 0) : NULL;
  strncpy(g_model_path, m, sizeof(g_model_path) - 1);
  strncpy(g_backend, b ? b : "npu", sizeof(g_backend) - 1);
  if (c) strncpy(g_cache_dir, c, sizeof(g_cache_dir) - 1); else g_cache_dir[0] = 0;
  if (d) strncpy(g_dispatch_dir, d, sizeof(g_dispatch_dir) - 1); else g_dispatch_dir[0] = 0;
  g_max_tokens = maxTokens > 0 ? maxTokens : 1280;
  env->ReleaseStringUTFChars(jModel, m);
  if (b) env->ReleaseStringUTFChars(jBackend, b);
  if (c) env->ReleaseStringUTFChars(jCache, c);
  if (d) env->ReleaseStringUTFChars(jDispatch, d);
  int st = run_cmd(1);
  if (st != 0) {
    jclass ex = env->FindClass("java/lang/RuntimeException");
    if (ex) env->ThrowNew(ex, g_error[0] ? g_error : "init failed");
  }
  return st;
}

JNIEXPORT jstring JNICALL
Java_com_npullm_server_NpuEngine_nativeGenerate(JNIEnv* env, jobject thiz, jstring jPrompt) {
  const char* p = env->GetStringUTFChars(jPrompt, 0);
  pthread_mutex_lock(&g_mu);
  free(g_prompt); g_prompt = strdup(p);
  pthread_mutex_unlock(&g_mu);
  env->ReleaseStringUTFChars(jPrompt, p);
  int st = run_cmd(2);
  if (st != 0) {
    jclass ex = env->FindClass("java/lang/RuntimeException");
    if (ex) env->ThrowNew(ex, g_error[0] ? g_error : "generate failed");
    return NULL;
  }
  pthread_mutex_lock(&g_mu);
  jstring out = env->NewStringUTF(g_result ? g_result : "");
  pthread_mutex_unlock(&g_mu);
  return out;
}

JNIEXPORT void JNICALL
Java_com_npullm_server_NpuEngine_nativeClose(JNIEnv* env, jobject thiz) {
  if (g_worker_started) run_cmd(3);
  free(g_result); g_result = NULL;
  free(g_prompt); g_prompt = NULL;
}

JNIEXPORT jstring JNICALL
Java_com_npullm_server_NpuEngine_nativeLastError(JNIEnv* env, jobject thiz) {
  return env->NewStringUTF(g_error);
}

} // extern "C"
