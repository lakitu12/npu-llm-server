// lm_shell: minimal shell CLI over the LiteRT-LM C API (direct link, no JNI).
// Usage: lm_shell <model> <backend> <dispatch_dir|-> <cache_dir|-> <max_tokens> "<prompt>"
// Links against liblitert-lm.so (android_arm64). Run on device as shell:
//   LD_LIBRARY_PATH=. ./lm_shell model.litertlm npu ./dispatch ./cache 1280 "你好"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct LiteRtLmEngine LiteRtLmEngine;
typedef struct LiteRtLmEngineSettings LiteRtLmEngineSettings;
typedef struct LiteRtLmSession LiteRtLmSession;
typedef struct LiteRtLmSessionConfig LiteRtLmSessionConfig;
typedef struct LiteRtLmInputData LiteRtLmInputData;
typedef struct LiteRtLmResponses LiteRtLmResponses;
typedef enum { kLiteRtLmInputDataTypeText = 0 } LiteRtLmInputDataType;

extern "C" {
LiteRtLmEngineSettings* litert_lm_engine_settings_create(const char*, const char*, const char*, const char*);
void litert_lm_engine_settings_delete(LiteRtLmEngineSettings*);
void litert_lm_engine_settings_set_cache_dir(LiteRtLmEngineSettings*, const char*);
void litert_lm_engine_settings_set_litert_dispatch_lib_dir(LiteRtLmEngineSettings*, const char*);
void litert_lm_engine_settings_set_max_num_tokens(LiteRtLmEngineSettings*, int);
LiteRtLmEngine* litert_lm_engine_create(const LiteRtLmEngineSettings*);
void litert_lm_engine_delete(LiteRtLmEngine*);
LiteRtLmSession* litert_lm_engine_create_session(LiteRtLmEngine*, LiteRtLmSessionConfig*);
void litert_lm_session_delete(LiteRtLmSession*);
LiteRtLmSessionConfig* litert_lm_session_config_create(void);
void litert_lm_session_config_delete(LiteRtLmSessionConfig*);
void litert_lm_session_config_set_max_output_tokens(LiteRtLmSessionConfig*, int);
LiteRtLmInputData* litert_lm_input_data_create(LiteRtLmInputDataType, const void*, size_t);
void litert_lm_input_data_delete(LiteRtLmInputData*);
LiteRtLmResponses* litert_lm_session_generate_content(LiteRtLmSession*, const LiteRtLmInputData* const*, size_t);
void litert_lm_responses_delete(LiteRtLmResponses*);
int litert_lm_responses_get_num_candidates(const LiteRtLmResponses*);
const char* litert_lm_responses_get_response_text_at(const LiteRtLmResponses*, int);
}

static double elapsed(const struct timespec& a, const struct timespec& b) {
  return (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
}

int main(int argc, char** argv) {
  if (argc < 7) {
    fprintf(stderr, "usage: %s <model> <backend> <dispatch_dir|-> <cache_dir|-> <max_tokens> \"<prompt>\"\n", argv[0]);
    return 2;
  }
  const char* model = argv[1];
  const char* backend = argv[2];
  const char* dispatch = strcmp(argv[3], "-") != 0 ? argv[3] : NULL;
  const char* cache = strcmp(argv[4], "-") != 0 ? argv[4] : NULL;
  int max_tok = atoi(argv[5]);
  const char* prompt = argv[6];

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  LiteRtLmEngineSettings* s = litert_lm_engine_settings_create(model, backend, NULL, NULL);
  if (!s) { fprintf(stderr, "settings_create failed\n"); return 1; }
  if (cache) litert_lm_engine_settings_set_cache_dir(s, cache);
  if (dispatch) litert_lm_engine_settings_set_litert_dispatch_lib_dir(s, dispatch);
  litert_lm_engine_settings_set_max_num_tokens(s, max_tok > 0 ? max_tok : 1280);
  LiteRtLmEngine* e = litert_lm_engine_create(s);
  litert_lm_engine_settings_delete(s);
  if (!e) { fprintf(stderr, "engine_create FAILED\n"); return 1; }
  LiteRtLmSessionConfig* sc = litert_lm_session_config_create();
  if (sc) litert_lm_session_config_set_max_output_tokens(sc, 512);
  LiteRtLmSession* sess = litert_lm_engine_create_session(e, sc);
  if (sc) litert_lm_session_config_delete(sc);
  if (!sess) { fprintf(stderr, "create_session FAILED\n"); litert_lm_engine_delete(e); return 1; }
  clock_gettime(CLOCK_MONOTONIC, &t1);
  fprintf(stderr, "[init ok backend=%s %.1fs]\n", backend, elapsed(t0, t1));

  LiteRtLmInputData* in = litert_lm_input_data_create(kLiteRtLmInputDataTypeText, prompt, strlen(prompt));
  if (!in) { fprintf(stderr, "input_create FAILED\n"); return 1; }
  const LiteRtLmInputData* arr[1] = { in };
  clock_gettime(CLOCK_MONOTONIC, &t0);
  LiteRtLmResponses* r = litert_lm_session_generate_content(sess, arr, 1);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  litert_lm_input_data_delete(in);
  if (!r) { fprintf(stderr, "generate FAILED\n"); return 1; }
  int n = litert_lm_responses_get_num_candidates(r);
  const char* t = n > 0 ? litert_lm_responses_get_response_text_at(r, 0) : "";
  printf("%s\n", t ? t : "");
  fprintf(stderr, "[generate %.1fs candidates=%d]\n", elapsed(t0, t1), n);
  litert_lm_responses_delete(r);
  litert_lm_session_delete(sess);
  litert_lm_engine_delete(e);
  return 0;
}
