#include "ggml-backend.h"
#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using nlohmann::json;
using vectors = std::map<std::string, std::vector<float>>;

struct capture {
    vectors states, writes;
    bool keys = false;
    bool valid = true;
};

static bool is_key_site(const std::string &name) {
    const std::string prefix = "dory_key-";
    return name.size() > prefix.size() && name.compare(0, prefix.size(), prefix) == 0 &&
           std::all_of(name.begin() + prefix.size(), name.end(), [](char c) { return c >= '0' && c <= '9'; });
}

static bool read_state(ggml_tensor *tensor, bool ask, void *opaque) {
    auto &run = *static_cast<capture *>(opaque);
    const std::string name = ggml_get_name(tensor);
    const bool key = is_key_site(name);
    const bool wanted = run.states.count(name) || (run.keys && key);
    if (ask) {
        return wanted;
    }
    if (!wanted) {
        return true;
    }
    const auto groups = key ? tensor->ne[1] : 1;
    const auto tokens = key ? tensor->ne[2] : tensor->ne[1];
    const auto stride = key ? tensor->nb[2] : tensor->nb[1];
    const auto width = tensor->ne[0];
    auto patch = run.writes.find(name);
    if (tensor->type != GGML_TYPE_F32 || tensor->nb[0] != sizeof(float) ||
        tokens < 1 || tensor->ne[3] != 1 ||
        (!key && (tensor->ne[2] != 1 || width != 2560)) ||
        (key && (width != 256 || groups != 4)) ||
        (patch != run.writes.end() && patch->second.size() != size_t(width * groups))) {
        std::cerr << "Unexpected " << name << " type " << ggml_type_name(tensor->type)
                  << " shape " << tensor->ne[0] << 'x' << tensor->ne[1] << 'x'
                  << tensor->ne[2] << 'x' << tensor->ne[3] << " stride " << tensor->nb[0] << '\n';
        run.valid = false;
        return false;
    }
    auto &values = run.states[name];
    values.resize(width * groups);
    for (int64_t group = 0; group < groups; ++group) {
        const auto offset = (tokens - 1) * stride + group * tensor->nb[1];
        if (patch != run.writes.end()) {
            ggml_backend_tensor_set(tensor, patch->second.data() + group * width, offset, width * sizeof(float));
        }
        ggml_backend_tensor_get(tensor, values.data() + group * width, offset, width * sizeof(float));
    }
    return true;
}

static std::vector<llama_token> tokenize(const llama_vocab *vocab, const std::string &prompt) {
    std::vector<llama_token> tokens(513);
    const int count = llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()),
                                     tokens.data(), static_cast<int32_t>(tokens.size()), false, true);
    if (count < 1) {
        throw std::runtime_error("Tokenization outside available context");
    }
    tokens.resize(count);
    return tokens;
}

static std::string head_fingerprint(const float *logits, int count) {
    uint64_t hash = 14695981039346656037ull;
    for (int i = 0; i < count; ++i) {
        uint32_t bits;
        static_assert(sizeof(bits) == sizeof(float));
        std::memcpy(&bits, logits + i, sizeof(bits));
        for (int shift = 0; shift < 32; shift += 8) {
            hash ^= (bits >> shift) & 255u;
            hash *= 1099511628211ull;
        }
    }
    return std::to_string(hash);
}

int main(int argc, char **argv) {
    try {
        if (argc < 5 || argc > 6) {
            throw std::runtime_error("Usage: dory_readout MODEL LOOPS GPU_LAYERS PROMPTS [BATCH]");
        }
        const int loops = std::stoi(argv[2]);
        const int batch = argc == 6 ? std::stoi(argv[5]) : 512;
        if (loops < 1 || loops > 4 || batch < 32 || batch > 512) {
            throw std::runtime_error("Invalid loop or batch count");
        }
        std::ifstream source(argv[4]);
        if (!source) {
            throw std::runtime_error("Cannot read rendered prompts");
        }
        ggml_backend_load_all();
        llama_backend_init();
        llama_model_kv_override overrides[2]{};
        const std::string loop_key = "dory.num_loops";
        loop_key.copy(overrides[0].key, loop_key.size());
        overrides[0].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        overrides[0].val_i64 = loops;
        auto model_params = llama_model_default_params();
        model_params.kv_overrides = overrides;
        model_params.n_gpu_layers = std::stoi(argv[3]);
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], model_params), llama_model_free
        );
        if (!model) {
            throw std::runtime_error("Dory model load failed");
        }
        const llama_vocab *vocab = llama_model_get_vocab(model.get());
        capture readout;
        auto params = llama_context_default_params();
        params.n_ctx = 512;
        params.n_batch = batch;
        params.n_ubatch = batch;
        params.n_threads = 1;
        params.n_threads_batch = 1;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        params.cb_eval = read_state;
        params.cb_eval_user_data = &readout;
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_init_from_model(model.get(), params), llama_free
        );
        if (!context) {
            throw std::runtime_error("Dory context setup failed");
        }
        for (std::string line; std::getline(source, line);) {
            const json row = json::parse(line);
            const std::string prompt = row.at("rendered");
            auto tokens = tokenize(vocab, prompt);
            if (tokens.size() > size_t(batch)) {
                throw std::runtime_error("Prompt exceeds one prefill batch");
            }
            readout = {};
            readout.keys = row.value("keys", false);
            readout.writes = row.value("writes", vectors{});
            for (int loop = 1; loop <= loops; ++loop) {
                readout.states["l_out-" + std::to_string(16 + 40 * loop - 1)] = {};
            }
            for (const auto &[name, values] : readout.writes) {
                if ((!readout.states.count(name) && !(readout.keys && is_key_site(name))) ||
                    !std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); })) {
                    throw std::runtime_error("Invalid intervention tensor " + name);
                }
            }
            llama_memory_clear(llama_get_memory(context.get()), true);
            const auto start = std::chrono::steady_clock::now();
            if (llama_decode(context.get(), llama_batch_get_one(tokens.data(), tokens.size())) != 0 ||
                !readout.valid) {
                throw std::runtime_error("Dory prompt decode failed");
            }
            const float *logits = llama_get_logits_ith(context.get(), -1);
            const int vocabulary_size = llama_vocab_n_tokens(vocab);
            if (!logits || !std::all_of(logits, logits + vocabulary_size, [](float x) { return std::isfinite(x); })) {
                throw std::runtime_error("Missing or non-finite native logits");
            }
            const llama_token best = std::max_element(logits, logits + vocabulary_size) - logits;
            char piece[256];
            const int bytes = llama_token_to_piece(vocab, best, piece, sizeof(piece), 0, true);
            if (bytes < 0) {
                throw std::runtime_error("Native token piece exceeds output buffer");
            }
            for (const auto &[name, values] : readout.states) {
                if (values.empty() || !std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); })) {
                    throw std::runtime_error("Missing or non-finite tensor " + name);
                }
            }
            for (const auto &[name, values] : readout.writes) {
                if (readout.states.at(name) != values) {
                    throw std::runtime_error("Intervention tensor did not restore exact requested F32 values: " + name);
                }
            }
            json options = json::object();
            for (const std::string option : row.at("options").get<std::vector<std::string>>()) {
                const auto ids = tokenize(vocab, prompt + " " + option);
                if (ids.size() != tokens.size() + 1 || !std::equal(tokens.begin(), tokens.end(), ids.begin())) {
                    throw std::runtime_error("Candidate is not a single continuation token: " + option);
                }
                options[option] = {{"token", ids.back()}, {"logit", logits[ids.back()]}};
            }
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            auto result = json{
                {"id", row.at("id")}, {"loops", loops}, {"prompt_ids", tokens}, {"token_id", best},
                {"text", std::string(piece, bytes)}, {"logit", logits[best]}, {"seconds", seconds},
                {"full_head_fingerprint", head_fingerprint(logits, vocabulary_size)},
                {"options", options}, {"states", readout.states},
            };
            std::cout << result.dump(-1, ' ', false, json::error_handler_t::replace) << std::endl;
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
