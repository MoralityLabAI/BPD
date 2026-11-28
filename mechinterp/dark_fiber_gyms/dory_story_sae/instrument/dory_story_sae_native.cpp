// Bounded Dory residual capture and intervention at the storyworld action decision.
#define main original_dory_readout_entry
#include "dory_readout.cpp"
#undef main

#include <set>

struct tower_observer {
    vectors observed, writes;
    std::set<std::string> sites;
    std::set<std::string> written;
    bool sample = false;
    bool valid = true;
};

static bool observe_tower(ggml_tensor *tensor, bool ask, void *opaque) {
    auto &run = *static_cast<tower_observer *>(opaque);
    const std::string name = ggml_get_name(tensor);
    const bool wanted = run.sites.count(name) != 0;
    if (ask) return wanted;
    if (!wanted) return true;
    const int64_t width = tensor->ne[0];
    const int64_t length = tensor->ne[1];
    if (tensor->type != GGML_TYPE_F32 || width != 2560 || length < 1 || tensor->ne[2] != 1 ||
        tensor->ne[3] != 1 || tensor->nb[0] != sizeof(float)) {
        run.valid = false;
        return false;
    }
    const auto patch = run.writes.find(name);
    if (patch != run.writes.end()) {
        if (length != 1 || patch->second.size() != size_t(width) || !run.written.insert(name).second) {
            run.valid = false;
            return false;
        }
        ggml_backend_tensor_set(tensor, patch->second.data(), 0, width * sizeof(float));
    }
    if (run.sample) {
        for (int fraction = 0; fraction < 8; ++fraction) {
            const int64_t pos = (length - 1) * fraction / 7;
            auto &row = run.observed[name + "@" + std::to_string(pos)];
            row.resize(width);
            ggml_backend_tensor_get(tensor, row.data(), pos * tensor->nb[1], width * sizeof(float));
        }
    } else {
        auto &row = run.observed[name];
        row.resize(width);
        ggml_backend_tensor_get(tensor, row.data(), (length - 1) * tensor->nb[1], width * sizeof(float));
        if (patch != run.writes.end() && row != patch->second) {
            run.valid = false;
            return false;
        }
    }
    return true;
}

static const float *read_head(llama_context *ctx, int count) {
    const float *values = llama_get_logits_ith(ctx, -1);
    if (!values || !std::all_of(values, values + count, [](float x) { return std::isfinite(x); })) {
        throw std::runtime_error("Invalid full-vocabulary head");
    }
    return values;
}

static int run_tower(const char *model_path, int loops, const char *input_path) {
    ggml_backend_load_all();
    llama_backend_init();
    llama_model_kv_override overrides[2]{};
    std::strcpy(overrides[0].key, "dory.num_loops");
    overrides[0].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
    overrides[0].val_i64 = loops;
    auto mp = llama_model_default_params();
    mp.kv_overrides = overrides;
    mp.n_gpu_layers = 113;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(model_path, mp),
                                                                   llama_model_free);
    if (!model) throw std::runtime_error("Failed to load Dory");
    const llama_vocab *vocab = llama_model_get_vocab(model.get());
    const int count = llama_vocab_n_tokens(vocab);
    tower_observer run;
    auto cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = cp.n_ubatch = 512;
    cp.n_threads = cp.n_threads_batch = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval = observe_tower;
    cp.cb_eval_user_data = &run;
    std::unique_ptr<llama_context, decltype(&llama_free)> context(llama_init_from_model(model.get(), cp), llama_free);
    if (!context) throw std::runtime_error("Failed to create Dory context");
    std::ifstream input(input_path);
    if (!input) throw std::runtime_error("Missing inputs");
    for (std::string line; std::getline(input, line);) {
        const json row = json::parse(line);
        auto ids = tokenize(vocab, row.at("rendered"));
        if (ids.size() > cp.n_batch) throw std::runtime_error("Native prefill capacity exceeded");
        llama_memory_clear(llama_get_memory(context.get()), true);
        std::set<std::string> sites;
        for (int loop = 0; loop < loops; ++loop) {
            for (int site : {25, 45, 55}) sites.insert("l_out-" + std::to_string(site + 40 * loop));
        }
        run = {};
        if (row.value("sample", false)) {
            run.sites = sites;
            run.sample = true;
        }
        if (llama_decode(context.get(), llama_batch_get_one(ids.data(), ids.size())) != 0 || !run.valid) {
            throw std::runtime_error("Story prompt prefill failed");
        }
        const std::string initial_head = head_fingerprint(read_head(context.get(), count), count);
        auto samples = std::move(run.observed);
        auto suffix = row.at("suffix_ids").get<std::vector<llama_token>>();
        if (suffix.empty() || suffix.size() > 16) throw std::runtime_error("Invalid action prefix");
        run = {};
        json greedy_prefix = json::array();
        for (size_t index = 0; index < suffix.size(); ++index) {
            const auto *logits = read_head(context.get(), count);
            greedy_prefix.push_back(std::max_element(logits, logits + count) - logits);
            auto token = suffix[index];
            if (token < 0 || token >= count || llama_vocab_is_eog(vocab, token)) {
                throw std::runtime_error("Invalid forced token");
            }
            if (index + 1 == suffix.size()) {
                if (row.value("capture", false)) run.sites = sites;
                run.writes = row.value("writes", vectors{});
                for (const auto &[site, vector] : run.writes) {
                    if (!sites.count(site) || !std::all_of(vector.begin(), vector.end(), [](float x) { return std::isfinite(x); })) {
                        throw std::runtime_error("Unregistered intervention");
                    }
                    run.sites.insert(site);
                }
            }
            if (llama_decode(context.get(), llama_batch_get_one(&token, 1)) != 0 || !run.valid) {
                throw std::runtime_error("Story action prefix decode failed");
            }
        }
        if (run.written.size() != run.writes.size() || run.observed.size() != run.sites.size()) {
            throw std::runtime_error("Incomplete write or state capture coverage");
        }
        const auto *logits = read_head(context.get(), count);
        const auto best = std::max_element(logits, logits + count) - logits;
        const auto labels = row.at("action_ids").get<std::map<std::string, llama_token>>();
        json actions = json::object();
        for (const auto &[label, token] : labels) {
            if (token < 0 || token >= count) throw std::runtime_error("Bad action vocabulary ID");
            actions[label] = logits[token];
        }
        std::cout << json({{"id", row.at("id")}, {"loops", loops}, {"prompt_ids", ids},
                           {"suffix_ids", suffix}, {"greedy_prefix", greedy_prefix},
                           {"initial_head", initial_head}, {"action_head", head_fingerprint(logits, count)},
                           {"top", best}, {"action_logits", actions}, {"written", run.written},
                           {"states", run.observed}, {"samples", samples}}).dump() << std::endl;
    }
    return 0;
}

int main(int argc, char **argv) {
    try {
        if (argc != 4) throw std::runtime_error("Usage: tower MODEL LOOPS CASES");
        int loops = std::stoi(argv[2]);
        if (loops < 1 || loops > 2) throw std::runtime_error("This pilot uses one or two loops");
        return run_tower(argv[1], loops, argv[3]);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
