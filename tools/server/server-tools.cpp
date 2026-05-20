#include "server-tools.h"
#include "server-clips-goal-router.h"
#include "server-goal.h"
#include "server-goal-executor.h"
#include "server-supervision.h"
#include "server-tool-envelope.h"
#include "server-trace-registry.h"

#include <cpp-httplib/httplib.h>
#include <sheredom/subprocess.h>

#include <filesystem>
#include <fstream>
#include <regex>
#include <thread>
#include <chrono>
#include <atomic>
#include <optional>
#include <algorithm>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <climits>
#include <iterator>
#include <sstream>

namespace fs = std::filesystem;

//
// internal helpers
//

static std::vector<char *> to_cstr_vec(const std::vector<std::string> & v) {
    std::vector<char *> r;
    r.reserve(v.size() + 1);
    for (const auto & s : v) {
        r.push_back(const_cast<char *>(s.c_str()));
    }
    r.push_back(nullptr);
    return r;
}

struct run_proc_result {
    std::string output;
    int  exit_code = -1;
    bool timed_out = false;
};

static run_proc_result run_process(
        const std::vector<std::string> & args,
        size_t max_output,
        int timeout_secs) {
    run_proc_result res;

    subprocess_s proc;
    auto argv = to_cstr_vec(args);

    int options = subprocess_option_no_window
                | subprocess_option_combined_stdout_stderr
                | subprocess_option_inherit_environment
                | subprocess_option_search_user_path;

    if (subprocess_create(argv.data(), options, &proc) != 0) {
        res.output = "failed to spawn process";
        return res;
    }

    std::atomic<bool> done{false};
    std::atomic<bool> timed_out{false};

    std::thread timeout_thread([&]() {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_secs);
        while (!done.load()) {
            if (std::chrono::steady_clock::now() >= deadline) {
                timed_out.store(true);
                subprocess_terminate(&proc);
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    FILE * f = subprocess_stdout(&proc);
    std::string output;
    bool truncated = false;
    if (f) {
        char buf[4096];
        while (fgets(buf, sizeof(buf), f) != nullptr) {
            if (!truncated) {
                size_t len = strlen(buf);
                if (output.size() + len <= max_output) {
                    output.append(buf, len);
                } else {
                    output.append(buf, max_output - output.size());
                    truncated = true;
                }
            }
        }
    }

    done.store(true);
    if (timeout_thread.joinable()) {
        timeout_thread.join();
    }

    subprocess_join(&proc, &res.exit_code);
    subprocess_destroy(&proc);

    res.output    = output;
    res.timed_out = timed_out.load();
    if (truncated) {
        res.output += "\n[output truncated]";
    }
    return res;
}

static uint64_t fnv1a64_bytes(const std::string & text) {
    uint64_t hash = 1469598103934665603ULL;
    for (unsigned char ch : text) {
        hash ^= ch;
        hash *= 1099511628211ULL;
    }
    return hash;
}

static std::string sanitize_json_text(std::string text) {
    for (char & ch : text) {
        const unsigned char byte = static_cast<unsigned char>(ch);
        if (byte < 0x20 && ch != '\n' && ch != '\r' && ch != '\t') {
            ch = ' ';
        }
    }
    return text;
}

static json tool_error_json(std::string message) {
    return {{"error", sanitize_json_text(std::move(message))}};
}

static bool json_boolish_value(const json & body, const std::string & key, bool default_value) {
    if (!body.contains(key) || body.at(key).is_null()) {
        return default_value;
    }
    const json & value = body.at(key);
    if (value.is_boolean()) {
        return value.get<bool>();
    }
    if (value.is_number_integer()) {
        return value.get<int>() != 0;
    }
    if (value.is_string()) {
        std::string s = value.get<std::string>();
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        if (s == "true" || s == "1" || s == "yes") {
            return true;
        }
        if (s == "false" || s == "0" || s == "no") {
            return false;
        }
    }
    return default_value;
}

static bool json_has_nonempty_value(const json & body, const std::string & key) {
    if (!body.contains(key) || body.at(key).is_null()) {
        return false;
    }
    const json & value = body.at(key);
    if (value.is_string()) {
        return !value.get<std::string>().empty();
    }
    if (value.is_boolean()) {
        return value.get<bool>();
    }
    if (value.is_array() || value.is_object()) {
        return !value.empty();
    }
    return true;
}

struct remote_lan_agent_runtime_config {
    std::string host = "127.0.0.1";
    int port = 18080;
    std::string tools_path = "/mcp";
    int timeout_secs = 120;
    int max_continuation_steps = 4096;
};

static std::string trim_copy(std::string value) {
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

static std::optional<std::string> env_string_nonempty(const char * key) {
    const char * raw = std::getenv(key);
    if (raw == nullptr) {
        return std::nullopt;
    }
    std::string value = trim_copy(raw);
    if (value.empty()) {
        return std::nullopt;
    }
    return value;
}

static std::optional<int> parse_int_strict(const std::string & value) {
    try {
        size_t pos = 0;
        int parsed = std::stoi(value, &pos, 10);
        if (pos != value.size()) {
            return std::nullopt;
        }
        return parsed;
    } catch (...) {
        return std::nullopt;
    }
}

static std::optional<int> env_int_value(const char * key) {
    const auto value = env_string_nonempty(key);
    if (!value) {
        return std::nullopt;
    }
    return parse_int_strict(*value);
}

static const json * resolve_lan_agent_config_json(const common_params * params) {
    if (params == nullptr || params->webui_config_json.empty()) {
        return nullptr;
    }
    try {
        static thread_local json parsed = json::object();
        parsed = json::parse(params->webui_config_json);
        if (!parsed.is_object()) {
            return nullptr;
        }
        if (parsed.contains("serverTools") && parsed.at("serverTools").is_object()) {
            const json & server_tools = parsed.at("serverTools");
            if (server_tools.contains("lanAgent") && server_tools.at("lanAgent").is_object()) {
                return &server_tools.at("lanAgent");
            }
        }
        if (parsed.contains("lanAgent") && parsed.at("lanAgent").is_object()) {
            return &parsed.at("lanAgent");
        }
        return &parsed;
    } catch (const std::exception & e) {
        SRV_WRN("server_tools: failed to parse webui_config_json for LAN agent config: %s\n", e.what());
        return nullptr;
    }
}

static std::optional<std::string> json_string_value(const json * cfg, const std::string & key) {
    if (cfg == nullptr || !cfg->is_object() || !cfg->contains(key) || !cfg->at(key).is_string()) {
        return std::nullopt;
    }
    std::string value = trim_copy(cfg->at(key).get<std::string>());
    if (value.empty()) {
        return std::nullopt;
    }
    return value;
}

static std::optional<int> json_int_value(const json * cfg, const std::string & key) {
    if (cfg == nullptr || !cfg->is_object() || !cfg->contains(key) || cfg->at(key).is_null()) {
        return std::nullopt;
    }
    const json & value = cfg->at(key);
    if (value.is_number_integer()) {
        return value.get<int>();
    }
    if (value.is_string()) {
        return parse_int_strict(trim_copy(value.get<std::string>()));
    }
    return std::nullopt;
}

static remote_lan_agent_runtime_config resolve_remote_lan_agent_runtime_config(const common_params * params) {
    remote_lan_agent_runtime_config cfg;

    if (auto value = env_string_nonempty("LLAMA_SERVER_TOOL_LAN_AGENT_HOST")) {
        cfg.host = *value;
    }
    if (auto value = env_int_value("LLAMA_SERVER_TOOL_LAN_AGENT_PORT")) {
        cfg.port = *value;
    }
    if (auto value = env_string_nonempty("LLAMA_SERVER_TOOL_LAN_AGENT_TOOLS_PATH")) {
        cfg.tools_path = *value;
    }
    if (auto value = env_int_value("LLAMA_SERVER_TOOL_LAN_AGENT_TIMEOUT_SECS")) {
        cfg.timeout_secs = *value;
    }
    if (auto value = env_int_value("LLAMA_SERVER_TOOL_LAN_AGENT_MAX_CONTINUATION_STEPS")) {
        cfg.max_continuation_steps = *value;
    }

    const json * json_cfg = resolve_lan_agent_config_json(params);

    if (auto value = json_string_value(json_cfg, "serverToolLanAgentHost")) {
        cfg.host = *value;
    } else if (auto value = json_string_value(json_cfg, "host")) {
        cfg.host = *value;
    }
    if (auto value = json_int_value(json_cfg, "serverToolLanAgentPort")) {
        cfg.port = *value;
    } else if (auto value = json_int_value(json_cfg, "port")) {
        cfg.port = *value;
    }
    if (auto value = json_string_value(json_cfg, "serverToolLanAgentToolsPath")) {
        cfg.tools_path = *value;
    } else if (auto value = json_string_value(json_cfg, "toolsPath")) {
        cfg.tools_path = *value;
    } else if (auto value = json_string_value(json_cfg, "path")) {
        cfg.tools_path = *value;
    }
    if (auto value = json_int_value(json_cfg, "serverToolLanAgentTimeoutSecs")) {
        cfg.timeout_secs = *value;
    } else if (auto value = json_int_value(json_cfg, "timeoutSecs")) {
        cfg.timeout_secs = *value;
    }
    if (auto value = json_int_value(json_cfg, "serverToolLanAgentMaxContinuationSteps")) {
        cfg.max_continuation_steps = *value;
    } else if (auto value = json_int_value(json_cfg, "maxContinuationSteps")) {
        cfg.max_continuation_steps = *value;
    }

    if (cfg.port <= 0) {
        cfg.port = 18080;
    }
    if (cfg.timeout_secs <= 0) {
        cfg.timeout_secs = 120;
    }
    if (cfg.max_continuation_steps <= 0) {
        cfg.max_continuation_steps = 4096;
    }
    if (cfg.tools_path.empty()) {
        cfg.tools_path = "/mcp";
    } else if (cfg.tools_path.front() != '/') {
        cfg.tools_path.insert(cfg.tools_path.begin(), '/');
    }

    return cfg;
}

static json extract_remote_lan_agent_result(const json & rpc_result) {
    if (!rpc_result.is_object()) {
        return tool_error_json("remote LAN agent MCP returned non-object JSON-RPC payload");
    }
    if (rpc_result.contains("error") && rpc_result.at("error").is_object()) {
        const json & err = rpc_result.at("error");
        return {
            {"error", err.value("message", std::string("remote LAN agent MCP error"))},
            {"error_code", err.value("code", 0)},
        };
    }
    if (!rpc_result.contains("result") || !rpc_result.at("result").is_object()) {
        return tool_error_json("remote LAN agent MCP response is missing result object");
    }

    const json & result = rpc_result.at("result");
    if (json_boolish_value(result, "isError", false)) {
        std::string message = "remote LAN agent MCP result isError";
        if (result.contains("structuredContent") && result.at("structuredContent").is_object()) {
            const json & sc = result.at("structuredContent");
            if (sc.contains("error") && sc.at("error").is_string()) {
                message = sc.at("error").get<std::string>();
            }
        }
        if (result.contains("content") && result.at("content").is_array() && !result.at("content").empty()) {
            const json & item0 = result.at("content").front();
            if (item0.is_object() && item0.contains("text") && item0.at("text").is_string()) {
                message = item0.at("text").get<std::string>();
            }
        }
        return tool_error_json(message);
    }

    if (result.contains("structuredContent") && result.at("structuredContent").is_object()) {
        return result.at("structuredContent");
    }

    if (result.contains("content") && result.at("content").is_array() && !result.at("content").empty()) {
        const json & item0 = result.at("content").front();
        if (item0.is_object() && item0.contains("text") && item0.at("text").is_string()) {
            const std::string text = item0.at("text").get<std::string>();
            try {
                json parsed = json::parse(text);
                if (parsed.is_object()) {
                    return parsed;
                }
            } catch (...) {
            }
            return {{"plain_text_response", text}};
        }
    }

    return tool_error_json("remote LAN agent MCP result does not contain structuredContent");
}

static json invoke_remote_lan_agent_tool(
        const remote_lan_agent_runtime_config & cfg,
        const std::string & tool_name,
        const json & params) {
    httplib::Client cli(cfg.host, cfg.port);
    cli.set_follow_location(true);
    cli.set_connection_timeout(5, 0);
    cli.set_read_timeout(cfg.timeout_secs, 0);
    cli.set_write_timeout(cfg.timeout_secs, 0);

    static std::atomic<int> rpc_id{5000};
    json body = {
        {"jsonrpc", "2.0"},
        {"id", ++rpc_id},
        {"method", "tools/call"},
        {"params", {
            {"name", tool_name},
            {"arguments", params},
        }},
    };

    auto res = cli.Post(
        cfg.tools_path,
        safe_json_to_str(body),
        "application/json; charset=utf-8");

    if (!res) {
        return tool_error_json("failed to invoke remote LAN agent MCP tool: " + tool_name);
    }

    try {
        json rpc_result = json::parse(res->body);
        return extract_remote_lan_agent_result(rpc_result);
    } catch (const std::exception & e) {
        return tool_error_json(std::string("remote LAN agent MCP returned invalid JSON: ") + e.what());
    }
}

static json consume_remote_lan_agent_continuation(
        const remote_lan_agent_runtime_config & cfg,
        const std::string & initial_tool_name,
        const json & initial_params) {
    std::string tool_name = initial_tool_name;
    json params = initial_params;
    json continuation_steps = json::array();

    for (int step = 0; step < cfg.max_continuation_steps; ++step) {
        SRV_INF("lan_agent_mcp_continue: step=%d tool='%s'\n", step, tool_name.c_str());
        json result = invoke_remote_lan_agent_tool(cfg, tool_name, params);

        const std::string status = result.value("status", "");
        const std::string status_code = result.value("status_code", "");
        const std::string acceptance_status = result.value("acceptance_status", "");
        const std::string task_completion = result.value("task_completion", "");
        const std::string batch_completion = result.value("batch_completion", "");
        const std::string content_read_completion = result.value("content_read_completion", "");
        const std::string goal_status = result.value("goal_status", "");
        const std::string current_file_path = result.value("current_file_path", result.value("path", ""));
        const bool file_complete = json_boolish_value(result, "file_complete", false);
        const bool directory_complete =
            json_boolish_value(result, "directory_complete", false) ||
            batch_completion == "complete";
        const bool continue_required =
            json_boolish_value(result, "continue_required", false) ||
            json_boolish_value(result, "auto_continue_required", false);
        const bool assistant_response_allowed =
            json_boolish_value(result, "assistant_response_allowed", true);
        const bool final_answer_allowed =
            json_boolish_value(result, "final_answer_allowed", true);
        const bool requires_continue =
            status == "needs_continue" ||
            acceptance_status == "continue" ||
            task_completion == "incomplete" ||
            batch_completion == "incomplete" ||
            content_read_completion == "incomplete" ||
            goal_status == "not_complete" ||
            continue_required ||
            !assistant_response_allowed ||
            !final_answer_allowed;

        continuation_steps.push_back({
            {"step", step},
            {"tool", tool_name},
            {"status", status},
            {"status_code", status_code},
            {"acceptance_status", acceptance_status},
            {"current_file_path", current_file_path},
            {"file_complete", file_complete},
            {"directory_complete", directory_complete},
            {"requires_continue", requires_continue},
        });

        SRV_INF(
            "lan_agent_mcp_continue: step=%d status='%s' acceptance='%s' file_complete=%d directory_complete=%d requires_continue=%d\n",
            step,
            status.c_str(),
            acceptance_status.c_str(),
            file_complete ? 1 : 0,
            directory_complete ? 1 : 0,
            requires_continue ? 1 : 0);

        if (json_has_nonempty_value(result, "error") ||
            json_has_nonempty_value(result, "error_code") ||
            json_has_nonempty_value(result, "error_message")) {
            result["continuation_steps"] = continuation_steps;
            return result;
        }

        if (status == "failed") {
            result["continuation_steps"] = continuation_steps;
            return result;
        }

        if (requires_continue) {
            json next_call = json::object();
            if (result.contains("next_call_json")) {
                const json & next_call_json = result.at("next_call_json");
                if (next_call_json.is_object()) {
                    next_call = next_call_json;
                } else if (next_call_json.is_string()) {
                    try {
                        json parsed = json::parse(next_call_json.get<std::string>());
                        if (parsed.is_object()) {
                            next_call = std::move(parsed);
                        }
                    } catch (...) {
                    }
                }
            }
            if (next_call.empty() &&
                result.contains("required_tool_arguments_json") &&
                result.at("required_tool_arguments_json").is_string()) {
                try {
                    json parsed = json::parse(result.at("required_tool_arguments_json").get<std::string>());
                    if (parsed.is_object()) {
                        next_call = std::move(parsed);
                    }
                } catch (...) {
                }
            }
            if (next_call.empty() &&
                result.contains("next_action_0_tool_name") &&
                result.at("next_action_0_tool_name").is_string() &&
                result.contains("next_action_0_params_json") &&
                result.at("next_action_0_params_json").is_string()) {
                try {
                    next_call = json{
                        {"name", result.at("next_action_0_tool_name").get<std::string>()},
                        {"arguments", json::parse(result.at("next_action_0_params_json").get<std::string>())},
                    };
                } catch (...) {
                }
            }

            if (!next_call.is_object() || next_call.empty()) {
                return {
                    {"error", "remote LAN agent MCP returned continuation-required result without valid next_call_json"},
                    {"continuation_steps", continuation_steps},
                };
            }

            const std::string next_tool_name = next_call.value("tool", next_call.value("name", ""));
            if (next_tool_name.empty()) {
                return {
                    {"error", "remote LAN agent MCP next_call_json is missing tool/name"},
                    {"continuation_steps", continuation_steps},
                };
            }

            tool_name = next_tool_name;
            if (next_call.contains("params") && next_call.at("params").is_object()) {
                params = next_call.at("params");
            } else if (next_call.contains("arguments") && next_call.at("arguments").is_object()) {
                params = next_call.at("arguments");
            } else {
                params = json::object();
            }
            continue;
        }

        if (status == "success" || directory_complete || file_complete) {
            result["continuation_steps"] = continuation_steps;
            return result;
        }

        result["continuation_steps"] = continuation_steps;
        return result;
    }

    return {
        {"error", "remote LAN agent MCP continuation exceeded max steps"},
        {"continuation_steps", continuation_steps},
    };
}

static json decorate_remote_lan_agent_result(
        const remote_lan_agent_runtime_config & cfg,
        json result) {
    result["delegated_to_remote_mcp"] = true;
    result["remote_host"] = cfg.host;
    result["remote_port"] = cfg.port;
    result["remote_path"] = cfg.tools_path;
    return result;
}

static std::vector<std::string> read_glob_patterns(
        const json & params,
        const char * single_key,
        const char * list_key,
        const std::vector<std::string> & fallback) {
    if (params.contains(list_key) && params.at(list_key).is_array()) {
        std::vector<std::string> patterns;
        for (const auto & item : params.at(list_key)) {
            if (item.is_string()) {
                patterns.push_back(item.get<std::string>());
            }
        }
        if (!patterns.empty()) {
            return patterns;
        }
    }

    if (params.contains(single_key) && params.at(single_key).is_string()) {
        return {params.at(single_key).get<std::string>()};
    }

    return fallback;
}

static bool matches_any_glob(const std::vector<std::string> & patterns, const std::string & value) {
    for (const auto & pattern : patterns) {
        if (glob_match(pattern, value)) {
            return true;
        }
        if (pattern.rfind("**/", 0) == 0 && glob_match(pattern.substr(3), value)) {
            return true;
        }
    }
    return false;
}

json server_tool::to_json() {
    return {
        {"display_name", display_name},
        {"tool", name},
        {"type", "builtin"},
        {"permissions", json{
            {"write", permission_write}
        }},
        {"definition", get_definition()},
    };
}

//
// read_file: read a file with optional line range and line-number prefix
//

static constexpr size_t SERVER_TOOL_READ_FILE_MAX_SIZE = 16 * 1024; // 16 KB

struct server_tool_read_file : server_tool {
    server_tool_read_file() {
        name = "read_file";
        display_name = "Read file";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Read the contents of a file. Optionally specify a 1-based line range. "
                                "If append_loc is true, each line is prefixed with its line number (e.g. \"1\u2192 ...\")."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",       {{"type", "string"},  {"description", "Path to the file"}}},
                        {"start_line", {{"type", "integer"}, {"description", "First line to read, 1-based (default: 1)"}}},
                        {"end_line",   {{"type", "integer"}, {"description", "Last line to read, 1-based inclusive (default: end of file)"}}},
                        {"append_loc", {{"type", "boolean"}, {"description", "Prefix each line with its line number"}}},
                    }},
                    {"required", json::array({"path"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string path  = params.at("path").get<std::string>();
        int  start_line   = json_value(params, "start_line", 1);
        int  end_line     = json_value(params, "end_line",  -1); // -1 = no limit
        bool append_loc   = json_value(params, "append_loc", false);

        std::error_code ec;
        uintmax_t file_size = fs::file_size(path, ec);
        if (ec) {
            return tool_error_json("cannot stat file: " + ec.message());
        }
        if (file_size > SERVER_TOOL_READ_FILE_MAX_SIZE && end_line == -1) {
            return tool_error_json(string_format(
                "file too large (%zu bytes, max %zu). Use start_line/end_line to read a portion.",
                (size_t)file_size, SERVER_TOOL_READ_FILE_MAX_SIZE));
        }

        std::ifstream f(path);
        if (!f) {
            return tool_error_json("failed to open file: " + path);
        }

        std::string result;
        std::string line;
        int lineno = 0;
        int lines_read = 0;
        bool reached_eof = true;
        bool output_truncated = false;

        while (std::getline(f, line)) {
            lineno++;
            if (lineno < start_line) continue;
            if (end_line != -1 && lineno > end_line) break;

            std::string out_line;
            if (append_loc) {
                out_line = std::to_string(lineno) + "\u2192 " + line + "\n";
            } else {
                out_line = line + "\n";
            }

            if (result.size() + out_line.size() > SERVER_TOOL_READ_FILE_MAX_SIZE) {
                result += "[output truncated]";
                output_truncated = true;
                reached_eof = false;
                break;
            }
            result += out_line;
            lines_read++;
        }

        if (!output_truncated && end_line != -1 && lineno >= end_line) {
            reached_eof = f.eof();
        }

        return {
            {"path", path},
            {"start_line", start_line},
            {"end_line", end_line == -1 ? lineno : end_line},
            {"append_loc", append_loc},
            {"plain_text_response", result},
            {"line_count_read", lines_read},
            {"reached_eof", reached_eof},
            {"output_truncated", output_truncated},
        };
    }
};

//
// file_info: inspect a file and return lightweight metadata
//

struct server_tool_file_info : server_tool {
    server_tool_file_info() {
        name = "file_info";
        display_name = "File info";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Return file metadata such as size, line count, and a stable content hash."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path", {{"type", "string"}, {"description", "Path to the file"}}},
                    }},
                    {"required", json::array({"path"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        const std::string path = params.at("path").get<std::string>();

        std::error_code ec;
        if (!fs::is_regular_file(path, ec) || ec) {
            return tool_error_json("path does not exist or is not a regular file: " + path);
        }

        const uintmax_t size_bytes = fs::file_size(path, ec);
        if (ec) {
            return tool_error_json("cannot stat file: " + ec.message());
        }

        std::ifstream f(path, std::ios::binary);
        if (!f) {
            return tool_error_json("failed to open file: " + path);
        }

        std::string contents((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        int line_count = 0;
        for (char ch : contents) {
            if (ch == '\n') {
                line_count++;
            }
        }
        if (!contents.empty() && contents.back() != '\n') {
            line_count++;
        }

        std::ostringstream hash_stream;
        hash_stream << std::hex << fnv1a64_bytes(contents);

        return {
            {"path", path},
            {"size_bytes", static_cast<uint64_t>(size_bytes)},
            {"line_count", line_count},
            {"content_hash", "fnv1a64:" + hash_stream.str()},
        };
    }
};

//
// file_glob_search: find files matching a glob pattern under a base directory
//

static constexpr size_t SERVER_TOOL_FILE_SEARCH_MAX_RESULTS = 100;

struct server_tool_file_glob_search : server_tool {
    server_tool_file_glob_search() {
        name = "file_glob_search";
        display_name = "File search";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Recursively search for files matching a glob pattern under a directory."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",             {{"type", "string"}, {"description", "Base directory to search in"}}},
                        {"include",          {{"type", "string"}, {"description", "Glob pattern for files to include (e.g. \"**/*.cpp\"). Default: **"}}},
                        {"include_patterns", {{"type", "array"},  {"description", "Optional array of include glob patterns"}}},
                        {"exclude",          {{"type", "string"}, {"description", "Glob pattern for files to exclude"}}},
                        {"exclude_patterns", {{"type", "array"},  {"description", "Optional array of exclude glob patterns"}}},
                        {"max_results",      {{"type", "integer"}, {"description", "Maximum number of results to return"}}},
                    }},
                    {"required", json::array({"path"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string base    = params.at("path").get<std::string>();
        const auto include_patterns = read_glob_patterns(params, "include", "include_patterns", {"**"});
        const auto exclude_patterns = read_glob_patterns(params, "exclude", "exclude_patterns", {});
        const size_t max_results = static_cast<size_t>(std::max(1, json_value(params, "max_results", static_cast<int>(SERVER_TOOL_FILE_SEARCH_MAX_RESULTS))));

        std::ostringstream output_text;
        json matches = json::array();
        size_t count = 0;
        size_t total_matches = 0;

        std::error_code ec;
        for (const auto & entry : fs::recursive_directory_iterator(base,
                fs::directory_options::skip_permission_denied, ec)) {
            if (!entry.is_regular_file()) continue;

            std::string rel = fs::relative(entry.path(), base, ec).string();
            if (ec) continue;
            std::replace(rel.begin(), rel.end(), '\\', '/');

            if (!matches_any_glob(include_patterns, rel)) continue;
            if (!exclude_patterns.empty() && matches_any_glob(exclude_patterns, rel)) continue;

            total_matches++;
            if (count >= max_results) {
                continue;
            }

            const std::string full_path = entry.path().string();
            output_text << full_path << "\n";
            matches.push_back(full_path);

            ++count;

        }

        output_text << "\n---\nTotal matches: " << total_matches << "\n";

        return {
            {"plain_text_response", output_text.str()},
            {"matches", matches},
            {"returned_count", count},
            {"total_matches", total_matches},
            {"path", base},
        };
    }
};

//
// grep_search: search for a regex pattern in files
//

static constexpr size_t SERVER_TOOL_GREP_SEARCH_MAX_RESULTS = 100;

struct server_tool_grep_search : server_tool {
    server_tool_grep_search() {
        name = "grep_search";
        display_name = "Grep search";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Search for a regex pattern in files under a path. Returns matching lines."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",                {{"type", "string"},  {"description", "File or directory to search in"}}},
                        {"pattern",             {{"type", "string"},  {"description", "Regular expression pattern to search for"}}},
                        {"include",             {{"type", "string"},  {"description", "Glob pattern to filter files (default: **)"}}},
                        {"exclude",             {{"type", "string"},  {"description", "Glob pattern to exclude files"}}},
                        {"return_line_numbers", {{"type", "boolean"}, {"description", "If true, include line numbers in results"}}},
                    }},
                    {"required", json::array({"path", "pattern"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string path    = params.at("path").get<std::string>();
        std::string pat_str = params.at("pattern").get<std::string>();
        std::string include = json_value(params, "include", std::string("**"));
        std::string exclude = json_value(params, "exclude", std::string(""));
        bool show_lineno    = json_value(params, "return_line_numbers", false);

        std::regex pattern;
        try {
            pattern = std::regex(pat_str);
        } catch (const std::regex_error & e) {
            return tool_error_json(std::string("invalid regex: ") + e.what());
        }

        std::ostringstream output_text;
        size_t total = 0;

        auto search_file = [&](const fs::path & fpath) {
            std::ifstream f(fpath);
            if (!f) return;
            std::string line;
            int lineno = 0;
            while (std::getline(f, line) && total < SERVER_TOOL_GREP_SEARCH_MAX_RESULTS) {
                lineno++;
                if (std::regex_search(line, pattern)) {
                    output_text << fpath.string() << ":";
                    if (show_lineno) {
                        output_text << lineno << ":";
                    }
                    output_text << line << "\n";
                    total++;
                }
            }
        };

        std::error_code ec;
        if (fs::is_regular_file(path, ec)) {
            search_file(path);
        } else if (fs::is_directory(path, ec)) {
            for (const auto & entry : fs::recursive_directory_iterator(path,
                    fs::directory_options::skip_permission_denied, ec)) {
                if (!entry.is_regular_file()) continue;
                if (total >= SERVER_TOOL_GREP_SEARCH_MAX_RESULTS) break;

                std::string rel = fs::relative(entry.path(), path, ec).string();
                if (ec) continue;
                std::replace(rel.begin(), rel.end(), '\\', '/');

                if (!glob_match(include, rel)) continue;
                if (!exclude.empty() && glob_match(exclude, rel)) continue;

                search_file(entry.path());
            }
        } else {
            return {{"error", "path does not exist: " + path}};
        }

        output_text << "\n\n---\nTotal matches: " << total << "\n";

        return {{"plain_text_response", output_text.str()}};
    }
};

//
// exec_shell_command: run an arbitrary shell command
//

static constexpr size_t SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE = 16 * 1024; // 16 KB
static constexpr int    SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_TIMEOUT     = 60;        // seconds

struct server_tool_exec_shell_command : server_tool {
    server_tool_exec_shell_command() {
        name = "exec_shell_command";
        display_name = "Execute shell command";
        permission_write = true;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Execute a shell command and return its output (stdout and stderr combined)."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"command",         {{"type", "string"},  {"description", "Shell command to execute"}}},
                        {"timeout",         {{"type", "integer"}, {"description", string_format("Timeout in seconds (default 10, max %d)", SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_TIMEOUT)}}},
                        {"max_output_size", {{"type", "integer"}, {"description", string_format("Maximum output size in bytes (default %zu)", SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE)}}},
                    }},
                    {"required", json::array({"command"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string command   = params.at("command").get<std::string>();
        int    timeout        = json_value(params, "timeout",         10);
        size_t max_output     = (size_t) json_value(params, "max_output_size", (int) SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE);

        timeout    = std::min(timeout,    SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_TIMEOUT);
        max_output = std::min(max_output, SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE);

#ifdef _WIN32
        std::vector<std::string> args = {"cmd", "/c", command};
#else
        std::vector<std::string> args = {"sh", "-c", command};
#endif

        auto res = run_process(args, max_output, timeout);

        std::string text_output = res.output;
        text_output += string_format("\n[exit code: %d]", res.exit_code);
        if (res.timed_out) {
            text_output += " [exit due to timed out]";
        }

        return {{"plain_text_response", text_output}};
    }
};

//
// write_file: create or overwrite a file
//

struct server_tool_write_file : server_tool {
    server_tool_write_file() {
        name = "write_file";
        display_name = "Write file";
        permission_write = true;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Write content to a file, creating it (including parent directories) if it does not exist. May use with edit_file for more complex edits."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",    {{"type", "string"}, {"description", "Path of the file to write"}}},
                        {"content", {{"type", "string"}, {"description", "Content to write"}}},
                    }},
                    {"required", json::array({"path", "content"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string path    = params.at("path").get<std::string>();
        std::string content = params.at("content").get<std::string>();

        std::error_code ec;
        fs::path fpath(path);
        if (fpath.has_parent_path()) {
            fs::create_directories(fpath.parent_path(), ec);
            if (ec) {
                return {{"error", "failed to create directories: " + ec.message()}};
            }
        }

        std::ofstream f(path, std::ios::binary);
        if (!f) {
            return {{"error", "failed to open file for writing: " + path}};
        }
        f << content;
        if (!f) {
            return {{"error", "failed to write file: " + path}};
        }

        return {{"result", "file written successfully"}, {"path", path}, {"bytes", content.size()}};
    }
};

//
// edit_file: edit file content via line-based changes
//

struct server_tool_edit_file : server_tool {
    server_tool_edit_file() {
        name = "edit_file";
        display_name = "Edit file";
        permission_write = true;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description",
                    "Edit a file by applying a list of line-based changes. "
                    "Each change targets a 1-based inclusive line range and has a mode: "
                    "\"replace\" (replace lines with content), "
                    "\"delete\" (remove lines, content must be empty string), "
                    "\"append\" (insert content after line_end). "
                    "Set line_start to -1 to target the end of file (line_end is ignored in that case). "
                    "Changes must not overlap. They are applied in reverse line order automatically."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",    {{"type", "string"}, {"description", "Path to the file to edit"}}},
                        {"changes", {
                            {"type", "array"},
                            {"description", "List of changes to apply"},
                            {"items", {
                                {"type", "object"},
                                {"properties", {
                                    {"mode",       {{"type", "string"},  {"description", "\"replace\", \"delete\", or \"append\""}}},
                                    {"line_start", {{"type", "integer"}, {"description", "First line of the range (1-based); use -1 for end of file"}}},
                                    {"line_end",   {{"type", "integer"}, {"description", "Last line of the range (1-based, inclusive); ignored when line_start is -1"}}},
                                    {"content",    {{"type", "string"},  {"description", "Content to insert; must be empty string for delete mode"}}},
                                }},
                                {"required", json::array({"mode", "line_start", "line_end", "content"})},
                            }},
                        }},
                    }},
                    {"required", json::array({"path", "changes"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string path = params.at("path").get<std::string>();
        const json & changes = params.at("changes");

        if (!changes.is_array()) {
            return {{"error", "\"changes\" must be an array"}};
        }

        // read file into lines
        std::ifstream fin(path);
        if (!fin) {
            return {{"error", "failed to open file: " + path}};
        }
        std::vector<std::string> lines;
        {
            std::string line;
            while (std::getline(fin, line)) {
                lines.push_back(line);
            }
        }
        fin.close();

        // validate and collect changes, then sort descending by line_start
        struct change_entry {
            std::string mode;
            int line_start; // 1-based
            int line_end;   // 1-based inclusive
            std::string content;
        };
        std::vector<change_entry> entries;
        entries.reserve(changes.size());

        for (const auto & ch : changes) {
            change_entry e;
            e.mode       = ch.at("mode").get<std::string>();
            e.line_start = ch.at("line_start").get<int>();
            e.line_end   = ch.at("line_end").get<int>();
            e.content    = ch.at("content").get<std::string>();

            if (e.mode != "replace" && e.mode != "delete" && e.mode != "append") {
                return {{"error", "invalid mode \"" + e.mode + "\"; must be replace, delete, or append"}};
            }
            if (e.mode == "delete" && !e.content.empty()) {
                return {{"error", "content must be empty string for delete mode"}};
            }
            int n = (int) lines.size();
            if (e.line_start == -1) {
                // -1 means end of file; line_end is ignored — normalize to point past last line
                e.line_start = n + 1;
                e.line_end   = n + 1;
            } else {
                if (e.line_start < 1 || e.line_end < e.line_start) {
                    return {{"error", string_format("invalid line range [%d, %d]", e.line_start, e.line_end)}};
                }
                if (e.line_end > n) {
                    return {{"error", string_format("line_end %d exceeds file length %d", e.line_end, n)}};
                }
            }
            entries.push_back(std::move(e));
        }

        // sort descending so earlier-indexed changes don't shift later ones
        std::sort(entries.begin(), entries.end(), [](const change_entry & a, const change_entry & b) {
            return a.line_start > b.line_start;
        });

        // apply changes (0-based indices internally)
        for (const auto & e : entries) {
            int idx_start = e.line_start - 1; // 0-based
            int idx_end   = e.line_end   - 1; // 0-based inclusive

            // split content into lines (preserve trailing newline awareness)
            std::vector<std::string> new_lines;
            if (!e.content.empty()) {
                std::istringstream ss(e.content);
                std::string ln;
                while (std::getline(ss, ln)) {
                    new_lines.push_back(ln);
                }
                // if content ends with \n, getline consumed it — no extra empty line needed
                // if content does NOT end with \n, last line is still captured correctly
            }

            if (e.mode == "replace") {
                // erase [idx_start, idx_end] and insert new_lines
                lines.erase(lines.begin() + idx_start, lines.begin() + idx_end + 1);
                lines.insert(lines.begin() + idx_start, new_lines.begin(), new_lines.end());
            } else if (e.mode == "delete") {
                lines.erase(lines.begin() + idx_start, lines.begin() + idx_end + 1);
            } else { // append
                // idx_end + 1 may equal lines.size() when line_start == -1 (end of file)
                lines.insert(lines.begin() + idx_end + 1, new_lines.begin(), new_lines.end());
            }
        }

        // write file back
        std::ofstream fout(path, std::ios::binary);
        if (!fout) {
            return {{"error", "failed to open file for writing: " + path}};
        }
        for (size_t i = 0; i < lines.size(); i++) {
            fout << lines[i];
            if (i + 1 < lines.size()) {
                fout << "\n";
            }
        }
        if (!lines.empty()) {
            fout << "\n";
        }
        if (!fout) {
            return {{"error", "failed to write file: " + path}};
        }

        return {{"result", "file edited successfully"}, {"path", path}, {"lines", (int) lines.size()}};
    }
};

//
// apply_diff: apply a unified diff via git apply
//

struct server_tool_apply_diff : server_tool {
    server_tool_apply_diff() {
        name = "apply_diff";
        display_name = "Apply diff";
        permission_write = true;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Apply a unified diff to edit one or more files using git apply. Use this instead of edit_file when the changes are complex."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"diff", {{"type", "string"}, {"description", "Unified diff content in git diff format"}}},
                    }},
                    {"required", json::array({"diff"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string diff = params.at("diff").get<std::string>();

        // write diff to a temporary file
        static std::atomic<int> counter{0};
        std::string tmp_path = (fs::temp_directory_path() /
            ("llama_patch_" + std::to_string(++counter) + ".patch")).string();

        {
            std::ofstream f(tmp_path, std::ios::binary);
            if (!f) {
                return {{"error", "failed to create temp patch file"}};
            }
            f << diff;
        }

        auto res = run_process({"git", "apply", tmp_path}, 4096, 10);

        std::error_code ec;
        fs::remove(tmp_path, ec);

        if (res.exit_code != 0) {
            return {{"error", "git apply failed (exit " + std::to_string(res.exit_code) + "): " + res.output}};
        }
        return {{"result", "patch applied successfully"}};
    }
};

//
// lan_agent_list_directory: compatibility wrapper for codex-lan-agent directory listing
//

struct server_tool_lan_agent_list_directory : server_tool {
    explicit server_tool_lan_agent_list_directory(remote_lan_agent_runtime_config cfg)
        : cfg(std::move(cfg)) {
        name = "lan_agent_list_directory";
        display_name = "LAN agent list directory";
        permission_write = false;
    }
    remote_lan_agent_runtime_config cfg;

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "List files under a directory for codex-lan-agent compatibility."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"directory_path", {{"type", "string"}, {"description", "Directory to list"}}},
                        {"max_entries",    {{"type", "integer"}, {"description", "Maximum number of file entries to return"}}},
                        {"request_id",     {{"type", "string"}, {"description", "Optional request identifier"}}},
                        {"trace_id",       {{"type", "string"}, {"description", "Optional trace identifier"}}},
                    }} ,
                    {"required", json::array({"directory_path"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        SRV_INF("lan_agent_list_directory: delegating to remote LAN agent MCP continuation consumer\n", 0);
        return decorate_remote_lan_agent_result(cfg, consume_remote_lan_agent_continuation(cfg, name, params));
    }
};

//
// lan_agent_read_text_file: compatibility wrapper for codex-lan-agent paged text reads
//

struct server_tool_lan_agent_read_text_file : server_tool {
    explicit server_tool_lan_agent_read_text_file(remote_lan_agent_runtime_config cfg)
        : cfg(std::move(cfg)) {
        name = "lan_agent_read_text_file";
        display_name = "LAN agent read text file";
        permission_write = false;
    }
    remote_lan_agent_runtime_config cfg;

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Read a text file through the remote LAN agent MCP paged continuation protocol."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"file_path",  {{"type", "string"}, {"description", "File path to read"}}},
                        {"start_line", {{"type", "integer"}, {"description", "1-based starting line"}}},
                        {"max_lines",  {{"type", "integer"}, {"description", "Maximum number of lines to read"}}},
                        {"trace_id",   {{"type", "string"}, {"description", "Optional trace identifier"}}},
                    }},
                    {"required", json::array({"file_path"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        SRV_INF("lan_agent_read_text_file: delegating to remote LAN agent MCP continuation consumer\n", 0);
        return decorate_remote_lan_agent_result(cfg, consume_remote_lan_agent_continuation(cfg, name, params));
    }
};

//
// lan_agent_read_directory_files: compatibility wrapper for codex-lan-agent bulk file reads
//

struct server_tool_lan_agent_read_directory_files : server_tool {
    explicit server_tool_lan_agent_read_directory_files(remote_lan_agent_runtime_config cfg)
        : cfg(std::move(cfg)) {
        name = "lan_agent_read_directory_files";
        display_name = "LAN agent read directory files";
        permission_write = false;
    }
    remote_lan_agent_runtime_config cfg;

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Read matching files from a directory as one continuous task for codex-lan-agent compatibility."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"directory_path",   {{"type", "string"}, {"description", "Directory to scan"}}},
                        {"include",          {{"type", "string"}, {"description", "Optional include glob"}}},
                        {"include_patterns", {{"type", "array"}, {"description", "Optional include glob patterns"}}},
                        {"exclude",          {{"type", "string"}, {"description", "Optional exclude glob"}}},
                        {"exclude_patterns", {{"type", "array"}, {"description", "Optional exclude glob patterns"}}},
                        {"max_entries",      {{"type", "integer"}, {"description", "Optional maximum number of files to read overall; omit to read all matched files"}}},
                        {"batch_size",       {{"type", "integer"}, {"description", "Deprecated for this compatibility path. Files are processed sequentially one by one; this value is only reported back for visibility."}}},
                    }},
                    {"required", json::array({"directory_path"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        SRV_INF("lan_agent_read_directory_files: delegating to remote LAN agent MCP continuation consumer\n",0);
        return decorate_remote_lan_agent_result(cfg, consume_remote_lan_agent_continuation(cfg, name, params));
    }
};

//
// lan_agent_get_remote_session: compatibility wrapper for codex-lan-agent remote-session lookup
//

struct server_tool_lan_agent_get_remote_session : server_tool {
    explicit server_tool_lan_agent_get_remote_session(remote_lan_agent_runtime_config cfg)
        : cfg(std::move(cfg)) {
        name = "lan_agent_get_remote_session";
        display_name = "LAN agent get remote session";
        permission_write = false;
    }
    remote_lan_agent_runtime_config cfg;

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Load one remote-session conversation by session_id through the remote LAN agent MCP bridge."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"session_id", {{"type", "string"}, {"description", "Remote session identifier"}}},
                    }},
                    {"required", json::array({"session_id"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        SRV_INF("lan_agent_get_remote_session: delegating to remote LAN agent MCP tool bridge\n", 0);
        return decorate_remote_lan_agent_result(cfg, invoke_remote_lan_agent_tool(cfg, name, params));
    }
};

//
// lan_agent_resolve_remote_session_task_refs: compatibility wrapper for 8095 remote-session task ref resolution
//

struct server_tool_lan_agent_resolve_remote_session_task_refs : server_tool {
    explicit server_tool_lan_agent_resolve_remote_session_task_refs(remote_lan_agent_runtime_config cfg)
        : cfg(std::move(cfg)) {
        name = "lan_agent_resolve_remote_session_task_refs";
        display_name = "LAN agent resolve remote session task refs";
        permission_write = false;
    }
    remote_lan_agent_runtime_config cfg;

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Resolve task refs from the 8095 remote-session supervision/task surface before reading files from 18080."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"session_id",    {{"type", "string"}, {"description", "Remote session identifier"}}},
                        {"task_group_id", {{"type", "string"}, {"description", "Optional remote session task group"}}},
                        {"task_id",       {{"type", "string"}, {"description", "Optional task identifier"}}},
                        {"runner",        {{"type", "string"}, {"description", "Optional runner label"}}},
                    }},
                    {"required", json::array({"session_id"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        SRV_INF("lan_agent_resolve_remote_session_task_refs: delegating to remote LAN agent MCP tool bridge\n", 0);
        return decorate_remote_lan_agent_result(cfg, invoke_remote_lan_agent_tool(cfg, name, params));
    }
};

//
// lan_agent_list_remote_session_tasks: compatibility wrapper for 8095 remote-session task listing
//

struct server_tool_lan_agent_list_remote_session_tasks : server_tool {
    explicit server_tool_lan_agent_list_remote_session_tasks(remote_lan_agent_runtime_config cfg)
        : cfg(std::move(cfg)) {
        name = "lan_agent_list_remote_session_tasks";
        display_name = "LAN agent list remote session tasks";
        permission_write = false;
    }
    remote_lan_agent_runtime_config cfg;

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "List task bindings from the 8095 remote-session supervision/task surface for unified remote AI task discovery."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"session_id",    {{"type", "string"}, {"description", "Optional remote session identifier"}}},
                        {"task_group_id", {{"type", "string"}, {"description", "Optional task group identifier"}}},
                        {"runner",        {{"type", "string"}, {"description", "Optional runner label"}}},
                        {"max_entries",   {{"type", "integer"}, {"description", "Optional maximum number of entries"}}},
                    }},
                    {"additionalProperties", false},
                }},
            }},
        };
    }

    json invoke(json params) override {
        SRV_INF("lan_agent_list_remote_session_tasks: delegating to remote LAN agent MCP tool bridge\n", 0);
        return decorate_remote_lan_agent_result(cfg, invoke_remote_lan_agent_tool(cfg, name, params));
    }
};

//
// lan_agent_analyze_dialog_slices: compatibility wrapper for codex-lan-agent dialog slice inspection
//

struct server_tool_lan_agent_analyze_dialog_slices : server_tool {
    explicit server_tool_lan_agent_analyze_dialog_slices(remote_lan_agent_runtime_config cfg)
        : cfg(std::move(cfg)) {
        name = "lan_agent_analyze_dialog_slices";
        display_name = "LAN agent analyze dialog slices";
        permission_write = false;
    }
    remote_lan_agent_runtime_config cfg;

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Inspect stored dialog_slices through the remote LAN agent MCP bridge."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"session_id",  {{"type", "string"}, {"description", "Optional session identifier"}}},
                        {"max_entries", {{"type", "integer"}, {"description", "Optional maximum number of entries"}}},
                    }},
                    {"additionalProperties", false},
                }},
            }},
        };
    }

    json invoke(json params) override {
        SRV_INF("lan_agent_analyze_dialog_slices: delegating to remote LAN agent MCP tool bridge\n", 0);
        return decorate_remote_lan_agent_result(cfg, invoke_remote_lan_agent_tool(cfg, name, params));
    }
};

//
// lan_agent_list_recent_remote_events: compatibility wrapper for codex-lan-agent event feed lookup
//

struct server_tool_lan_agent_list_recent_remote_events : server_tool {
    explicit server_tool_lan_agent_list_recent_remote_events(remote_lan_agent_runtime_config cfg)
        : cfg(std::move(cfg)) {
        name = "lan_agent_list_recent_remote_events";
        display_name = "LAN agent list recent remote events";
        permission_write = false;
    }
    remote_lan_agent_runtime_config cfg;

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Return recent remote control events through the remote LAN agent MCP bridge."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"max_entries",     {{"type", "integer"}, {"description", "Optional maximum number of events"}}},
                        {"include_auto",    {{"type", "boolean"}, {"description", "Whether to include auto events"}}},
                        {"include_noise",   {{"type", "boolean"}, {"description", "Whether to include noise events"}}},
                        {"ai_only",         {{"type", "boolean"}, {"description", "Whether to keep only AI-facing events"}}},
                        {"since_timestamp", {{"type", "string"}, {"description", "Optional lower timestamp bound"}}},
                        {"request_type",    {{"type", "string"}, {"description", "Optional request type filter"}}},
                        {"command_name",    {{"type", "string"}, {"description", "Optional command name filter"}}},
                    }},
                    {"additionalProperties", false},
                }},
            }},
        };
    }

    json invoke(json params) override {
        SRV_INF("lan_agent_list_recent_remote_events: delegating to remote LAN agent MCP tool bridge\n", 0);
        return decorate_remote_lan_agent_result(cfg, invoke_remote_lan_agent_tool(cfg, name, params));
    }
};

//
// lan_agent_get_task: compatibility wrapper for codex-lan-agent queued task inspection
//

struct server_tool_lan_agent_get_task : server_tool {
    explicit server_tool_lan_agent_get_task(remote_lan_agent_runtime_config cfg)
        : cfg(std::move(cfg)) {
        name = "lan_agent_get_task";
        display_name = "LAN agent get task";
        permission_write = false;
    }
    remote_lan_agent_runtime_config cfg;

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Load one queued or completed LAN agent task status and resolved result refs through the remote MCP bridge."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"task_id", {{"type", "string"}, {"description", "LAN agent task identifier"}}},
                    }},
                    {"required", json::array({"task_id"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        SRV_INF("lan_agent_get_task: delegating to remote LAN agent MCP tool bridge\n", 0);
        return decorate_remote_lan_agent_result(cfg, invoke_remote_lan_agent_tool(cfg, name, params));
    }
};

//
// lan_agent_resolve_task_result: compatibility wrapper for task/log ref resolution
//

struct server_tool_lan_agent_resolve_task_result : server_tool {
    explicit server_tool_lan_agent_resolve_task_result(remote_lan_agent_runtime_config cfg)
        : cfg(std::move(cfg)) {
        name = "lan_agent_resolve_task_result";
        display_name = "LAN agent resolve task result";
        permission_write = false;
    }
    remote_lan_agent_runtime_config cfg;

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Resolve one task_id or task-log(...) reference into actual result and evidence refs through the remote MCP bridge."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"task_id",  {{"type", "string"}, {"description", "Optional LAN agent task identifier"}}},
                        {"task_ref", {{"type", "string"}, {"description", "Optional task-log(...) or task:... reference"}}},
                    }},
                    {"additionalProperties", false},
                }},
            }},
        };
    }

    json invoke(json params) override {
        SRV_INF("lan_agent_resolve_task_result: delegating to remote LAN agent MCP tool bridge\n", 0);
        return decorate_remote_lan_agent_result(cfg, invoke_remote_lan_agent_tool(cfg, name, params));
    }
};

//
// public API
//

static std::vector<std::unique_ptr<server_tool>> build_tools(const remote_lan_agent_runtime_config & lan_agent_cfg) {
    std::vector<std::unique_ptr<server_tool>> tools;
    tools.push_back(std::make_unique<server_tool_read_file>());
    tools.push_back(std::make_unique<server_tool_file_info>());
    tools.push_back(std::make_unique<server_tool_file_glob_search>());
    tools.push_back(std::make_unique<server_tool_grep_search>());
    tools.push_back(std::make_unique<server_tool_lan_agent_list_directory>(lan_agent_cfg));
    tools.push_back(std::make_unique<server_tool_lan_agent_read_text_file>(lan_agent_cfg));
    tools.push_back(std::make_unique<server_tool_lan_agent_read_directory_files>(lan_agent_cfg));
    tools.push_back(std::make_unique<server_tool_lan_agent_get_task>(lan_agent_cfg));
    tools.push_back(std::make_unique<server_tool_lan_agent_resolve_task_result>(lan_agent_cfg));
    tools.push_back(std::make_unique<server_tool_lan_agent_list_remote_session_tasks>(lan_agent_cfg));
    tools.push_back(std::make_unique<server_tool_lan_agent_get_remote_session>(lan_agent_cfg));
    tools.push_back(std::make_unique<server_tool_lan_agent_resolve_remote_session_task_refs>(lan_agent_cfg));
    tools.push_back(std::make_unique<server_tool_lan_agent_analyze_dialog_slices>(lan_agent_cfg));
    tools.push_back(std::make_unique<server_tool_lan_agent_list_recent_remote_events>(lan_agent_cfg));
    tools.push_back(std::make_unique<server_tool_exec_shell_command>());
    tools.push_back(std::make_unique<server_tool_write_file>());
    tools.push_back(std::make_unique<server_tool_edit_file>());
    tools.push_back(std::make_unique<server_tool_apply_diff>());
    return tools;
}

static json build_single_tool_state_snapshot(
        const server_tool_call_envelope & delivery,
        const json & result) {
    const bool success = !result.is_object() || !result.contains("error") || result.at("error").is_null();
    return json{
        {"goal_complete", success},
        {"target_count", 1},
        {"completed_count", success ? 1 : 0},
        {"failed_count", success ? 0 : 1},
        {"skipped_count", 0},
        {"pending_count", 0},
        {"next_actions", json::array()},
        {"alarms", success ? json::array() : json::array({build_alarm_event(
            "ERROR",
            "TOOL_RESULT_NOT_VERIFIED",
            "tool execution failed or returned an error payload")})},
        {"failure_mode", success ? "" : "TOOL_RESULT_NOT_VERIFIED"},
        {"goal_closed", true},
        {"final_status_hint", success ? "COMPLETE" : "FAILED"},
    };
}

static json build_single_tool_response(
        const server_goal_envelope & goal,
        const server_tool_call_envelope & delivery,
        const json & result,
        const server_goal_route_decision & pre,
        const server_goal_route_decision & post,
        const server_goal_route_decision & acceptance) {
    const bool success = delivery.delivery_status == "SUCCESS";
    const std::string final_status = !pre.allowed
        ? "NEEDS_HUMAN_APPROVAL"
        : (success && post.verified ? "COMPLETE" : "FAILED");
    const std::string supervision_state = !pre.allowed
        ? "NEEDS_HUMAN_APPROVAL"
        : (success && post.verified ? "COMPLETE" : "FAILED");
    const std::string execution_disposition = !pre.allowed || !post.verified
        ? "RETURN_FAILURE_REPORT"
        : "FINALIZE_ANSWER";
    server_supervision_envelope supervision;
    supervision.supervision_status = supervision_state;
    supervision.execution_disposition = execution_disposition;
    supervision.response_allowed = success && post.verified;
    supervision.failure_mode = success && post.verified ? "" : (!pre.allowed ? pre.failure_mode : post.failure_mode);

    const json alarms = !pre.allowed ? pre.alarms : post.alarms;
    if (alarms.is_array()) {
        for (const auto & alarm : alarms) {
            if (!alarm.is_object()) {
                continue;
            }
            supervision.alarm_code = alarm.value("alarm_code", alarm.value("reason", ""));
            supervision.alarm_message = alarm.value("alarm_message", "");
            if (!supervision.alarm_code.empty() || !supervision.alarm_message.empty()) {
                break;
            }
        }
    }

    return json{
        {"record_model", "tool_delivery_result_v2"},
        {"status", final_status},
        {"goal", to_json(goal)},
        {"delivery", to_json(delivery)},
        {"pre_guard", to_json(pre)},
        {"post_guard", to_json(post)},
        {"acceptance", {
            {"goal_id", goal.goal_id},
            {"acceptance_status", success && post.verified ? "COMPLETE" : "NOT_COMPLETE"},
            {"final_status", final_status},
            {"failure_mode", supervision.failure_mode},
            {"supervision", to_json(supervision)},
            {"supervision_state", supervision_state},
            {"execution_disposition", execution_disposition},
            {"assistant_response_allowed", success && post.verified},
            {"alarm_code", supervision.alarm_code},
            {"alarm_message", supervision.alarm_message},
            {"progress", to_json(acceptance.progress)},
            {"next_actions", acceptance.next_actions},
            {"alarms", alarms},
        }},
        {"supervision", to_json(supervision)},
        {"result", result},
    };
}

static json build_single_tool_compatible_response(
        const server_goal_envelope & goal,
        const server_tool_call_envelope & delivery,
        const json & result,
        const server_goal_route_decision & pre,
        const server_goal_route_decision & post,
        const server_goal_route_decision & acceptance) {
    json response = result.is_object() ? result : json{{"result", result}};
    response["record_model"] = "tool_result_with_envelope_v1";
    response["tool_envelope"] = build_single_tool_response(goal, delivery, result, pre, post, acceptance);
    response["supervision"] = response["tool_envelope"]["supervision"];
    response["delivery"] = response["tool_envelope"]["delivery"];
    response["acceptance"] = response["tool_envelope"]["acceptance"];
    return response;
}

static server_goal_execution_controller::loop_policy build_loop_policy(const json & body) {
    server_goal_execution_controller::loop_policy policy;
    const json & policy_body = body.contains("loop_policy") && body.at("loop_policy").is_object()
        ? body.at("loop_policy")
        : body;

    policy.max_steps = std::max(1, json_value(policy_body, "max_steps", policy.max_steps));
    policy.max_same_action_repeat = std::max(1, json_value(policy_body, "max_same_action_repeat", policy.max_same_action_repeat));
    policy.max_read_files = std::max(1, json_value(policy_body, "max_read_files", policy.max_read_files));
    policy.max_pending_actions_per_round = std::max(1, json_value(policy_body, "max_pending_actions_per_round", policy.max_pending_actions_per_round));
    return policy;
}

void server_tools::setup(const std::vector<std::string> & enabled_tools, const common_params * params) {
    if (!enabled_tools.empty()) {
        std::unordered_set<std::string> enabled_set(enabled_tools.begin(), enabled_tools.end());
        const auto lan_agent_cfg = resolve_remote_lan_agent_runtime_config(params);
        auto all_tools = build_tools(lan_agent_cfg);

        tools.clear();
        for (auto & t : all_tools) {
            if (enabled_set.count(t->name) > 0 || enabled_set.count("all") > 0) {
                tools.push_back(std::move(t));
            }
        }
    }

    handle_get = [this](const server_http_req &) -> server_http_res_ptr {
        auto res = std::make_unique<server_http_res>();
        try {
            json result = json::array();
            for (const auto & t : tools) {
                result.push_back(t->to_json());
            }
            res->data = safe_json_to_str(result);
        } catch (const std::exception & e) {
            SRV_ERR("got exception: %s\n", e.what());
            res->status = 500;
            res->data   = safe_json_to_str(format_error_response(e.what(), ERROR_TYPE_SERVER));
        }
        return res;
    };

    handle_post = [this](const server_http_req & req) -> server_http_res_ptr {
        auto res = std::make_unique<server_http_res>();
        try {
            json body = json::parse(req.body);
            if (body.contains("goal_type") || (body.contains("goal") && body.at("goal").is_object())) {
                server_goal_envelope goal = parse_server_goal_envelope(body);
                const auto policy = build_loop_policy(body);
                server_goal_execution_controller controller(
                    [this](const std::string & name, const json & params) {
                        return invoke(name, params);
                    },
                    [this](const std::string & name) {
                        return is_write_tool(name);
                    });
                json result = controller.execute_goal_until_closed(goal, policy);
                res->data = safe_json_to_str(result);
                return res;
            }

            std::string tool_name = body.at("tool").get<std::string>();
            json params = body.value("params", json::object());
            const std::string trace_id = body.value("trace_id", std::string("trace-") + gen_tool_call_id());
            const std::string request_id = body.value("request_id", std::string("req-") + gen_tool_call_id());
            const std::string goal_id = body.value("goal_id", std::string("goal-tool-") + gen_tool_call_id());

            server_goal_envelope goal;
            goal.goal_id = goal_id;
            goal.request_id = request_id;
            goal.trace_id = trace_id;
            goal.goal_type = "SINGLE_TOOL_CALL";
            goal.status = "RUNNING";

            server_tool_call_envelope delivery;
            delivery.tool_call_id = body.value("tool_call_id", gen_tool_call_id());
            delivery.goal_id = goal_id;
            delivery.request_id = request_id;
            delivery.trace_id = trace_id;
            delivery.tool_name = tool_name;
            delivery.params = params;
            delivery.safety_class = is_write_tool(tool_name) ? "WRITE" : "READ_ONLY";

            server_clips_goal_router router;
            server_trace_registry::record_stage(trace_id, "tool_call_start", json{
                {"goal_id", goal_id},
                {"request_id", request_id},
                {"trace_id", trace_id},
                {"tool_call", to_json(delivery)},
            });

            const bool permission_write = is_write_tool(tool_name);
            const server_goal_route_decision pre = router.pre_guard(goal, delivery, permission_write);
            server_trace_registry::append_event(trace_id, "clips_pre_guard", json{
                {"goal_id", goal_id},
                {"request_id", request_id},
                {"tool_call", to_json(delivery)},
                {"decision", to_json(pre)},
            });

            json result;
            server_goal_route_decision post;
            if (!pre.allowed) {
                result = json{
                    {"error", pre.reason.empty() ? "tool call requires human approval" : pre.reason},
                    {"supervision", {
                        {"blocked", true},
                        {"failure_mode", pre.failure_mode},
                    }},
                };
                delivery.delivery_status = "BLOCKED";
                post.decision_type = "POST_GUARD";
                post.allowed = false;
                post.verified = false;
                post.failure_mode = pre.failure_mode;
                post.reason = pre.reason;
                post.alarms = pre.alarms;
            } else {
                result = invoke(tool_name, params);
                delivery.delivery_status = result.contains("error") ? "FAILED" : "SUCCESS";
                delivery.result_hash = build_server_result_hash(result);
                server_trace_registry::append_event(trace_id, "mcp_tool_result", json{
                    {"goal_id", goal_id},
                    {"request_id", request_id},
                    {"tool_call", to_json(delivery)},
                    {"result", result},
                });
                post = router.post_guard(goal, delivery, result);
            }

            server_trace_registry::append_event(trace_id, "clips_post_guard", json{
                {"goal_id", goal_id},
                {"request_id", request_id},
                {"tool_call", to_json(delivery)},
                {"decision", to_json(post)},
            });

            const server_goal_route_decision acceptance = router.acceptance_check(
                goal,
                build_single_tool_state_snapshot(delivery, result));
            server_trace_registry::append_event(trace_id, "acceptance_check", json{
                {"goal_id", goal_id},
                {"request_id", request_id},
                {"decision", to_json(acceptance)},
            });
            server_trace_registry::record_stage(trace_id, "tool_call_final", json{
                {"goal_id", goal_id},
                {"request_id", request_id},
                {"trace_id", trace_id},
                {"tool_call", to_json(delivery)},
                {"pre_guard", to_json(pre)},
                {"post_guard", to_json(post)},
                {"acceptance", to_json(acceptance)},
                {"result", result},
            });

            if (body.value("return_goal_envelope", false)) {
                json response = build_single_tool_response(goal, delivery, result, pre, post, acceptance);
                res->data = safe_json_to_str(response);
                return res;
            }

            if (!pre.allowed) {
                res->status = 403;
            } else if (result.contains("error")) {
                res->status = 400;
            }
            res->data = safe_json_to_str(
                build_single_tool_compatible_response(goal, delivery, result, pre, post, acceptance));
        } catch (const json::exception & e) {
            res->status = 400;
            res->data   = safe_json_to_str(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        } catch (const std::exception & e) {
            SRV_ERR("got exception: %s\n", e.what());
            res->status = 500;
            res->data   = safe_json_to_str(format_error_response(e.what(), ERROR_TYPE_SERVER));
        }
        return res;
    };
}

json server_tools::invoke(const std::string & name, const json & params) {
    for (auto & t : tools) {
        if (t->name == name) {
            return t->invoke(params);
        }
    }
    return tool_error_json("unknown tool: " + name);
}

bool server_tools::is_write_tool(const std::string & name) const {
    for (const auto & t : tools) {
        if (t->name == name) {
            return t->permission_write;
        }
    }
    return true;
}
