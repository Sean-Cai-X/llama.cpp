#include "server-common.h"

#include <stdexcept>
#include <string>
#include <vector>

json convert_responses_to_chatcmpl(const json & response_body) {
    if (!response_body.contains("input")) {
        throw std::invalid_argument("'input' is required");
    }
    if (!json_value(response_body, "previous_response_id", std::string{}).empty()) {
        throw std::invalid_argument("llama.cpp does not support 'previous_response_id'.");
    }

    const json input_value = response_body.at("input");
    json chatcmpl_body = response_body;
    chatcmpl_body.erase("input");
    std::vector<json> chatcmpl_messages;

    if (response_body.contains("instructions")) {
        chatcmpl_messages.push_back({
            {"role",    "system"},
            {"content", json_value(response_body, "instructions", std::string())},
        });
        chatcmpl_body.erase("instructions");
    }

    if (input_value.is_string()) {
        chatcmpl_messages.push_back({
            {"role",    "user"},
            {"content", input_value},
        });
    } else if (input_value.is_array()) {
        static auto exists_and_is_array = [](const json & j, const char * key) -> bool {
            return j.contains(key) && j.at(key).is_array();
        };
        static auto exists_and_is_string = [](const json & j, const char * key) -> bool {
            return j.contains(key) && j.at(key).is_string();
        };

        for (json item : input_value) {
            bool merge_prev = !chatcmpl_messages.empty() && chatcmpl_messages.back().value("role", "") == "assistant";

            if (exists_and_is_string(item, "content")) {
                item["content"] = json::array({
                    json {
                        {"text", item.at("content")},
                        {"type", "input_text"}
                    }
                });
            }

            if (exists_and_is_array(item, "content") &&
                exists_and_is_string(item, "role") &&
                (item.at("role") == "user" ||
                    item.at("role") == "system" ||
                    item.at("role") == "developer")
            ) {
                std::vector<json> chatcmpl_content;

                for (const json & input_item : item.at("content")) {
                    const std::string type = json_value(input_item, "type", std::string());

                    if (type == "input_text") {
                        if (!input_item.contains("text")) {
                            throw std::invalid_argument("'Input text' requires 'text'");
                        }
                        chatcmpl_content.push_back({
                            {"text", input_item.at("text")},
                            {"type", "text"},
                        });
                    } else if (type == "input_image") {
                        if (!input_item.contains("image_url")) {
                            throw std::invalid_argument("'image_url' is required");
                        }
                        chatcmpl_content.push_back({
                            {"image_url", json {
                                {"url", input_item.at("image_url")}
                            }},
                            {"type", "image_url"},
                        });
                    } else if (type == "input_file") {
                        throw std::invalid_argument("'input_file' is not supported by llamacpp at this moment");
                    } else {
                        throw std::invalid_argument("'type' must be one of 'input_text', 'input_image', or 'input_file'");
                    }
                }

                if (item.contains("type")) {
                    item.erase("type");
                }
                if (item.contains("status")) {
                    item.erase("status");
                }
                item["content"] = chatcmpl_content;

                chatcmpl_messages.push_back(item);
            } else if (exists_and_is_array(item, "content") &&
                exists_and_is_string(item, "role") &&
                item.at("role") == "assistant" &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "message"
            ) {
                auto chatcmpl_content = json::array();

                for (const auto & output_text : item.at("content")) {
                    const std::string type = json_value(output_text, "type", std::string());
                    if (type != "output_text") {
                        throw std::invalid_argument("'type' must be 'output_text'");
                    }
                    if (!exists_and_is_string(output_text, "text")) {
                        throw std::invalid_argument("'Output text' requires 'text'");
                    }
                    chatcmpl_content.push_back({
                        {"text", output_text.at("text")},
                        {"type", "text"},
                    });
                }

                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
                    if (!exists_and_is_array(prev_msg, "content")) {
                        prev_msg["content"] = json::array();
                    }
                    auto & prev_content = prev_msg["content"];
                    prev_content.insert(prev_content.end(), chatcmpl_content.begin(), chatcmpl_content.end());
                } else {
                    item.erase("status");
                    item.erase("type");
                    item["content"] = chatcmpl_content;
                    chatcmpl_messages.push_back(item);
                }
            } else if (exists_and_is_string(item, "arguments") &&
                exists_and_is_string(item, "call_id") &&
                exists_and_is_string(item, "name") &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "function_call"
            ) {
                json tool_call = {
                    {"function", json {
                        {"arguments", item.at("arguments")},
                        {"name",      item.at("name")},
                    }},
                    {"id",   item.at("call_id")},
                    {"type", "function"},
                };

                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
                    if (!exists_and_is_array(prev_msg, "tool_calls")) {
                        prev_msg["tool_calls"] = json::array();
                    }
                    prev_msg["tool_calls"].push_back(tool_call);
                } else {
                    chatcmpl_messages.push_back(json {
                        {"role",       "assistant"},
                        {"tool_calls", json::array({tool_call})}
                    });
                }
            } else if (exists_and_is_string(item, "call_id") &&
                (exists_and_is_string(item, "output") || exists_and_is_array(item, "output")) &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "function_call_output"
            ) {
                if (item.at("output").is_string()) {
                    chatcmpl_messages.push_back(json {
                        {"content",      item.at("output")},
                        {"role",         "tool"},
                        {"tool_call_id", item.at("call_id")},
                    });
                } else {
                    json chatcmpl_outputs = item.at("output");
                    for (json & chatcmpl_output : chatcmpl_outputs) {
                        if (!chatcmpl_output.contains("type") || chatcmpl_output.at("type") != "input_text") {
                            throw std::invalid_argument("Output of tool call should be 'Input text'");
                        }
                        chatcmpl_output["type"] = "text";
                    }
                    chatcmpl_messages.push_back(json {
                        {"content",      chatcmpl_outputs},
                        {"role",         "tool"},
                        {"tool_call_id", item.at("call_id")},
                    });
                }
            } else if (exists_and_is_array(item, "summary") &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "reasoning") {
                if (!exists_and_is_array(item, "content")) {
                    throw std::invalid_argument("item['content'] is not an array");
                }
                if (item.at("content").empty()) {
                    throw std::invalid_argument("item['content'] is empty");
                }
                if (!exists_and_is_string(item.at("content")[0], "text")) {
                    throw std::invalid_argument("item['content']['text'] is not a string");
                }

                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
                    prev_msg["reasoning_content"] = item.at("content")[0].at("text");
                } else {
                    chatcmpl_messages.push_back(json {
                        {"role", "assistant"},
                        {"content", json::array()},
                        {"reasoning_content", item.at("content")[0].at("text")},
                    });
                }
            } else {
                throw std::invalid_argument("Cannot determine type of 'item'");
            }
        }
    } else {
        throw std::invalid_argument("'input' must be a string or array of objects");
    }

    chatcmpl_body["messages"] = chatcmpl_messages;

    if (response_body.contains("tools")) {
        if (!response_body.at("tools").is_array()) {
            throw std::invalid_argument("'tools' must be an array of objects");
        }
        std::vector<json> chatcmpl_tools;
        for (json resp_tool : response_body.at("tools")) {
            json chatcmpl_tool;

            if (json_value(resp_tool, "type", std::string()) != "function") {
                throw std::invalid_argument("'type' of tool must be 'function'");
            }
            resp_tool.erase("type");
            chatcmpl_tool["type"] = "function";

            if (!resp_tool.contains("strict")) {
                resp_tool["strict"] = true;
            }
            chatcmpl_tool["function"] = resp_tool;
            chatcmpl_tools.push_back(chatcmpl_tool);
        }
        chatcmpl_body.erase("tools");
        chatcmpl_body["tools"] = chatcmpl_tools;
    }

    if (response_body.contains("max_output_tokens")) {
        chatcmpl_body.erase("max_output_tokens");
        chatcmpl_body["max_tokens"] = response_body["max_output_tokens"];
    }

    return chatcmpl_body;
}
