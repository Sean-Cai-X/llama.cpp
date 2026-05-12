#include "server-embedding-routes.h"

#include "server-task.h"

#include "log.h"

#include <vector>

namespace server_embedding_routes {

std::unique_ptr<server_res_generator> handle_embeddings(
        const server_http_req & req,
        const route_context & ctx,
        task_response_type res_type,
        const response_factory & make_response) {
    auto res = make_response();

    if (!ctx.params.embedding) {
        res->error(format_error_response(
            "This server does not support embeddings. Start it with `--embeddings`",
            ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE &&
        ctx.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response(
            "Pooling type 'none' is not OAI compatible. Please use a different pooling type",
            ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response(
            "\"input\" or \"content\" must be provided",
            ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");

        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response(
                "The format to return the embeddings in. Can be either float or base64",
                ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    auto tokenized_prompts = tokenize_input_prompts(
        ctx.vocab,
        ctx.mctx,
        prompt,
        true,
        true);

    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response(
                "Input content cannot be empty",
                ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = 2; // default to Euclidean/L2 norm
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize");

        if (ctx.pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG(
                "embd_normalize is not supported by pooling type %d, ignoring it\n",
                ctx.pooling_type);
        }
    }

    json responses = json::array();
    auto & rd = res->rd;

    {
        std::vector<server_task> tasks;
        tasks.reserve(tokenized_prompts.size());

        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks));
    }

    auto all_results = rd.wait_for_all(req.should_stop);

    if (all_results.is_terminated) {
        return res; // connection is closed
    }

    if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    }

    for (auto & result : all_results.results) {
        GGML_ASSERT(dynamic_cast<server_task_result_embd *>(result.get()) != nullptr);
        responses.push_back(result->to_json());
    }

    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(
            body,
            ctx.model_name,
            responses,
            use_base64)
        : json(responses);

    res->ok(root);
    return res;
}

std::unique_ptr<server_res_generator> handle_rerank(
        const server_http_req & req,
        const route_context & ctx,
        const response_factory & make_response) {
    auto res = make_response();

    if (!ctx.params.embedding ||
        ctx.params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
        res->error(format_error_response(
            "This server does not support reranking. Start it with `--reranking`",
            ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    const json body = json::parse(req.body);

    // if true, use TEI API format, otherwise use Jina API format
    // Jina: https://jina.ai/reranker/
    // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
    const bool is_tei_format = body.contains("texts");

    json query;
    if (body.count("query") == 1) {
        query = body.at("query");

        if (!query.is_string()) {
            res->error(format_error_response(
                "\"query\" must be a string",
                ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    } else {
        res->error(format_error_response(
            "\"query\" must be provided",
            ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    std::vector<std::string> documents = json_value(
        body,
        "documents",
        json_value(body, "texts", std::vector<std::string>()));

    if (documents.empty()) {
        res->error(format_error_response(
            "\"documents\" must be a non-empty string array",
            ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const int top_n = json_value(body, "top_n", static_cast<int>(documents.size()));

    json responses = json::array();
    auto & rd = res->rd;

    {
        std::vector<server_task> tasks;
        tasks.reserve(documents.size());

        for (size_t i = 0; i < documents.size(); i++) {
            auto tokens = format_prompt_rerank(
                ctx.model,
                ctx.vocab,
                ctx.mctx,
                query,
                documents[i]);

            server_task task(SERVER_TASK_TYPE_RERANK);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokens);

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks));
    }

    auto all_results = rd.wait_for_all(req.should_stop);

    if (all_results.is_terminated) {
        return res; // connection is closed
    }

    if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    }

    for (auto & result : all_results.results) {
        GGML_ASSERT(dynamic_cast<server_task_result_rerank *>(result.get()) != nullptr);
        responses.push_back(result->to_json());
    }

    json root = format_response_rerank(
        body,
        ctx.model_name,
        responses,
        is_tei_format,
        documents,
        top_n);

    res->ok(root);
    return res;
}

} // namespace server_embedding_routes