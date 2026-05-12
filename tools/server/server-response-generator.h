#pragma once

#include "server-common.h"
#include "server-http.h"
#include "server-queue.h"

struct server_res_generator : server_http_res {
    server_response_reader rd;

    server_res_generator(
            server_queue & queue_tasks,
            server_response & queue_results,
            int http_polling_seconds,
            int sleep_idle_seconds,
            bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, http_polling_seconds) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;

        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }

    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }

    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};