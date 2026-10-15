#pragma once

#include <httplib.h>

namespace moecher::web {

// Registers static web assets (HTML, CSS, JS, Service Worker), /health,
// and the reverse proxy routes for embedded browser previews.
void register_web_routes(httplib::Server& svr);

} // namespace moecher::web
