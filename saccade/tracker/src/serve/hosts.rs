use std::sync::Arc;

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

/// The origin policy. A browser page always names its origin, so every
/// request that carries one must name an allowlisted origin - on POST
/// and GET alike, which closes the cross-site form and no-cors forgery
/// of human-tier writes. An allowlisted read gets its origin echoed
/// back, never a wildcard. A POST also names its body as JSON: the one
/// content type a cross-site form cannot claim. A request with no
/// Origin header never came from a page; the CLI and scripts walk in.
/// The Host names this server answers to, beyond its loopback names:
/// the bind address at the serving port, plus serve-time entries.
#[derive(Clone, Default)]
pub struct Hosts {
    pub bind: Option<String>,
    pub extra: Arc<[String]>,
}

pub(crate) async fn origin_policy(
    axum::Extension(allowed): axum::Extension<Arc<[String]>>,
    axum::Extension(hosts): axum::Extension<Hosts>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let origin = request
        .headers()
        .get(axum::http::header::ORIGIN)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    // the DNS-rebinding guard: a page on an attacker's domain resolves
    // that domain to this server and sends its own name as Host. A
    // rebinding always carries a domain name, so IP literals answer, as
    // do loopback names, the bind address and serve-time entries;
    // another name refuses before routing
    if let Some(host) = request
        .headers()
        .get(axum::http::header::HOST)
        .and_then(|value| value.to_str().ok())
    {
        let name = host
            .rsplit_once(':')
            .filter(|(_, port)| port.chars().all(|c| c.is_ascii_digit()))
            .map(|(before, _)| before)
            .unwrap_or(host);
        let literal = name
            .trim_start_matches('[')
            .trim_end_matches(']')
            .parse::<std::net::IpAddr>()
            .is_ok();
        let loopback = name == "localhost";
        let bound = hosts.bind.as_deref() == Some(host);
        let listed = hosts.extra.iter().any(|allowed| allowed == host);
        if !literal && !loopback && !bound && !listed {
            let mut refusal = (
                StatusCode::FORBIDDEN,
                Json(json!({
                    "error": {
                        "code": "host_forbidden",
                        "detail": format!(
                            "host '{host}' is not this server; call it by an IP address, its loopback name or the bind address, or allow the host at serve time"
                        ),
                    }
                })),
            )
                .into_response();
            // an allowlisted page reads the refusal instead of a blind
            // network failure
            if let Some(origin) = origin.as_deref()
                && allowed.iter().any(|candidate| candidate == origin)
                && let Ok(value) = axum::http::HeaderValue::from_str(origin)
            {
                refusal
                    .headers_mut()
                    .insert("access-control-allow-origin", value);
                refusal
                    .headers_mut()
                    .append("vary", axum::http::HeaderValue::from_static("Origin"));
            }
            return refusal;
        }
    }
    if let Some(origin) = &origin
        && !allowed.iter().any(|candidate| candidate == origin)
    {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({
                "error": {
                    "code": "origin_forbidden",
                    "detail": format!(
                        "origin '{origin}' is not allowed here; the server allows its shells and the origins configured at serve time"
                    ),
                }
            })),
        )
            .into_response();
    }
    if request.method() == axum::http::Method::OPTIONS {
        // a browser preflights every cross-origin write (a POST, PATCH,
        // PUT or DELETE naming Content-Type: application/json is not a
        // simple request): an allowed origin's preflight answers with
        // its own origin echoed and the verbs the doors carry, so the
        // suite's writes can leave the shells. The write boundary is
        // unchanged - any device that reaches the port may write - this
        // only lets the already-allowed pages complete one.
        let allowed_response = |origin: &str| {
            let mut response = (StatusCode::NO_CONTENT, "").into_response();
            if let Ok(value) = axum::http::HeaderValue::from_str(origin) {
                response
                    .headers_mut()
                    .insert("access-control-allow-origin", value);
            }
            response.headers_mut().insert(
                "access-control-allow-methods",
                axum::http::HeaderValue::from_static("GET, POST, PATCH, PUT, DELETE, OPTIONS"),
            );
            response.headers_mut().insert(
                "access-control-allow-headers",
                axum::http::HeaderValue::from_static("Content-Type"),
            );
            response
                .headers_mut()
                .append("vary", axum::http::HeaderValue::from_static("Origin"));
            response
        };
        if let Some(origin) = origin.as_deref() {
            if allowed.iter().any(|candidate| candidate == origin) {
                return allowed_response(origin);
            }
            return (
                StatusCode::FORBIDDEN,
                Json(json!({
                    "error": {
                        "code": "origin_forbidden",
                        "detail": format!(
                            "origin '{origin}' is not allowed here; the server allows its shells and the origins configured at serve time"
                        ),
                    }
                })),
            )
                .into_response();
        }
        // a preflight naming no origin is no browser's: refuse it like
        // any other request
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "code": "bad_request",
                    "detail": "an OPTIONS preflight names its origin; this one named none"
                }
            })),
        )
            .into_response();
    }
    if request.method() == axum::http::Method::POST {
        let json = request
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.split(';').next())
            .map(str::trim)
            .map(|media_type| media_type.eq_ignore_ascii_case("application/json"))
            .unwrap_or(false);
        if !json {
            return (
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                Json(json!({
                    "error": {
                        "code": "content_type_required",
                        "detail": "POST bodies ride Content-Type: application/json; a form or text body is refused, however it arrived"
                    }
                })),
            )
                .into_response();
        }
    }
    let mut response = next.run(request).await;
    if let Some(origin) = &origin
        && allowed.iter().any(|candidate| candidate == origin)
        && let Ok(value) = axum::http::HeaderValue::from_str(origin)
    {
        // the echo rides every answered request, reads and writes
        // alike: a write's answer carries the sequence the client's
        // echo filter and its next settled write need
        response
            .headers_mut()
            .insert("access-control-allow-origin", value);
        response
            .headers_mut()
            .append("vary", axum::http::HeaderValue::from_static("Origin"));
    }
    response
}
