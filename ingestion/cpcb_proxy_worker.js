/**
 * cpcb_proxy_worker.js
 * A minimal Cloudflare Worker that relays the project's CPCB snapshot and
 * historical-report requests from the Worker's own egress IP.
 *
 * Why: CPCB's hosts may block or fail to route GitHub Actions runner IP ranges.
 * Cloudflare Workers use a different egress network, so relaying these requests
 * can avoid runner-specific connectivity failures.
 *
 * It only allows GET /resource/* to api.data.gov.in and POST
 * /cpcb-common-report/api/excel/export to airquality.cpcb.gov.in. Each request
 * is forwarded with its method, query, headers, and body intact. Everything
 * else is rejected, so this does not act as an open proxy.
 *
 * Deploy:
 *   1. https://dash.cloudflare.com -> Workers & Pages -> Create -> Create Worker.
 *   2. Paste this file as the Worker's code, replacing the default template.
 *   3. Deploy. Copy the assigned *.workers.dev URL (or a custom route, if you've
 *      set one up) -- that's your CPCB_PROXY_URL.
 *   4. Set CPCB_PROXY_URL to that URL (no trailing slash) as a GitHub Actions
 *      secret / environment variable alongside the other CPCB_* ones.
 *
 * (Equivalently via Wrangler CLI: `wrangler deploy cpcb_proxy_worker.js`, after
 * `wrangler login` and a minimal wrangler.toml naming this file as `main`.)
 *
 * Note: the api-key travels in the query string the same as it always has when
 * calling data.gov.in directly (that's how data.gov.in's API is designed) -- this
 * Worker doesn't add a new place it's exposed, but anyone with your *.workers.dev
 * URL could still read it off an intercepted request the same way they could
 * against the direct URL. If that matters for your key, rotate it periodically
 * via data.gov.in's dashboard, same as you would without a proxy in front.
 */

const SNAPSHOT_UPSTREAM = "https://api.data.gov.in";
const REPORT_UPSTREAM = "https://airquality.cpcb.gov.in";

export default {
  async fetch(request) {
    const incoming = new URL(request.url);
    let upstream;
    if (request.method === "GET" && incoming.pathname.startsWith("/resource/")) {
      upstream = SNAPSHOT_UPSTREAM;
    } else if (request.method === "POST" &&
               incoming.pathname === "/cpcb-common-report/api/excel/export") {
      upstream = REPORT_UPSTREAM;
    } else {
      return new Response("Route or method is not allowed", { status: 405 });
    }
    const upstreamUrl = new URL(incoming.pathname + incoming.search, upstream);

    let upstreamResponse;
    try {
      upstreamResponse = await fetch(new Request(upstreamUrl.toString(), request));
    } catch (err) {
      // Surface a transport-level upstream failure as a 502. The caller keeps
      // its retry policy and logs the specific failure.
      return new Response(`cpcb_proxy_worker: upstream fetch failed: ${err}`, { status: 502 });
    }

    return new Response(upstreamResponse.body, {
      status: upstreamResponse.status,
      headers: {
        "Content-Type": upstreamResponse.headers.get("Content-Type") || "application/json",
      },
    });
  },
};
