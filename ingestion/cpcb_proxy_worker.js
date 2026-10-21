/**
 * cpcb_proxy_worker.js
 * A minimal Cloudflare Worker that relays requests to api.data.gov.in from the
 * Worker's own egress IP, for use as CPCB_PROXY_URL (see config.py / fetch_cpcb.py).
 *
 * Why: some of data.gov.in's outages/unreachability have been specific to
 * GitHub Actions runner IP ranges rather than a true origin outage (a GH Actions
 * runner uses shared, well-known IP blocks that a government API's edge/WAF may
 * rate-limit or block outright, independent of whether api.data.gov.in itself is
 * up). Cloudflare Workers run on Cloudflare's own edge network, a different IP
 * space, so routing through one sidesteps that specific failure mode without
 * changing the request itself in any way fetch_cpcb.py or data.gov.in would see.
 *
 * What it does: forwards the path and query string unchanged to
 * https://api.data.gov.in<path>?<query> (so fetch_cpcb.py's
 * "/resource/<id>?api-key=...&format=json&..." arrives at data.gov.in exactly as
 * sent) and streams the response straight back, with its status code and
 * Content-Type preserved. Nothing is logged, cached, or modified in either
 * direction -- this is a pass-through, not a cache or a transform.
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

const UPSTREAM = "https://api.data.gov.in";

export default {
  async fetch(request) {
    if (request.method !== "GET") {
      return new Response("Only GET is relayed", { status: 405 });
    }

    const incoming = new URL(request.url);
    const upstreamUrl = new URL(incoming.pathname + incoming.search, UPSTREAM);

    let upstreamResponse;
    try {
      upstreamResponse = await fetch(upstreamUrl.toString(), {
        method: "GET",
        headers: { "Accept": "application/json" },
        // get_with_retry() on the fetch_cpcb.py side already retries; this
        // Worker makes exactly one attempt per request it receives so retry
        // policy stays in one place instead of compounding across two layers.
      });
    } catch (err) {
      // A transport-level failure reaching api.data.gov.in from the Worker --
      // surfaced as a 502 so fetch_cpcb.py's get_with_retry() (which retries
      // 502/503/504/520-524) treats it exactly like it would a direct failure.
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