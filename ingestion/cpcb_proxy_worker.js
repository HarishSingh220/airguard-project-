// cpcb_proxy_worker.js
//
// Minimal Cloudflare Worker that relays GET requests to api.data.gov.in from
// Cloudflare's own IP ranges, to get around api.data.gov.in refusing
// connections from GitHub Actions runner IPs (confirmed: fast TCP connection
// failure, not a slow/overloaded server).
//
// It forwards the path and query string unchanged and passes the upstream
// response straight through, so fetch_cpcb.py needs no changes beyond
// pointing CPCB_PROXY_URL at wherever this Worker ends up deployed.
//
// DEPLOY (free tier, no credit card required):
//   1. https://dash.cloudflare.com -> sign up / log in.
//   2. Workers & Pages -> Create -> Create Worker.
//   3. Give it a name (e.g. "cpcb-proxy") -> Deploy.
//   4. Click "Edit code", replace the default script with this whole file,
//      then "Deploy" again.
//   5. Copy the Worker's URL (shown on its overview page, looks like
//      https://cpcb-proxy.<your-subdomain>.workers.dev).
//   6. In your GitHub repo: Settings -> Secrets and variables -> Actions ->
//      New repository secret. Name: CPCB_PROXY_URL. Value: the URL from
//      step 5 (no trailing slash).
//   7. Make sure ingest.yml passes CPCB_PROXY_URL through to the job's env
//      (see the snippet in that file).
//
// No Cloudflare account secrets are needed inside the Worker itself -- the
// CPCB_API_KEY stays where it already is, passed as a query param by
// fetch_cpcb.py and simply forwarded here.

const UPSTREAM = "https://api.data.gov.in";

export default {
  async fetch(request) {
    if (request.method !== "GET") {
      return new Response("Only GET is relayed", { status: 405 });
    }

    const incoming = new URL(request.url);
    const upstreamUrl = new URL(incoming.pathname + incoming.search, UPSTREAM);

    const upstreamResponse = await fetch(upstreamUrl.toString(), {
      method: "GET",
      headers: { "Accept": "application/json" },
    });

    // Pass the upstream body and status straight through. Strip
    // hop-by-hop / encoding headers that don't make sense to replay verbatim
    // from a second hop; let the Worker runtime set fresh ones.
    const headers = new Headers(upstreamResponse.headers);
    headers.delete("content-encoding");
    headers.delete("content-length");
    headers.delete("transfer-encoding");

    return new Response(upstreamResponse.body, {
      status: upstreamResponse.status,
      headers,
    });
  },
};