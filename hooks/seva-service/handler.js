import http from "node:http";

function checkStatus(url, timeoutMs = 1500) {
  return new Promise((resolve) => {
    const req = http.get(url, (res) => {
      // just need a response code
      res.resume();
      resolve({ ok: res.statusCode && res.statusCode >= 200 && res.statusCode < 300, status: res.statusCode });
    });
    req.on("error", (err) => resolve({ ok: false, error: String(err) }));
    req.setTimeout(timeoutMs, () => {
      req.destroy(new Error("timeout"));
    });
  });
}

const handler = async (event) => {
  if (event.type !== "gateway" || event.action !== "startup") return;

  // Fire-and-forget so we don't slow startup.
  void (async () => {
    const url = "http://127.0.0.1:18790/status";
    const res = await checkStatus(url);

    if (res.ok) {
      console.log("[clawmemory] SEVA service OK", res.status);
      return;
    }

    console.warn("[clawmemory] SEVA service not reachable.");
    console.warn("[clawmemory] Install/start it via the ClawMemory installer (GitHub: kamb5686/ClawMemory). ");
  })();
};

export default handler;
