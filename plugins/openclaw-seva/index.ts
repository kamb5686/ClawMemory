import os from "node:os";

type SevaPluginConfig = {
  baseUrl?: string;
  timeoutMs?: number;
  defaultRecallK?: number;
  startupCheck?: boolean;
};

function normalizeBaseUrl(raw?: string) {
  const base = (raw || "http://127.0.0.1:18790").trim().replace(/\/+$/, "");
  return base || "http://127.0.0.1:18790";
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number) {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), timeoutMs);
  const wrapped = (async () => {
    try {
      // @ts-ignore - consumer supplies fetch using controller.signal
      return await promise;
    } finally {
      clearTimeout(t);
    }
  })();
  return { controller, wrapped };
}

async function fetchJson(params: {
  baseUrl: string;
  path: string;
  method?: string;
  body?: any;
  timeoutMs: number;
}) {
  const url = `${params.baseUrl}${params.path.startsWith("/") ? "" : "/"}${params.path}`;

  const init: RequestInit = {
    method: params.method ?? (params.body ? "POST" : "GET"),
    headers: params.body ? { "content-type": "application/json" } : undefined,
    body: params.body ? JSON.stringify(params.body) : undefined,
  };

  const controller = new AbortController();
  init.signal = controller.signal;

  const timer = setTimeout(() => controller.abort(), params.timeoutMs);
  try {
    const res = await fetch(url, init);
    const text = await res.text();
    let data: any = null;
    try {
      data = text ? JSON.parse(text) : null;
    } catch {
      data = { raw: text };
    }

    if (!res.ok) {
      const msg = data?.error ? String(data.error) : `${res.status} ${res.statusText}`;
      const err: any = new Error(msg);
      err.status = res.status;
      err.data = data;
      throw err;
    }

    return data;
  } finally {
    clearTimeout(timer);
  }
}

function formatJsonPreview(obj: any, max = 1400) {
  const raw = JSON.stringify(obj, null, 2);
  if (raw.length <= max) return raw;
  return raw.slice(0, max) + "\n…";
}

function helpText() {
  return [
    "SEVA commands:",
    "- /seva status",
    "- /seva on | off",
    "- /seva mode                (show current)",
    "- /seva mode <name>         (set mode)",
    "- /seva mode list           (best-effort list)",
    "- /seva recall [k] <query>",
    "- /seva memory status",
    "- /seva memory prune [--dry-run] [--max N] [--policy oldest|least_reinforced]",
    "- /seva verify [--provider <name>] [--all] <claim>",
    "- /seva wolfram status|on|off|set <appid>",
    "- /seva doctor",
    "",
    "Notes:",
    "- SEVA runs locally (default http://127.0.0.1:18790).",
    "- This command bypasses the LLM agent.",
  ].join("\n");
}

function summarizeStatus(status: any) {
  const lines: string[] = [];
  lines.push(`enabled: ${status?.enabled}`);
  lines.push(`intercept: ${status?.intercept}`);
  if (status?.memory?.episodic) {
    lines.push(`memory.episodic: ${status.memory.episodic.enabled} (episodes: ${status.memory.episodic.episodes})`);
  }
  if (status?.memory?.semantic) {
    lines.push(`memory.semantic: ${status.memory.semantic.enabled} (items: ${status.memory.semantic.items})`);
  }
  if (status?.verification) {
    lines.push(`verification: enabled=${status.verification.enabled} wikipedia=${status.verification.wikipedia}`);
  }
  if (status?.scoring) {
    lines.push(`scoring: enabled=${status.scoring.enabled}`);
  }
  if (status?.paths) {
    lines.push(`config: ${status.paths.config}`);
    lines.push(`data:   ${status.paths.data_dir}`);
  }
  if (status?.deps) {
    lines.push(`deps: titanmind=${status.deps.titanmind} wikipedia_checker=${status.deps.wikipedia_checker}`);
  }
  return lines.join("\n");
}

export default function register(api: any) {
  const cfg = (api.pluginConfig ?? {}) as SevaPluginConfig;
  const baseUrl = normalizeBaseUrl(cfg.baseUrl);
  const timeoutMs = Math.max(200, Math.min(30000, cfg.timeoutMs ?? 2500));
  const defaultRecallK = Math.max(1, Math.min(50, cfg.defaultRecallK ?? 5));

  const doStatus = async () => {
    return await fetchJson({ baseUrl, path: "/status", timeoutMs });
  };

  if (cfg.startupCheck !== false) {
    api.registerHook(
      "gateway:startup",
      async () => {
        // Fire-and-forget; don't block gateway startup.
        void (async () => {
          try {
            await doStatus();
            api.logger.info(`[clawmemory-seva] SEVA reachable at ${baseUrl}`);
          } catch (err: any) {
            api.logger.warn(`[clawmemory-seva] SEVA not reachable at ${baseUrl} (${String(err?.message ?? err)})`);
            api.logger.warn(`[clawmemory-seva] Run /seva doctor for troubleshooting.`);
          }
        })();
      },
      { name: "clawmemory-seva-startup" }
    );
  }

  api.registerCommand({
    name: "seva",
    description: "Control the local SEVA service (status/mode/recall/verify/on/off/doctor).",
    acceptsArgs: true,
    requireAuth: true,
    handler: async (ctx: any) => {
      const raw = (ctx.args || "").trim();
      if (!raw) {
        return { text: helpText() };
      }

      const [sub, ...restTokens] = raw.split(/\s+/).filter(Boolean);
      const subcmd = (sub || "").toLowerCase();
      const rest = restTokens.join(" ").trim();

      try {
        if (subcmd === "help") {
          return { text: helpText() };
        }

        if (subcmd === "status") {
          const status = await doStatus();
          return { text: summarizeStatus(status) };
        }

        if (subcmd === "doctor") {
          const lines: string[] = [];
          lines.push(`baseUrl: ${baseUrl}`);
          lines.push(`host: ${os.hostname()} (${process.platform})`);

          try {
            const status = await doStatus();
            lines.push("\n/status OK\n" + summarizeStatus(status));
          } catch (e: any) {
            lines.push(`\n/status FAILED: ${String(e?.message ?? e)}`);
          }

          try {
            const config = await fetchJson({ baseUrl, path: "/config", timeoutMs });
            const mode = config?.runtime?.mode;
            lines.push(`\n/config OK (runtime.mode=${mode ?? "<unset>"})`);
          } catch (e: any) {
            lines.push(`\n/config FAILED: ${String(e?.message ?? e)}`);
          }

          lines.push("\nIf SEVA is not running:");
          if (process.platform === "linux") {
            lines.push("- systemd: systemctl status openclaw-seva ; systemctl --user status openclaw-seva");
          } else if (process.platform === "darwin") {
            lines.push("- launchd: launchctl list | grep -i seva");
          }
          lines.push("- confirm port: lsof -iTCP:18790 -sTCP:LISTEN (macOS/Linux)");

          return { text: lines.join("\n") };
        }

        if (subcmd === "on" || subcmd === "off") {
          const enabled = subcmd === "on";
          const config = await fetchJson({
            baseUrl,
            path: "/config-set",
            method: "POST",
            body: { set: [`enabled=${enabled}`] },
            timeoutMs,
          });
          return { text: `SEVA enabled=${enabled}\n\n${formatJsonPreview(config, 900)}` };
        }

        if (subcmd === "mode") {
          const arg = rest;
          if (!arg) {
            // Show current mode (best-effort)
            const config = await fetchJson({ baseUrl, path: "/config", timeoutMs });
            const mode = config?.runtime?.mode ?? "<unset>";
            return { text: `Current SEVA mode: ${mode}\n\nUse: /seva mode <name> or /seva mode list` };
          }
          if (arg.toLowerCase() === "list") {
            // Best-effort: provoke an error that includes available modes.
            try {
              await fetchJson({ baseUrl, path: "/mode-set", method: "POST", body: { mode: "__list__" }, timeoutMs });
            } catch (e: any) {
              const modes = e?.data?.modes;
              if (Array.isArray(modes) && modes.length) {
                return { text: `Available modes:\n- ${modes.join("\n- ")}` };
              }
              return { text: `Mode list unavailable. Error: ${String(e?.message ?? e)}` };
            }
            return { text: "Mode list unavailable." };
          }

          const result = await fetchJson({
            baseUrl,
            path: "/mode-set",
            method: "POST",
            body: { mode: arg },
            timeoutMs,
          });
          if (result?.success) {
            return { text: `SEVA mode set: ${result.mode}` };
          }
          return { text: formatJsonPreview(result) };
        }

        if (subcmd === "wolfram") {
          const action = String(restTokens[0] ?? "status").toLowerCase();

          const readWolframStatus = async () => {
            const config = await fetchJson({ baseUrl, path: "/config", timeoutMs });
            const w = config?.verification?.wolfram ?? {};
            const enabled = Boolean(w?.enabled);
            const appid = String(w?.appid ?? "").trim();
            const hasAppid = appid.length > 0;
            const masked = hasAppid ? `${appid.slice(0, 4)}…${appid.slice(-3)}` : "<unset>";
            return { enabled, hasAppid, masked };
          };

          if (action === "status") {
            const s = await readWolframStatus();
            return { text: `wolfram.enabled: ${s.enabled}\nwolfram.appid: ${s.masked}` };
          }

          if (action === "on") {
            const s = await readWolframStatus();
            if (!s.hasAppid) {
              return { text: "Wolfram is missing an AppID. Set it first:\n/seva wolfram set <appid>" };
            }
            const config = await fetchJson({
              baseUrl,
              path: "/config-set",
              method: "POST",
              body: { set: ["verification.wolfram.enabled=true"] },
              timeoutMs,
            });
            return { text: `Wolfram enabled.\n\n${formatJsonPreview(config, 900)}` };
          }

          if (action === "off") {
            const config = await fetchJson({
              baseUrl,
              path: "/config-set",
              method: "POST",
              body: { set: ["verification.wolfram.enabled=false"] },
              timeoutMs,
            });
            return { text: `Wolfram disabled.\n\n${formatJsonPreview(config, 900)}` };
          }

          if (action === "set") {
            const appid = restTokens.slice(1).join(" ").trim();
            if (!appid) {
              return { text: "Usage: /seva wolfram set <appid>" };
            }
            const config = await fetchJson({
              baseUrl,
              path: "/config-set",
              method: "POST",
              body: { set: [`verification.wolfram.appid=${appid}`, "verification.wolfram.enabled=true"] },
              timeoutMs,
            });
            return { text: `Wolfram AppID set and enabled.\n\n${formatJsonPreview(config, 900)}` };
          }

          return { text: "Usage: /seva wolfram status|on|off|set <appid>" };
        }

        if (subcmd === "recall") {
          if (!rest) {
            return { text: "Usage: /seva recall [k] <query>" };
          }

          // Allow: /seva recall 8 some query
          const maybeK = Number(restTokens[0]);
          const hasK = Number.isFinite(maybeK) && maybeK > 0 && String(restTokens[0]).match(/^\d+$/);
          const k = hasK ? Math.max(1, Math.min(50, maybeK)) : defaultRecallK;
          const query = (hasK ? restTokens.slice(1).join(" ") : rest).trim();
          if (!query) {
            return { text: "Usage: /seva recall [k] <query>" };
          }

          const data = await fetchJson({
            baseUrl,
            path: "/recall",
            method: "POST",
            body: { query, k },
            timeoutMs,
          });

          const results: any[] = Array.isArray(data?.results) ? data.results : [];
          const lines: string[] = [];
          lines.push(`query: ${query}`);
          lines.push(`k: ${k}`);
          lines.push("");
          if (!results.length) {
            lines.push("(no results)");
            return { text: lines.join("\n") };
          }
          results.slice(0, k).forEach((r, i) => {
            const score = r?.score == null ? "" : ` score=${Number(r.score).toFixed(3)}`;
            const src = r?.source ? `(${r.source}${score})` : "";
            const text = String(r?.text ?? "").replace(/\s+/g, " ").trim();
            lines.push(`${i + 1}. ${src} ${text}`.trim());
          });
          return { text: lines.join("\n") };
        }

        if (subcmd === "memory") {
          const action = String(restTokens[0] ?? "status").toLowerCase();

          if (action === "status") {
            const data = await fetchJson({ baseUrl, path: "/memory/status", timeoutMs });
            return { text: formatJsonPreview(data, 1600) };
          }

          if (action === "prune") {
            // flags: --dry-run, --max N, --policy name
            const tokens = restTokens.slice(1);
            let dryRun = false;
            let maxItems: number | undefined;
            let policy: string | undefined;

            for (let i = 0; i < tokens.length; i++) {
              const tok = tokens[i];
              if (tok === "--dry-run") {
                dryRun = true;
                continue;
              }
              if (tok === "--max" && i + 1 < tokens.length) {
                const n = Number(tokens[i + 1]);
                if (Number.isFinite(n) && n >= 0) maxItems = Math.floor(n);
                i += 1;
                continue;
              }
              if (tok === "--policy" && i + 1 < tokens.length) {
                policy = tokens[i + 1];
                i += 1;
                continue;
              }
            }

            const body: any = { dry_run: dryRun };
            if (maxItems != null) body.max_items = maxItems;
            if (policy) body.policy = policy;

            const data = await fetchJson({
              baseUrl,
              path: "/memory/prune",
              method: "POST",
              body,
              timeoutMs: Math.max(timeoutMs, 8000),
            });
            return { text: formatJsonPreview(data, 1600) };
          }

          return { text: "Usage: /seva memory status | prune [--dry-run] [--max N] [--policy oldest|least_reinforced]" };
        }

        if (subcmd === "verify") {
          if (!rest) {
            return { text: "Usage: /seva verify [--provider <name>] [--all] <claim>" };
          }

          // flags: --provider <name>, --all
          const tokens = restTokens.slice();
          let provider = null;
          let all = false;
          const claimParts = [];
          for (let i = 0; i < tokens.length; i++) {
            const tok = tokens[i];
            if (tok === "--all") {
              all = true;
              continue;
            }
            if (tok === "--provider" && i + 1 < tokens.length) {
              provider = tokens[i + 1];
              i += 1;
              continue;
            }
            claimParts.push(tok);
          }
          const claim = claimParts.join(" ").trim();
          if (!claim) return { text: "Usage: /seva verify [--provider <name>] [--all] <claim>" };

          const body = { claim };
          if (all) body.all = true;
          if (provider) body.providers = [provider];

          const data = await fetchJson({
            baseUrl,
            path: "/verify",
            method: "POST",
            body,
            timeoutMs: Math.max(timeoutMs, 8000),
          });

          return { text: formatJsonPreview(data, 1800) };
        }

        return { text: `Unknown subcommand: ${subcmd}\n\n${helpText()}` };
      } catch (e: any) {
        const detail = e?.data ? `\n\n${formatJsonPreview(e.data, 1200)}` : "";
        return { text: `SEVA request failed: ${String(e?.message ?? e)}${detail}` };
      }
    },
  });
}
