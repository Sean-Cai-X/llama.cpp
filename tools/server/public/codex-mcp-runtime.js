(function () {
  "use strict";

  const APPROVAL_KEY = "codex.mcp.requireApproval";
  const AUTO_STACK_KEY = "codex.mcp.autoStack";
  const WEBUI_CONFIG_KEY = "LlamaCppWebui.config";
  const LOW_RISK_COMMANDS = new Set([
    "health",
    "chat-status",
    "task-latest",
    "task",
    "log-latest",
    "diff",
    "read_latest_log",
    "get_git_diff",
    "thread-report",
  ]);

  function getFlag(key, defaultValue) {
    try {
      const value = localStorage.getItem(key);
      return value === null ? defaultValue : value !== "false";
    } catch {
      return defaultValue;
    }
  }

  function setFlag(key, value) {
    try {
      localStorage.setItem(key, value ? "true" : "false");
    } catch {
      // Ignore private-mode storage failures.
    }
  }

  function readWebUiConfig() {
    try {
      const raw = localStorage.getItem(WEBUI_CONFIG_KEY);
      if (!raw) return {};
      const parsed = JSON.parse(raw);
      return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
    } catch {
      return {};
    }
  }

  function writeWebUiConfigValue(key, value) {
    try {
      const current = readWebUiConfig();
      current[key] = value;
      localStorage.setItem(WEBUI_CONFIG_KEY, JSON.stringify(current));
    } catch {
      // Ignore private-mode storage failures.
    }
  }

  function getApprovalRequired() {
    const config = readWebUiConfig();
    if (typeof config.mcpAutoAuthorizeTools === "boolean") {
      return !config.mcpAutoAuthorizeTools;
    }
    return getFlag(APPROVAL_KEY, true);
  }

  function setApprovalRequired(value) {
    setFlag(APPROVAL_KEY, !!value);
    writeWebUiConfigValue("mcpAutoAuthorizeTools", !value);
  }

  function safeJsonParse(value) {
    if (typeof value !== "string" || value.trim() === "") return null;
    try {
      return JSON.parse(value);
    } catch {
      return null;
    }
  }

  function bodyToText(body) {
    if (typeof body === "string") return body;
    if (body instanceof Blob || body instanceof FormData || body instanceof URLSearchParams) return null;
    return null;
  }

  function messageText(content) {
    if (typeof content === "string") return content;
    try {
      return JSON.stringify(content);
    } catch {
      return "";
    }
  }

  function isMcpRichMessage(message) {
    if (!message) return false;
    if (message.tool_calls || message.tool_call_id) return true;
    const text = messageText(message.content);
    return text.length > 1200 && /\b(MCP|tool|resource|prompt|local_cli|codex-lan-agent|evidence|fallback)\b/i.test(text);
  }

  function extractKeyLines(text) {
    const safeText = typeof text === "string" ? text : text == null ? "" : String(text);
    const lines = [];
    const fieldRegex = /"(?:name|uri|tool|command|action_id|ok|error|fallback|evidence|result|log_path)"\s*:\s*(?:"[^"]{0,220}"|true|false|null|\d+)/g;
    for (const match of safeText.matchAll(fieldRegex)) {
      lines.push(match[0]);
      if (lines.length >= 32) break;
    }
    if (lines.length > 0) return lines.join("\n");
    return safeText.slice(0, 2200);
  }

  function compressMcpContent(content, index) {
    const text = messageText(content);
    return [
      `[AUTO-STACKED MCP CONTEXT #${index}]`,
      "This older MCP/tool context was semantically compressed for the outgoing model request only.",
      "Original content remains in the chat history; this hook compresses context, it does not delete it.",
      "",
      "Key information extracted for the model:",
      extractKeyLines(text),
    ].join("\n");
  }

  function compressMcpMessages(messages) {
    if (!Array.isArray(messages) || !getFlag(AUTO_STACK_KEY, true)) return messages;
    let lastRichIndex = -1;
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (isMcpRichMessage(messages[i])) {
        lastRichIndex = i;
        break;
      }
    }
    if (lastRichIndex < 0) return messages;
    return messages.map((message, index) => {
      if (index >= lastRichIndex || !isMcpRichMessage(message)) return message;
      return {
        ...message,
        content: compressMcpContent(message.content, index),
        reasoning_content: undefined,
      };
    });
  }

  function getToolPayload(payload) {
    if (!payload || typeof payload !== "object") return null;
    if (payload.method === "tools/call" && payload.params) {
      return {
        name: payload.params.name,
        args: payload.params.arguments || {},
      };
    }
    if (payload.name && payload.arguments) {
      return {
        name: payload.name,
        args: payload.arguments,
      };
    }
    return null;
  }

  function isHighRiskTool(name, args) {
    const command = String(args && (args.command || args.action_id || args.profile || "")).toLowerCase();
    const text = `${name || ""} ${command} ${JSON.stringify(args || {})}`;
    if (LOW_RISK_COMMANDS.has(command)) return false;
    return /(write|delete|remove|modify|patch|apply|shell|exec|run|build|test|commit|git|upload|download|local_cli|cli|profile)/i.test(text);
  }

  function confirmTool(tool) {
    if (!tool || !getApprovalRequired() || !isHighRiskTool(tool.name, tool.args)) return;
    const text = [
      "MCP high-risk step requires confirmation.",
      "",
      `Tool: ${tool.name}`,
      `Args: ${JSON.stringify(tool.args || {}, null, 2).slice(0, 1400)}`,
      "",
      "Allow this MCP step?",
    ].join("\n");
    if (!window.confirm(text)) {
      throw new Error(`MCP tool execution denied: ${tool.name}`);
    }
  }

  function findMcpSettingsMount() {
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    const anchors = [];
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const text = (node.nodeValue || "").trim();
      if (text === "Agentic loop max turns" || text === "Max lines per tool preview") {
        anchors.push(node.parentElement);
      }
    }
    const anchor = anchors[0];
    if (!anchor) return null;
    let container = anchor;
    for (let i = 0; i < 6 && container && container.parentElement; i += 1) {
      const text = container.textContent || "";
      if (text.includes("Agentic loop max turns") && text.includes("Max lines per tool preview")) {
        return container;
      }
      container = container.parentElement;
    }
    return anchor.parentElement;
  }

  function installToggle() {
    if (document.getElementById("codex-mcp-runtime-panel")) return;
    const mount = findMcpSettingsMount();
    if (!mount) return;
    const panel = document.createElement("div");
    panel.id = "codex-mcp-runtime-panel";
    panel.style.cssText = [
      "margin:18px 0",
      "padding:14px 0",
      "border-top:1px solid rgba(255,255,255,.12)",
      "border-bottom:1px solid rgba(255,255,255,.08)",
      "display:flex",
      "flex-direction:column",
      "gap:12px",
      "font:14px system-ui",
    ].join(";");

    function createRow(title, description, key) {
      const label = document.createElement("label");
      label.style.cssText = "display:flex;gap:12px;align-items:flex-start;cursor:pointer";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.style.cssText = "margin-top:3px;width:18px;height:18px";
      const text = document.createElement("div");
      const heading = document.createElement("div");
      heading.textContent = title;
      heading.style.cssText = "font-weight:600;color:inherit";
      const desc = document.createElement("div");
      desc.textContent = description;
      desc.style.cssText = "margin-top:4px;color:rgba(255,255,255,.68);font-size:13px;line-height:1.4";
      text.append(heading, desc);
      label.append(input, text);
      input.checked = key === APPROVAL_KEY ? getApprovalRequired() : getFlag(key, true);
      input.onchange = () => {
        if (key === APPROVAL_KEY) {
          setApprovalRequired(input.checked);
          return;
        }
        setFlag(key, input.checked);
      };
      return label;
    }

    panel.append(
      createRow(
        "MCP Confirm",
        "Ask before high-risk MCP actions such as run, build, write, upload, download, shell, or local_cli.",
        APPROVAL_KEY,
      ),
      createRow(
        "MCP AutoStack",
        "Automatically compress older MCP/tool context for the outgoing model request without deleting chat history.",
        AUTO_STACK_KEY,
      ),
    );
    mount.appendChild(panel);
  }

  const nativeFetch = window.fetch.bind(window);
  window.fetch = async function codexMcpFetch(input, init) {
    const nextInit = init ? { ...init } : init;
    const bodyText = bodyToText(nextInit && nextInit.body);
    const payload = safeJsonParse(bodyText);

    if (payload && Array.isArray(payload.messages)) {
      payload.messages = compressMcpMessages(payload.messages);
      nextInit.body = JSON.stringify(payload);
    }

    const tool = getToolPayload(payload);
    if (tool) confirmTool(tool);

    return nativeFetch(input, nextInit);
  };

  window.codexMcpRuntime = {
    compressMcpMessages,
    getFlag,
    setFlag,
    confirmTool,
  };

  function scheduleInstall() {
    installToggle();
    const observer = new MutationObserver(() => installToggle());
    observer.observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", scheduleInstall);
  } else {
    scheduleInstall();
  }
})();
