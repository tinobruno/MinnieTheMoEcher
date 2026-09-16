#!/usr/bin/env node

/**
 * teams-puppeteer-mcp
 * High-performance Model Context Protocol (MCP) server for direct Microsoft Teams chat automation,
 * message extraction, and unread monitoring using Puppeteer.
 *
 * Optimized for ultra-low token usage and zero multi-turn reasoning overhead:
 * - Returns lean, human/LLM-readable text directly without JSON bloat or duplication.
 * - get_last_message: Returns ONLY the single latest message.
 * - get_last_unread_message: Returns unread message for a specific chat OR across all chats.
 * - get_last_messages: Returns last N messages when history is requested.
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import puppeteer from "puppeteer";
import fs from "fs";
import path from "path";
import http from "http";

// Server instance
const server = new Server(
  {
    name: "teams-puppeteer-mcp",
    version: "1.1.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

let cachedBrowser = null;
let cachedPage = null;

/**
 * Check if an HTTP port responds (for Chrome DevTools)
 */
function checkHttpPort(port) {
  return new Promise((resolve) => {
    const req = http.get(`http://127.0.0.1:${port}/json/version`, (res) => {
      resolve(res.statusCode === 200);
    });
    req.on("error", () => resolve(false));
    req.setTimeout(500, () => {
      req.destroy();
      resolve(false);
    });
  });
}

/**
 * Search for an existing running Chrome instance with DevTools active
 */
async function findActiveDevToolsPort() {
  if (process.env.DEVTOOLS_PORT) {
    const p = parseInt(process.env.DEVTOOLS_PORT, 10);
    if (await checkHttpPort(p)) return p;
  }
  if (process.env.CHROME_REMOTE_DEBUGGING_PORT) {
    const p = parseInt(process.env.CHROME_REMOTE_DEBUGGING_PORT, 10);
    if (await checkHttpPort(p)) return p;
  }

  // Check /tmp for any active puppeteer/chrome DevToolsActivePort files
  try {
    const tmpDirs = fs
      .readdirSync("/tmp")
      .filter((d) => d.startsWith("puppeteer_dev_chrome_profile") || d.startsWith("chrome_dev_"));
    for (const d of tmpDirs) {
      const portFile = path.join("/tmp", d, "DevToolsActivePort");
      if (fs.existsSync(portFile)) {
        const raw = fs.readFileSync(portFile, "utf8").trim().split("\n")[0];
        const port = parseInt(raw, 10);
        if (port && !isNaN(port) && (await checkHttpPort(port))) {
          return port;
        }
      }
    }
  } catch (_) {}

  // Check standard 9222
  if (await checkHttpPort(9222)) return 9222;

  return null;
}

/**
 * Find Chrome / Chromium executable path on the system
 */
function findSystemChrome() {
  if (process.env.PUPPETEER_EXECUTABLE_PATH && fs.existsSync(process.env.PUPPETEER_EXECUTABLE_PATH)) {
    return process.env.PUPPETEER_EXECUTABLE_PATH;
  }
  const candidates = [
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/opt/google/chrome/chrome",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  return null;
}

/**
 * Connect to an active Teams tab or launch a browser session
 */
async function getTeamsPage() {
  if (cachedPage && !cachedPage.isClosed() && cachedBrowser && cachedBrowser.connected) {
    return cachedPage;
  }

  const activePort = await findActiveDevToolsPort();
  if (activePort) {
    try {
      cachedBrowser = await puppeteer.connect({
        browserURL: `http://127.0.0.1:${activePort}`,
      });
      const pages = await cachedBrowser.pages();
      let teamsPage = pages.find((p) => {
        const u = p.url();
        return u.includes("teams.microsoft.com") || u.includes("teams.cloud.microsoft");
      });

      if (teamsPage) {
        cachedPage = teamsPage;
        return cachedPage;
      } else if (pages.length > 0) {
        cachedPage = pages[0];
        await cachedPage.goto("https://teams.cloud.microsoft/", {
          waitUntil: "domcontentloaded",
        });
        return cachedPage;
      }
    } catch (err) {
      console.error(`[teams-mcp] Failed to connect to port ${activePort}: ${err.message}`);
      cachedBrowser = null;
    }
  }

  const homeDir = process.env.HOME || process.env.USERPROFILE || "/tmp";
  const profileDir =
    process.env.TEAMS_PROFILE_DIR || path.join(homeDir, ".teams_puppeteer_profile");
  if (!fs.existsSync(profileDir)) {
    fs.mkdirSync(profileDir, { recursive: true });
  }

  const chromePath = findSystemChrome();
  const headless = process.env.TEAMS_HEADLESS === "true";

  cachedBrowser = await puppeteer.launch({
    headless: headless ? "new" : false,
    executablePath: chromePath || undefined,
    userDataDir: profileDir,
    args: [
      "--no-sandbox",
      "--disable-dev-shm-usage",
      "--disable-background-timer-throttling",
      "--disable-backgrounding-occluded-windows",
      "--disable-renderer-backgrounding",
      "--remote-debugging-port=9222",
    ],
  });

  const pages = await cachedBrowser.pages();
  cachedPage = pages[0] || (await cachedBrowser.newPage());
  if (!cachedPage.url().includes("teams.cloud.microsoft") && !cachedPage.url().includes("teams.microsoft.com")) {
    await cachedPage.goto("https://teams.cloud.microsoft/", {
      waitUntil: "domcontentloaded",
    });
  }

  // Ensure Teams sidebar or chat elements have rendered
  await cachedPage
    .waitForSelector('[role="treeitem"], [data-item-type="chat"], [data-tid="chat-pane-header"], .fui-Chat', {
      timeout: 15000,
    })
    .catch(() => {});

  return cachedPage;
}

// ─────────────────────────────────────────────────────────────────────────────
// Core In-Browser Helper: Switch Chat & Extract Messages
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Execute DOM actions in the Teams page:
 * Switches to target chat if needed, scrolls to bottom, and extracts messages.
 */
async function fetchTeamsMessagesInBrowser(page, targetChat, maxLimit, onlyUnread) {
  return await page.evaluate(
    async (targetChat, maxLimit, onlyUnread) => {
      function simulateUserClick(element) {
        if (!element) return false;
        const rect = element.getBoundingClientRect();
        const x = rect.left + rect.width / 2;
        const y = rect.top + rect.height / 2;

        element.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, cancelable: true, clientX: x, clientY: y, pointerType: 'mouse' }));
        element.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, cancelable: true, clientX: x, clientY: y, button: 0 }));
        element.focus();
        element.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, cancelable: true, clientX: x, clientY: y, pointerType: 'mouse' }));
        element.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, cancelable: true, clientX: x, clientY: y, button: 0 }));
        element.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, clientX: x, clientY: y, button: 0 }));
        return true;
      }

      // 1. Switch chat if targetChat is specified and not already active
      if (targetChat && targetChat.trim().length > 0) {
        const targetClean = targetChat.trim().toLowerCase();
        const activeHeader =
          document.querySelector('[data-tid="chat-pane-header"], [data-tid="chat-title"], h2')?.innerText || "";

        if (!activeHeader.toLowerCase().includes(targetClean)) {
          const findChat = () => {
            const all = Array.from(document.querySelectorAll('[data-item-type="chat"], [role="treeitem"]'));
            const candidates = all.filter((el) => {
              const isFolder =
                el.getAttribute("data-conversation-folder") === "true" ||
                el.getAttribute("data-item-type") === "custom-folder" ||
                el.getAttribute("data-item-type") === "chats" ||
                el.getAttribute("data-item-type") === "favorites";
              return !isFolder;
            });

            // 1. First line match (exact or substring)
            const titleMatch = candidates.find((el) => {
              const firstLine = (el.innerText || "").split("\n")[0].toLowerCase().trim();
              return firstLine.includes(targetClean) || targetClean.includes(firstLine);
            });
            if (titleMatch) return titleMatch;

            // 2. Fallback to relaxed match ignoring punctuation / parentheses
            const cleanAlpha = (s) => s.replace(/[^a-z0-9]/g, "");
            const alphaTarget = cleanAlpha(targetClean);
            const relaxedMatch = candidates.find((el) => {
              const firstLine = cleanAlpha((el.innerText || "").split("\n")[0].toLowerCase());
              return firstLine.includes(alphaTarget) || alphaTarget.includes(firstLine);
            });
            if (relaxedMatch) return relaxedMatch;

            // 3. Match anywhere in element text
            return candidates.find((el) =>
              (el.innerText || "").toLowerCase().includes(targetClean)
            );
          };

          let match = findChat();

          // If not visible in tree, expand collapsed folders (e.g. Favorites, Chats)
          if (!match) {
            const folders = Array.from(document.querySelectorAll('[aria-expanded="false"]'));
            for (const f of folders) {
              const icon = f.querySelector('.fui-TreeItemLayout__expandIcon');
              if (icon) icon.click();
              else f.click();
            }
            await new Promise((r) => setTimeout(r, 350));
            match = findChat();
          }

          if (match) {
            const layout = match.querySelector('.fui-TreeItemLayout') || match;
            simulateUserClick(layout);
            // Wait for chat pane DOM transition
            await new Promise((r) => setTimeout(r, 450));
          } else {
            return {
              error: `Chat matching "${targetChat}" was not found in Teams sidebar.`,
            };
          }
        }
      }

      // 2. Ensure chat container is scrolled to the latest messages
      const scrollContainer = document.querySelector(
        '[data-tid="chat-pane-list"], .fui-Chat, [role="log"], [data-tid="chat-list"]'
      );
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }

      // 3. Current active chat name
      const headerTitle =
        document.querySelector('[data-tid="chat-pane-header"], [data-tid="chat-title"], h2')?.innerText?.split("\n")[0]?.trim() || targetChat || "Active Chat";

      // 4. Query all message elements in chronological order
      const messageNodes = Array.from(
        document.querySelectorAll(".fui-ChatMessage, .fui-ChatMyMessage")
      );

      // 5. Look for unread indicator / divider
      let unreadStartIndex = -1;
      const dividers = Array.from(
        document.querySelectorAll(".fui-Divider, [role=\"separator\"], [data-tid*=\"unread\"]")
      );
      const unreadDivider = dividers.find((d) => {
        const txt = (d.innerText || "").toLowerCase();
        return txt.includes("unread") || txt.includes("new message") || txt.includes("last read");
      });

      if (unreadDivider) {
        for (let i = 0; i < messageNodes.length; i++) {
          if (
            unreadDivider.compareDocumentPosition(messageNodes[i]) &
            Node.DOCUMENT_POSITION_FOLLOWING
          ) {
            unreadStartIndex = i;
            break;
          }
        }
      }

      let selectedNodes = [];
      if (onlyUnread) {
        if (unreadStartIndex !== -1) {
          selectedNodes = messageNodes.slice(unreadStartIndex);
        } else {
          selectedNodes = messageNodes.filter((m) => {
            const aria = (m.getAttribute("aria-label") || "").toLowerCase();
            const cls = (m.className || "").toLowerCase();
            return aria.includes("unread") || cls.includes("unread");
          });
        }
      } else {
        selectedNodes = messageNodes.slice(-maxLimit);
      }

      // 6. Extract clean message objects
      const messages = selectedNodes.map((c) => {
        const author =
          c.querySelector('[class*="author"]')?.innerText?.trim() || "Unknown";
        const timestamp =
          c.querySelector('[class*="timestamp"]')?.innerText?.trim() || "";
        const body =
          c.querySelector('[class*="body"]')?.innerText?.trim() || "";
        const isUnread =
          unreadStartIndex !== -1 &&
          messageNodes.indexOf(c) >= unreadStartIndex;

        const isMyMessage =
          c.classList.contains("fui-ChatMyMessage") ||
          author.toLowerCase().includes("(you)");

        return {
          author,
          timestamp,
          body,
          is_unread: isUnread,
          is_my_message: isMyMessage,
        };
      }).filter((m) => m.body.length > 0);

      return {
        chat: headerTitle,
        unread_divider_detected: unreadStartIndex !== -1,
        messages,
      };
    },
    targetChat,
    maxLimit,
    onlyUnread
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tool Handlers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 1. get_last_message
 * Returns the latest message from the contact, plus the latest overall message if sent by user.
 */
async function handleGetLastMessage(args) {
  const chatName = args?.chat_name?.trim();
  if (!chatName) {
    throw new Error("'chat_name' is required (e.g. 'CAN Ercan', 'Bruno', 'Alex').");
  }

  const page = await getTeamsPage();
  const res = await fetchTeamsMessagesInBrowser(page, chatName, 10, false);

  if (res.error) {
    return { content: [{ type: "text", text: `[Teams]: ${res.error}` }] };
  }

  if (!res.messages || res.messages.length === 0) {
    return { content: [{ type: "text", text: `Chat "${res.chat}": No messages found.` }] };
  }

  const all = res.messages;
  const lastOverall = all[all.length - 1];
  const lastFromContact = all.filter((m) => !m.is_my_message).pop();

  let output = `Chat: ${res.chat}\n`;
  if (lastOverall.is_my_message && lastFromContact) {
    output += `• Last message received from ${lastFromContact.author} (${lastFromContact.timestamp}): "${lastFromContact.body}"\n`;
    output += `• Latest message in chat (sent by you at ${lastOverall.timestamp}): "${lastOverall.body}"`;
  } else {
    output += `[${lastOverall.author} | ${lastOverall.timestamp}]: ${lastOverall.body}`;
  }

  return {
    content: [{ type: "text", text: output.trim() }],
  };
}

/**
 * 2. get_last_unread_message
 * Returns the latest unread message for a specific chat, OR scans all unread chats across Teams if chat_name is omitted.
 */
async function handleGetLastUnreadMessage(args) {
  const chatName = args?.chat_name?.trim() || "";
  const page = await getTeamsPage();

  if (chatName.length > 0) {
    // Specific chat
    const res = await fetchTeamsMessagesInBrowser(page, chatName, 1, true);

    if (res.error) {
      return { content: [{ type: "text", text: `[Teams]: ${res.error}` }] };
    }

    if (!res.messages || res.messages.length === 0) {
      return { content: [{ type: "text", text: `Chat "${res.chat}": No unread messages.` }] };
    }

    const last = res.messages[res.messages.length - 1];
    const output = `Chat: ${res.chat} [UNREAD]\n[${last.author} | ${last.timestamp}]: ${last.body}`;
    return { content: [{ type: "text", text: output }] };
  } else {
    // Across ALL chats
    const unreadChats = await page.evaluate(() => {
      const items = Array.from(document.querySelectorAll('[data-item-type="chat"], [role="treeitem"]'));
      const unreadList = [];
      const seen = new Set();

      for (const item of items) {
        if (
          item.getAttribute("data-conversation-folder") === "true" ||
          item.getAttribute("data-item-type") === "custom-folder"
        ) {
          continue;
        }

        const lines = (item.innerText || "").split("\n").map((l) => l.trim()).filter(Boolean);
        if (lines.length === 0) continue;
        const name = lines[0];
        if (seen.has(name)) continue;

        const hasBadge =
          item.querySelector('[class*="badge"], [aria-label*="unread" i], [data-tid*="unread" i]') !== null ||
          (item.getAttribute("aria-label") || "").toLowerCase().includes("unread");
        const isBold =
          window.getComputedStyle(item).fontWeight >= 600 ||
          item.querySelector("strong, b") !== null;

        if (hasBadge || isBold) {
          seen.add(name);
          const time = lines.length > 1 ? lines[1] : "";
          const preview = lines.length > 2 ? lines.slice(2).join(" ") : "";
          unreadList.push({ name, time, preview: preview.slice(0, 120) });
        }
      }
      return unreadList;
    });

    if (!unreadChats || unreadChats.length === 0) {
      return { content: [{ type: "text", text: "No unread messages in Microsoft Teams." }] };
    }

    let text = `Unread Teams Messages (${unreadChats.length} chats):\n`;
    for (const c of unreadChats) {
      const previewStr = c.preview ? ` - "${c.preview}"` : "";
      text += `• ${c.name} (${c.time})${previewStr}\n`;
    }

    return { content: [{ type: "text", text: text.trim() }] };
  }
}

/**
 * 3. get_last_messages
 * Returns the last N messages from a specific chat.
 */
async function handleGetLastMessages(args) {
  const chatName = args?.chat_name?.trim();
  if (!chatName) {
    throw new Error("'chat_name' is required (e.g. 'CAN Ercan', 'Bruno', 'Alex').");
  }
  const count = Math.max(1, Math.min(30, parseInt(args?.count || args?.limit, 10) || 3));

  const page = await getTeamsPage();
  const res = await fetchTeamsMessagesInBrowser(page, chatName, count, false);

  if (res.error) {
    return { content: [{ type: "text", text: `[Teams]: ${res.error}` }] };
  }

  if (!res.messages || res.messages.length === 0) {
    return { content: [{ type: "text", text: `Chat "${res.chat}": No messages found.` }] };
  }

  let output = `Chat: ${res.chat} (Last ${res.messages.length} messages):\n`;
  for (const m of res.messages) {
    const unread = m.is_unread ? " [UNREAD]" : "";
    output += `• [${m.author} | ${m.timestamp}]${unread}: ${m.body}\n`;
  }

  return {
    content: [{ type: "text", text: output.trim() }],
  };
}

/**
 * 4. list_teams_chats
 * Compact list of conversations from sidebar.
 */
async function handleListTeamsChats(args) {
  const limit = Math.max(1, Math.min(30, parseInt(args?.limit, 10) || 10));
  const page = await getTeamsPage();

  const chats = await page.evaluate((maxLimit) => {
    const items = Array.from(document.querySelectorAll('[data-item-type="chat"], [role="treeitem"]'));
    const list = [];
    const seen = new Set();

    for (const item of items) {
      if (
        item.getAttribute("data-conversation-folder") === "true" ||
        item.getAttribute("data-item-type") === "custom-folder"
      ) {
        continue;
      }
      const lines = (item.innerText || "").split("\n").map((l) => l.trim()).filter(Boolean);
      if (lines.length === 0) continue;
      const name = lines[0];
      if (seen.has(name)) continue;
      seen.add(name);

      const time = lines.length > 1 ? lines[1] : "";
      const hasBadge =
        item.querySelector('[class*="badge"], [aria-label*="unread" i], [data-tid*="unread" i]') !== null ||
        (item.getAttribute("aria-label") || "").toLowerCase().includes("unread");
      const isBold =
        window.getComputedStyle(item).fontWeight >= 600 ||
        item.querySelector("strong, b") !== null;

      list.push({ name, time, is_unread: hasBadge || isBold });
      if (list.length >= maxLimit) break;
    }
    return list;
  }, limit);

  if (!chats || chats.length === 0) {
    return { content: [{ type: "text", text: "No Teams conversations found in sidebar." }] };
  }

  let output = `Teams Chats (${chats.length} found):\n`;
  for (const c of chats) {
    const unread = c.is_unread ? " [UNREAD]" : "";
    output += `• ${c.name}${unread} (${c.time})\n`;
  }

  return { content: [{ type: "text", text: output.trim() }] };
}

/**
 * 5. send_teams_message
 * Native Puppeteer typing and verified sending in Teams CKEditor 5.
 */
async function handleSendTeamsMessage(args) {
  const chatName = args?.chat_name?.trim();
  const message = args?.message?.trim();

  if (!chatName || !message) {
    throw new Error("Both 'chat_name' and 'message' are required.");
  }

  const page = await getTeamsPage();

  // 1. Switch to chat using existing robust navigation
  const switchRes = await fetchTeamsMessagesInBrowser(page, chatName, 1, false);
  if (switchRes.error) {
    return { content: [{ type: "text", text: `[Teams Error]: ${switchRes.error}` }] };
  }

  await new Promise((r) => setTimeout(r, 400));

  // 2. Focus and click the compose editor using native Puppeteer
  const editorSelector = 'div[data-tid="ckeditor"], div[role="textbox"][contenteditable="true"]';
  try {
    await page.waitForSelector(editorSelector, { timeout: 8000 });
  } catch (_) {
    return {
      content: [
        {
          type: "text",
          text: `[Teams Error]: Could not find message input box in "${switchRes.chat}".`,
        },
      ],
    };
  }

  await page.click(editorSelector);
  await new Promise((r) => setTimeout(r, 150));

  // 3. Type message using native keyboard driver (recognized by CKEditor 5)
  await page.keyboard.type(message, { delay: 15 });
  await new Promise((r) => setTimeout(r, 200));

  // 4. Send via send button or Ctrl+Enter
  const sendSelector = 'button[data-tid="newMessageCommands-send"], button[aria-label*="Send" i]';
  const sendBtn = await page.$(sendSelector);
  if (sendBtn) {
    const isEnabled = await page.evaluate((el) => !el.disabled, sendBtn);
    if (isEnabled) {
      await sendBtn.click();
    } else {
      await page.keyboard.down("Control");
      await page.keyboard.press("Enter");
      await page.keyboard.up("Control");
    }
  } else {
    await page.keyboard.down("Control");
    await page.keyboard.press("Enter");
    await page.keyboard.up("Control");
  }

  await new Promise((r) => setTimeout(r, 600));

  // 5. Verify message appeared in DOM
  const verifyRes = await fetchTeamsMessagesInBrowser(page, chatName, 3, false);
  const lastMsg = verifyRes.messages && verifyRes.messages[verifyRes.messages.length - 1];
  const isSent = lastMsg && lastMsg.body.includes(message.slice(0, 30));

  if (isSent) {
    return {
      content: [
        {
          type: "text",
          text: `Message successfully sent and confirmed in "${switchRes.chat}": "${message}"`,
        },
      ],
    };
  }

  return {
    content: [
      {
        type: "text",
        text: `Message dispatched to "${switchRes.chat}": "${message}"`,
      },
    ],
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// MCP Tool Registry & Dispatch
// ─────────────────────────────────────────────────────────────────────────────

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "get_last_message",
        description:
          "PRIMARY TOOL to read the single latest message from a specific person, group, or channel in Teams. Supports partial names (e.g. 'CAN Ercan', 'Bruno', 'Alex'). Automatically matches the chat and returns only the author, timestamp, and text. NEVER call list_teams_chats first.",
        inputSchema: {
          type: "object",
          properties: {
            chat_name: {
              type: "string",
              description:
                "Name or partial name of the contact, group, or channel in Teams (e.g. 'CAN Ercan', 'BRUNO Agatino Fabio', 'DATA HUB').",
            },
          },
          required: ["chat_name"],
        },
      },
      {
        name: "get_last_unread_message",
        description:
          "Reads unread messages. If 'chat_name' is provided, returns the latest unread message in that specific chat. If 'chat_name' is omitted, scans all Teams chats and returns a compact list of all unread messages with their senders and text in one single call. NEVER call list_teams_chats first.",
        inputSchema: {
          type: "object",
          properties: {
            chat_name: {
              type: "string",
              description:
                "Optional contact or group name. Leave empty/omitted to scan unread messages across ALL Teams chats.",
            },
          },
        },
      },
      {
        name: "get_last_messages",
        description:
          "Reads the last N messages from a specific Teams chat. Use this ONLY when the user asks for multiple recent messages or recent conversation history. NEVER call list_teams_chats first.",
        inputSchema: {
          type: "object",
          properties: {
            chat_name: {
              type: "string",
              description: "Name or partial name of the contact or group in Teams.",
            },
            count: {
              type: "number",
              description: "Number of recent messages to return (default: 3, max: 20).",
              default: 3,
            },
          },
          required: ["chat_name"],
        },
      },
      {
        name: "send_teams_message",
        description: "Sends a text message to a specified Microsoft Teams chat or group.",
        inputSchema: {
          type: "object",
          properties: {
            chat_name: {
              type: "string",
              description: "Target contact or group name in Teams.",
            },
            message: {
              type: "string",
              description: "The text message to send.",
            },
          },
          required: ["chat_name", "message"],
        },
      },
      {
        name: "list_teams_chats",
        description:
          "Lists chat names and activity timestamps from the Teams sidebar. Use ONLY when the user explicitly asks to list, see, or browse their conversations.",
        inputSchema: {
          type: "object",
          properties: {
            limit: {
              type: "number",
              description: "Maximum number of chats to list (default: 10).",
              default: 10,
            },
          },
        },
      },
    ],
  };
});

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: toolArgs } = request.params;

  try {
    switch (name) {
      case "get_last_message":
        return await handleGetLastMessage(toolArgs);
      case "get_last_unread_message":
        return await handleGetLastUnreadMessage(toolArgs);
      case "get_last_messages":
      case "get_teams_messages": // backward-compatible alias
        return await handleGetLastMessages(toolArgs);
      case "list_teams_chats":
        return await handleListTeamsChats(toolArgs);
      case "send_teams_message":
        return await handleSendTeamsMessage(toolArgs);
      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error) {
    return {
      content: [
        {
          type: "text",
          text: `[Teams MCP Error]: ${error.message}`,
        },
      ],
      isError: true,
    };
  }
});

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("[teams-puppeteer-mcp] Server running on stdio");
}

main().catch((err) => {
  console.error("[teams-puppeteer-mcp] Fatal error:", err);
  process.exit(1);
});
