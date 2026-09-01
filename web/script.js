const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const messagesContainer = document.getElementById('messages-container');
const welcomeScreen = document.getElementById('welcome-screen');

const tempSlider = document.getElementById('temp-slider');
const tempVal = document.getElementById('temp-val');
const tokensInput = document.getElementById('tokens-input');
const reasoningEffort = document.getElementById('reasoning-effort');
const thinkingBudget = document.getElementById('thinking-budget');
const thinkingEnabled = document.getElementById('thinking-enabled');
const systemPromptInput = document.getElementById('system-prompt');

let chatHistory = [];
let currentAbortController = null;

// Performance stats tracking
let sessionStats = {
    totalTurns: 0,
    totalPromptTokens: 0,
    totalCompletionTokens: 0,
    totalTtftMs: 0,
    totalDecodeTimeMs: 0
};

function updateStatsUI(lastStats) {
    const elLastTtft = document.getElementById('stat-last-ttft');
    const elLastPrefill = document.getElementById('stat-last-prefill');
    const elLastDecode = document.getElementById('stat-last-decode');
    const elLastTokens = document.getElementById('stat-last-tokens');

    const elAvgTtft = document.getElementById('stat-avg-ttft');
    const elAvgPrefill = document.getElementById('stat-avg-prefill');
    const elAvgDecode = document.getElementById('stat-avg-decode');
    const elTotalTokens = document.getElementById('stat-total-tokens');

    if (lastStats) {
        if (elLastTtft) elLastTtft.textContent = `${lastStats.ttftSec.toFixed(2)}s`;
        if (elLastPrefill) elLastPrefill.textContent = lastStats.prefillTps > 0 ? `${lastStats.prefillTps.toFixed(1)} t/s` : '-';
        if (elLastDecode) elLastDecode.textContent = lastStats.decodeTps > 0 ? `${lastStats.decodeTps.toFixed(1)} t/s` : '-';
        if (elLastTokens) elLastTokens.textContent = `${lastStats.completionTokens} tok`;
    } else {
        if (elLastTtft) elLastTtft.textContent = '-';
        if (elLastPrefill) elLastPrefill.textContent = '-';
        if (elLastDecode) elLastDecode.textContent = '-';
        if (elLastTokens) elLastTokens.textContent = '-';
    }

    if (sessionStats.totalTurns > 0) {
        const avgTtftSec = (sessionStats.totalTtftMs / sessionStats.totalTurns) / 1000.0;
        const totalPrefillSec = sessionStats.totalTtftMs / 1000.0;
        const totalDecodeSec = sessionStats.totalDecodeTimeMs / 1000.0;
        
        const avgPrefillTps = (totalPrefillSec > 0 && sessionStats.totalPromptTokens > 0)
            ? (sessionStats.totalPromptTokens / totalPrefillSec)
            : 0;
        const avgDecodeTps = (totalDecodeSec > 0 && sessionStats.totalCompletionTokens > 0)
            ? (sessionStats.totalCompletionTokens / totalDecodeSec)
            : 0;

        if (elAvgTtft) elAvgTtft.textContent = `${avgTtftSec.toFixed(2)}s`;
        if (elAvgPrefill) elAvgPrefill.textContent = avgPrefillTps > 0 ? `${avgPrefillTps.toFixed(1)} t/s` : '-';
        if (elAvgDecode) elAvgDecode.textContent = avgDecodeTps > 0 ? `${avgDecodeTps.toFixed(1)} t/s` : '-';
        if (elTotalTokens) elTotalTokens.textContent = `${sessionStats.totalCompletionTokens} tok (${sessionStats.totalTurns} turns)`;
    } else {
        if (elAvgTtft) elAvgTtft.textContent = '-';
        if (elAvgPrefill) elAvgPrefill.textContent = '-';
        if (elAvgDecode) elAvgDecode.textContent = '-';
        if (elTotalTokens) elTotalTokens.textContent = '-';
    }
}

let isGenerating = false;

function setGeneratingState(generating) {
    isGenerating = generating;
    if (generating) {
        sendBtn.classList.add('stop-mode');
        sendBtn.innerHTML = '<span class="material-symbols-outlined">stop</span>';
        sendBtn.title = 'Stop generation';
        sendBtn.disabled = false;
    } else {
        sendBtn.classList.remove('stop-mode');
        sendBtn.innerHTML = '<span class="material-symbols-outlined">send</span>';
        sendBtn.title = 'Send message';
        sendBtn.disabled = chatInput.value.trim() === '';
    }
}

function getApiBase() {
    if (typeof window !== 'undefined' && window.location && window.location.protocol.startsWith('http')) {
        return window.location.origin;
    }
    return 'http://localhost:8000';
}

function stopGeneration() {
    if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
    }
    const apiUrl = `${getApiBase()}/v1/chat/stop`;
    fetch(apiUrl, { method: 'POST' }).catch(() => {});
}

tempSlider.addEventListener('input', (e) => {
    tempVal.textContent = e.target.value;
});

chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + 'px';
    if (!isGenerating) {
        sendBtn.disabled = chatInput.value.trim() === '';
    }
});

chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!isGenerating && !sendBtn.disabled) sendMessage();
    }
});

sendBtn.addEventListener('click', () => {
    if (isGenerating) {
        stopGeneration();
    } else {
        sendMessage();
    }
});

function clearChat() {
    stopGeneration();
    chatHistory = [];
    sessionStats = {
        totalTurns: 0,
        totalPromptTokens: 0,
        totalCompletionTokens: 0,
        totalTtftMs: 0,
        totalDecodeTimeMs: 0
    };
    updateStatsUI(null);
    messagesContainer.innerHTML = '';
    messagesContainer.appendChild(welcomeScreen);
    welcomeScreen.style.display = 'block';
    chatInput.value = '';
    chatInput.style.height = 'auto';
    setGeneratingState(false);
    updateChatbarPreviewButtonVisibility();
    chatInput.focus();
}

// ============================================================================
// Resizable HTML Preview & Test Panel Implementation
// ============================================================================

const previewPanel = document.getElementById('preview-panel');
const resizerHandle = document.getElementById('resizer-handle');
const previewToggleBtn = document.getElementById('preview-toggle-btn');
const previewIframe = document.getElementById('preview-iframe');
const previewEmptyState = document.getElementById('preview-empty-state');
const previewCodeEditor = document.getElementById('preview-code-editor');
const editorLineNumbers = document.getElementById('editor-line-numbers');
const editorDocSize = document.getElementById('editor-doc-size');
const docStatusBadge = document.getElementById('doc-status-badge');
const consoleOutput = document.getElementById('console-output');
const consoleErrorBadge = document.getElementById('console-error-badge');
const badgeErrCount = document.getElementById('badge-err-count');
const badgeWarnCount = document.getElementById('badge-warn-count');
const badgeLogCount = document.getElementById('badge-log-count');
const viewportFrame = document.getElementById('viewport-frame');
const previewCanvas = document.getElementById('preview-canvas');
const previewMaximizeBtn = document.getElementById('preview-maximize-btn');

let isPreviewOpen = false;
let isMaximized = false;
let currentHtmlCode = '';
let consoleLogs = [];
let activeConsoleFilter = 'all';

// Initialize Panel State from localStorage
function initPreviewPanel() {
    const savedWidth = localStorage.getItem('moecher_preview_width');
    if (savedWidth && parseInt(savedWidth, 10) > 300) {
        previewPanel.style.width = `${parseInt(savedWidth, 10)}px`;
    } else {
        previewPanel.style.width = '540px';
    }

    // Default closed unless explicitly opened
    previewPanel.classList.add('collapsed');
    resizerHandle.classList.add('hidden');
    if (previewToggleBtn) previewToggleBtn.classList.remove('active');
    isPreviewOpen = false;

    setupResizer();
    setupTabs();
    setupViewportControls();
    setupCodeEditor();
    setupConsoleListener();
}

function openPreviewPanel() {
    isPreviewOpen = true;
    previewPanel.classList.remove('collapsed');
    resizerHandle.classList.remove('hidden');
    if (previewToggleBtn) previewToggleBtn.classList.add('active');
    localStorage.setItem('moecher_preview_open', 'true');
}

function closePreviewPanel() {
    isPreviewOpen = false;
    if (isMaximized) toggleMaximizePreview();
    previewPanel.classList.add('collapsed');
    resizerHandle.classList.add('hidden');
    if (previewToggleBtn) previewToggleBtn.classList.remove('active');
    localStorage.setItem('moecher_preview_open', 'false');
}

function togglePreviewPanel() {
    if (isPreviewOpen) {
        closePreviewPanel();
    } else {
        openPreviewPanel();
    }
}

function toggleMaximizePreview() {
    isMaximized = !isMaximized;
    if (isMaximized) {
        previewPanel.classList.add('maximized');
        resizerHandle.classList.add('hidden');
        previewMaximizeBtn.innerHTML = '<span class="material-symbols-outlined">fullscreen_exit</span>';
        previewMaximizeBtn.title = 'Restore Panel Size';
    } else {
        previewPanel.classList.remove('maximized');
        resizerHandle.classList.remove('hidden');
        previewMaximizeBtn.innerHTML = '<span class="material-symbols-outlined">fullscreen</span>';
        previewMaximizeBtn.title = 'Maximize Panel';
    }
}

// Resizer Dragging
function setupResizer() {
    let startX = 0;
    let startWidth = 0;
    let isDragging = false;

    function onMouseDown(e) {
        isDragging = true;
        startX = e.clientX;
        startWidth = previewPanel.getBoundingClientRect().width;
        document.body.classList.add('resizing-active');
        resizerHandle.classList.add('is-resizing');

        window.addEventListener('mousemove', onMouseMove);
        window.addEventListener('mouseup', onMouseUp);
        e.preventDefault();
    }

    function onMouseMove(e) {
        if (!isDragging) return;
        const delta = startX - e.clientX;
        const newWidth = Math.min(Math.max(startWidth + delta, 320), window.innerWidth - 320);
        previewPanel.style.width = `${newWidth}px`;
    }

    function onMouseUp() {
        if (!isDragging) return;
        isDragging = false;
        document.body.classList.remove('resizing-active');
        resizerHandle.classList.remove('is-resizing');
        localStorage.setItem('moecher_preview_width', parseInt(previewPanel.style.width, 10));
        window.removeEventListener('mousemove', onMouseMove);
        window.removeEventListener('mouseup', onMouseUp);
    }

    resizerHandle.addEventListener('mousedown', onMouseDown);
}

// Tabs
function setupTabs() {
    const tabButtons = document.querySelectorAll('.preview-tab');
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            switchPreviewTab(targetTab);
        });
    });
}

function switchPreviewTab(tabId) {
    document.querySelectorAll('.preview-tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

    const activeBtn = document.querySelector(`.preview-tab[data-tab="${tabId}"]`);
    const activePane = document.getElementById(tabId);

    if (activeBtn) activeBtn.classList.add('active');
    if (activePane) activePane.classList.add('active');

    if (tabId === 'tab-code') {
        updateEditorLineNumbers();
    }
}

// Viewport Emulation
function setupViewportControls() {
    const vpButtons = document.querySelectorAll('.viewport-controls .preview-tool-btn');
    vpButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            vpButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const vp = btn.getAttribute('data-viewport');
            
            viewportFrame.classList.remove('tablet-mode', 'mobile-mode');
            if (vp === '768px') {
                viewportFrame.classList.add('tablet-mode');
            } else if (vp === '375px') {
                viewportFrame.classList.add('mobile-mode');
            }
        });
    });
}

// Canvas Background Toggle
function togglePreviewBg() {
    if (previewCanvas.classList.contains('dark-bg')) {
        previewCanvas.classList.remove('dark-bg');
        previewCanvas.classList.add('light-bg');
    } else if (previewCanvas.classList.contains('light-bg')) {
        previewCanvas.classList.remove('light-bg');
        previewCanvas.classList.add('checker-bg');
    } else {
        previewCanvas.classList.remove('checker-bg');
        previewCanvas.classList.add('dark-bg');
    }
}

// Load HTML into preview iframe with console capture bridge
function loadHtmlIntoPreview(htmlCode, autoSwitchTab = true) {
    currentHtmlCode = htmlCode || '';
    if (!isPreviewOpen) openPreviewPanel();

    if (previewCodeEditor) {
        previewCodeEditor.value = currentHtmlCode;
        updateEditorLineNumbers();
    }

    clearConsoleLogs();
    renderPreviewIframe(currentHtmlCode);

    if (previewEmptyState) {
        if (currentHtmlCode.trim().length > 0) {
            previewEmptyState.classList.add('hidden');
        } else {
            previewEmptyState.classList.remove('hidden');
        }
    }

    if (docStatusBadge) {
        docStatusBadge.textContent = 'Active';
        docStatusBadge.classList.remove('modified');
    }

    if (autoSwitchTab) {
        switchPreviewTab('tab-preview');
    }
}

// Check if an HTML snippet is present in the conversation
function hasHtmlSnippet() {
    const codeBlocks = document.querySelectorAll('#messages-container pre code');
    for (let i = codeBlocks.length - 1; i >= 0; i--) {
        const code = codeBlocks[i].textContent || '';
        const lang = (codeBlocks[i].className || '').toLowerCase();
        if (lang.includes('html') || lang.includes('svg') || lang.includes('xml') ||
            code.includes('<!DOCTYPE') || code.includes('<!doctype') || code.includes('<html') || code.includes('<div') || 
            code.includes('<svg') || code.includes('<script') || code.includes('<style') ||
            code.includes('<canvas') || code.includes('<button') || code.includes('<body') || code.includes('<head')) {
            return true;
        }
    }
    if (currentHtmlCode && currentHtmlCode.trim().length > 0) {
        return true;
    }
    return false;
}

// Find the most recent HTML / UI snippet across chat messages
function getLatestHtmlCode() {
    const codeBlocks = document.querySelectorAll('#messages-container pre code');
    for (let i = codeBlocks.length - 1; i >= 0; i--) {
        const code = codeBlocks[i].textContent || '';
        const lang = (codeBlocks[i].className || '').toLowerCase();
        if (lang.includes('html') || lang.includes('svg') || lang.includes('xml') ||
            code.includes('<!DOCTYPE') || code.includes('<!doctype') || code.includes('<html') || code.includes('<div') || 
            code.includes('<svg') || code.includes('<script') || code.includes('<style') ||
            code.includes('<canvas') || code.includes('<button') || code.includes('<body') || code.includes('<head')) {
            return code;
        }
    }
    if (currentHtmlCode && currentHtmlCode.trim().length > 0) {
        return currentHtmlCode;
    }
    return '';
}

// Update visibility of the chat bar preview button (only show when HTML exists in response)
function updateChatbarPreviewButtonVisibility() {
    const btn = document.getElementById('preview-btn-chatbar');
    if (!btn) return;
    if (hasHtmlSnippet()) {
        btn.classList.remove('hidden');
    } else {
        btn.classList.add('hidden');
    }
}

// Chatbar Preview Button Click Handler
function previewLatestHtmlSnippet() {
    const code = getLatestHtmlCode();
    if (code && code.trim().length > 0) {
        loadHtmlIntoPreview(code, true);
    } else {
        togglePreviewPanel(true);
    }
}

function renderPreviewIframe(htmlCode) {
    if (!previewIframe) return;

    // Inject console interception & error listening bridge
    const consoleBridge = `
<script>
(function() {
    function serializeMsg(arg) {
        if (arg === null) return 'null';
        if (arg === undefined) return 'undefined';
        if (typeof arg === 'object') {
            try { return JSON.stringify(arg, null, 2); } catch (e) { return Object.prototype.toString.call(arg); }
        }
        return String(arg);
    }

    function sendToParent(type, args) {
        try {
            const formatted = args.map(serializeMsg).join(' ');
            window.parent.postMessage({
                type: 'moecher-console-log',
                level: type,
                message: formatted,
                timestamp: new Date().toLocaleTimeString()
            }, '*');
        } catch(e) {}
    }

    const _log = console.log;
    const _warn = console.warn;
    const _error = console.error;
    const _info = console.info;

    console.log = function(...args) { _log.apply(console, args); sendToParent('log', args); };
    console.warn = function(...args) { _warn.apply(console, args); sendToParent('warn', args); };
    console.error = function(...args) { _error.apply(console, args); sendToParent('error', args); };
    console.info = function(...args) { _info.apply(console, args); sendToParent('info', args); };

    window.addEventListener('error', function(e) {
        sendToParent('error', [e.message + (e.filename ? ' (' + e.filename + ':' + e.lineno + ')' : '')]);
    });

    window.addEventListener('unhandledrejection', function(e) {
        sendToParent('error', ['Unhandled Promise Rejection: ' + (e.reason ? (e.reason.stack || e.reason) : 'Unknown')]);
    });
})();
</script>
`;

    let finalHtml = htmlCode;
    if (finalHtml.includes('<head>')) {
        finalHtml = finalHtml.replace('<head>', '<head>' + consoleBridge);
    } else if (finalHtml.includes('<html>')) {
        finalHtml = finalHtml.replace('<html>', '<html><head>' + consoleBridge + '</head>');
    } else {
        finalHtml = consoleBridge + finalHtml;
    }

    previewIframe.srcdoc = finalHtml;
}

function reloadPreview() {
    if (previewCodeEditor) {
        renderPreviewIframe(previewCodeEditor.value);
    } else {
        renderPreviewIframe(currentHtmlCode);
    }
}

function openPreviewInNewTab() {
    const code = previewCodeEditor ? previewCodeEditor.value : currentHtmlCode;
    if (!code) return;
    const blob = new Blob([code], { type: 'text/html;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    window.open(url, '_blank');
}

function downloadPreviewHtml() {
    const code = previewCodeEditor ? previewCodeEditor.value : currentHtmlCode;
    if (!code) return;
    const blob = new Blob([code], { type: 'text/html;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'moecher_preview.html';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Code Editor Setup
function setupCodeEditor() {
    if (!previewCodeEditor) return;

    previewCodeEditor.addEventListener('input', () => {
        updateEditorLineNumbers();
        if (docStatusBadge) {
            docStatusBadge.textContent = 'Modified';
            docStatusBadge.classList.add('modified');
        }
    });

    previewCodeEditor.addEventListener('scroll', () => {
        if (editorLineNumbers) {
            editorLineNumbers.scrollTop = previewCodeEditor.scrollTop;
        }
    });

    // Support Tab key indentation
    previewCodeEditor.addEventListener('keydown', (e) => {
        if (e.key === 'Tab') {
            e.preventDefault();
            const start = previewCodeEditor.selectionStart;
            const end = previewCodeEditor.selectionEnd;
            previewCodeEditor.value = previewCodeEditor.value.substring(0, start) + '  ' + previewCodeEditor.value.substring(end);
            previewCodeEditor.selectionStart = previewCodeEditor.selectionEnd = start + 2;
            updateEditorLineNumbers();
        }
    });
}

function updateEditorLineNumbers() {
    if (!previewCodeEditor || !editorLineNumbers) return;
    const lines = previewCodeEditor.value.split('\n');
    const lineCount = lines.length;
    let numbersText = '';
    for (let i = 1; i <= lineCount; i++) {
        numbersText += i + '\n';
    }
    editorLineNumbers.textContent = numbersText;
    if (editorDocSize) {
        editorDocSize.textContent = `${lineCount} lines (${(previewCodeEditor.value.length / 1024).toFixed(1)} KB)`;
    }
}

function runCodeFromEditor() {
    if (!previewCodeEditor) return;
    currentHtmlCode = previewCodeEditor.value;
    renderPreviewIframe(currentHtmlCode);
    if (previewEmptyState) previewEmptyState.classList.add('hidden');
    if (docStatusBadge) {
        docStatusBadge.textContent = 'Active';
        docStatusBadge.classList.remove('modified');
    }
    switchPreviewTab('tab-preview');
}

function copyEditorCode() {
    if (!previewCodeEditor) return;
    navigator.clipboard.writeText(previewCodeEditor.value).then(() => {
        alert('Code copied to clipboard!');
    });
}

function clearEditorCode() {
    if (!previewCodeEditor) return;
    previewCodeEditor.value = '';
    updateEditorLineNumbers();
    currentHtmlCode = '';
    renderPreviewIframe('');
    if (previewEmptyState) previewEmptyState.classList.remove('hidden');
}

function formatEditorCode() {
    if (!previewCodeEditor || !previewCodeEditor.value) return;
    let formatted = '';
    let pad = 0;
    const tokens = previewCodeEditor.value.replace(/>\s*</g, '>\n<').split('\n');
    tokens.forEach(node => {
        let indent = 0;
        if (node.match(/.+<\/\w[^>]*>$/)) {
            indent = 0;
        } else if (node.match(/^<\/\w/)) {
            if (pad > 0) pad -= 1;
        } else if (node.match(/^<\w[^>]*[^\/]>.*$/)) {
            indent = 1;
        }
        let padding = '';
        for (let i = 0; i < pad; i++) padding += '  ';
        formatted += padding + node.trim() + '\n';
        pad += indent;
    });
    previewCodeEditor.value = formatted.trim();
    updateEditorLineNumbers();
}

// Console logs
function setupConsoleListener() {
    window.addEventListener('message', (event) => {
        if (event.data && event.data.type === 'moecher-console-log') {
            addConsoleLog(event.data.level, event.data.message, event.data.timestamp);
        }
    });

    const filterBtns = document.querySelectorAll('.console-filter-btn');
    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            activeConsoleFilter = btn.getAttribute('data-filter');
            renderConsoleLogs();
        });
    });
}

function addConsoleLog(level, message, timestamp) {
    consoleLogs.push({ level, message, timestamp });
    updateConsoleBadges();
    renderConsoleLogs();
}

function clearConsoleLogs() {
    consoleLogs = [];
    updateConsoleBadges();
    renderConsoleLogs();
}

function updateConsoleBadges() {
    const errors = consoleLogs.filter(l => l.level === 'error').length;
    const warns = consoleLogs.filter(l => l.level === 'warn').length;
    const logs = consoleLogs.filter(l => l.level === 'log' || l.level === 'info').length;

    if (badgeErrCount) badgeErrCount.textContent = errors;
    if (badgeWarnCount) badgeWarnCount.textContent = warns;
    if (badgeLogCount) badgeLogCount.textContent = logs;

    if (consoleErrorBadge) {
        if (errors > 0) {
            consoleErrorBadge.style.display = 'inline-block';
            consoleErrorBadge.textContent = errors;
        } else {
            consoleErrorBadge.style.display = 'none';
        }
    }
}

function renderConsoleLogs() {
    if (!consoleOutput) return;

    const filtered = consoleLogs.filter(entry => {
        if (activeConsoleFilter === 'all') return true;
        if (activeConsoleFilter === 'error') return entry.level === 'error';
        if (activeConsoleFilter === 'warn') return entry.level === 'warn';
        if (activeConsoleFilter === 'log') return entry.level === 'log' || entry.level === 'info';
        return true;
    });

    if (filtered.length === 0) {
        consoleOutput.innerHTML = '<div class="console-empty">No log messages found.</div>';
        return;
    }

    consoleOutput.innerHTML = '';
    filtered.forEach(entry => {
        const row = document.createElement('div');
        row.className = `console-entry ${entry.level}`;
        row.innerHTML = `<span class="console-time">${entry.timestamp || ''}</span><span class="console-msg">${escapeHtml(entry.message)}</span>`;
        consoleOutput.appendChild(row);
    });

    consoleOutput.scrollTop = consoleOutput.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Sample Demo HTML
function loadSampleHtml() {
    const sampleHtml = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Particle Network</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            overflow: hidden;
            background: radial-gradient(circle at center, #1b2735 0%, #090a0f 100%);
            font-family: system-ui, -apple-system, sans-serif;
            color: #ffffff;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100vh;
        }
        canvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }
        .hud {
            position: relative;
            z-index: 10;
            text-align: center;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(16px);
            padding: 24px 36px;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            box-shadow: 0 20px 50px rgba(0,0,0,0.5);
            user-select: none;
        }
        h1 {
            margin: 0 0 8px;
            font-size: 24px;
            background: linear-gradient(135deg, #60a5fa, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        p {
            margin: 0 0 16px;
            color: #94a3b8;
            font-size: 14px;
        }
        .btn {
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            border: none;
            color: white;
            padding: 10px 20px;
            border-radius: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .btn:hover {
            transform: scale(1.05);
            box-shadow: 0 0 20px rgba(139, 92, 246, 0.5);
        }
    </style>
</head>
<body>
    <canvas id="canvas"></canvas>
    <div class="hud">
        <h1>Moecher Live HTML Demo</h1>
        <p>Move your mouse or touch the canvas to interact with particles.</p>
        <button class="btn" onclick="burst()">Spawn Shockwave</button>
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let width = canvas.width = window.innerWidth;
        let height = canvas.height = window.innerHeight;

        window.addEventListener('resize', () => {
            width = canvas.width = window.innerWidth;
            height = canvas.height = window.innerHeight;
        });

        console.log("Interactive Particle Demo loaded successfully! Canvas dimensions:", width, height);

        const particles = [];
        const particleCount = 70;
        const mouse = { x: width/2, y: height/2, radius: 120 };

        window.addEventListener('mousemove', (e) => {
            mouse.x = e.clientX;
            mouse.y = e.clientY;
        });

        for (let i = 0; i < particleCount; i++) {
            particles.push({
                x: Math.random() * width,
                y: Math.random() * height,
                vx: (Math.random() - 0.5) * 1.5,
                vy: (Math.random() - 0.5) * 1.5,
                size: Math.random() * 3 + 1,
                color: 'hsl(' + (Math.random() * 60 + 200) + ', 80%, 70%)'
            });
        }

        function burst() {
            console.log("Shockwave triggered!");
            particles.forEach(p => {
                const dx = p.x - width/2;
                const dy = p.y - height/2;
                const dist = Math.hypot(dx, dy) || 1;
                p.vx += (dx / dist) * 10;
                p.vy += (dy / dist) * 10;
            });
        }

        function animate() {
            ctx.clearRect(0, 0, width, height);

            for (let i = 0; i < particles.length; i++) {
                const p = particles[i];
                p.x += p.vx;
                p.y += p.vy;
                p.vx *= 0.98;
                p.vy *= 0.98;

                if (p.x < 0 || p.x > width) p.vx *= -1;
                if (p.y < 0 || p.y > height) p.vy *= -1;

                // Mouse repel
                const dx = p.x - mouse.x;
                const dy = p.y - mouse.y;
                const dist = Math.hypot(dx, dy);
                if (dist < mouse.radius) {
                    const force = (mouse.radius - dist) / mouse.radius;
                    p.vx += (dx / dist) * force * 1.5;
                    p.vy += (dy / dist) * force * 1.5;
                }

                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                ctx.fillStyle = p.color;
                ctx.fill();

                // Connect nearby particles
                for (let j = i + 1; j < particles.length; j++) {
                    const p2 = particles[j];
                    const dist2 = Math.hypot(p.x - p2.x, p.y - p2.y);
                    if (dist2 < 100) {
                        ctx.beginPath();
                        ctx.strokeStyle = 'rgba(148, 163, 184, ' + (1 - dist2 / 100) * 0.25 + ')';
                        ctx.lineWidth = 1;
                        ctx.moveTo(p.x, p.y);
                        ctx.lineTo(p2.x, p2.y);
                        ctx.stroke();
                    }
                }
            }
            requestAnimationFrame(animate);
        }
        animate();
    <\/script>
</body>
</html>`;

    loadHtmlIntoPreview(sampleHtml);
}

// Comprehensive HTML Detection
function isHtmlContent(codeContent, lang = '') {
    if (!codeContent || typeof codeContent !== 'string') return false;
    const l = (lang || '').toLowerCase().trim();
    const htmlLangs = ['html', 'htm', 'xml', 'svg', 'xhtml', 'markup', 'web', 'php', 'vue', 'svelte', 'jsx', 'tsx', 'blade'];
    if (htmlLangs.includes(l)) return true;

    const trimmed = codeContent.trim();
    if (/<!doctype\s+html/i.test(trimmed)) return true;
    if (/<html[\s>]/i.test(trimmed)) return true;
    if (/<head[\s>]/i.test(trimmed) && /<\/head>/i.test(trimmed)) return true;
    if (/<body[\s>]/i.test(trimmed) && /<\/body>/i.test(trimmed)) return true;
    if (/<script[\s>]/i.test(trimmed) && /<\/script>/i.test(trimmed)) return true;
    if (/<style[\s>]/i.test(trimmed) && /<\/style>/i.test(trimmed)) return true;
    if (/<svg[\s>]/i.test(trimmed) && /<\/svg>/i.test(trimmed)) return true;

    // Detect common HTML structure tags (open + close or multiple tags)
    const tagMatches = trimmed.match(/<\/?([a-zA-Z][a-zA-Z0-9-]*)\b[^>]*>/g);
    if (tagMatches && tagMatches.length >= 2) {
        const commonHtmlTags = [
            'div', 'span', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 
            'ul', 'ol', 'li', 'table', 'tr', 'td', 'th', 'thead', 'tbody',
            'button', 'input', 'form', 'label', 'select', 'option', 'textarea',
            'canvas', 'section', 'article', 'nav', 'header', 'footer', 'main',
            'aside', 'iframe', 'video', 'audio', 'img', 'a', 'link', 'meta',
            'details', 'summary', 'style', 'script', 'style'
        ];
        return commonHtmlTags.some(tag => 
            new RegExp(`<${tag}[\\s>]`, 'i').test(trimmed) || 
            new RegExp(`</${tag}>`, 'i').test(trimmed)
        );
    }

    return false;
}

// Markdown rendering and code-block post-processing
function renderMarkdownContent(rawText, containerElement) {
    containerElement.innerHTML = marked.parse(rawText);

    // Decorate code blocks
    const codeBlocks = containerElement.querySelectorAll('pre > code');
    let hasPreviewableHtml = false;
    let firstHtmlSnippet = '';

    codeBlocks.forEach(codeEl => {
        const preEl = codeEl.parentElement;
        if (preEl.parentElement.classList.contains('code-block-wrapper')) return;

        const wrapper = document.createElement('div');
        wrapper.className = 'code-block-wrapper';

        // Detect language
        let lang = 'code';
        const classes = codeEl.className.split(' ');
        for (const cls of classes) {
            if (cls.startsWith('language-')) {
                lang = cls.replace('language-', '').toLowerCase();
                break;
            }
        }

        const codeContent = codeEl.textContent;
        const isHtmlCandidate = isHtmlContent(codeContent, lang);

        if (isHtmlCandidate && !firstHtmlSnippet) {
            hasPreviewableHtml = true;
            firstHtmlSnippet = codeContent;
        }

        const header = document.createElement('div');
        header.className = 'code-block-header';

        const langDiv = document.createElement('div');
        langDiv.className = 'code-block-lang';
        langDiv.innerHTML = `<span class="material-symbols-outlined lang-icon">${isHtmlCandidate ? 'html' : 'code'}</span> ${lang.toUpperCase()}`;

        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'code-block-actions';

        const previewBtn = document.createElement('button');
        previewBtn.className = isHtmlCandidate ? 'code-action-btn preview-btn' : 'code-action-btn';
        previewBtn.innerHTML = `<span class="material-symbols-outlined btn-icon">${isHtmlCandidate ? 'play_circle' : 'preview'}</span> Preview`;
        previewBtn.title = 'Test and render this snippet in the HTML preview panel';
        previewBtn.addEventListener('click', () => {
            loadHtmlIntoPreview(codeContent, true);
        });
        actionsDiv.appendChild(previewBtn);

        const copyBtn = document.createElement('button');
        copyBtn.className = 'code-action-btn';
        copyBtn.innerHTML = `<span class="material-symbols-outlined btn-icon">content_copy</span> Copy`;
        copyBtn.addEventListener('click', () => {
            navigator.clipboard.writeText(codeContent).then(() => {
                copyBtn.innerHTML = `<span class="material-symbols-outlined btn-icon">done</span> Copied!`;
                setTimeout(() => {
                    copyBtn.innerHTML = `<span class="material-symbols-outlined btn-icon">content_copy</span> Copy`;
                }, 2000);
            });
        });
        actionsDiv.appendChild(copyBtn);

        header.appendChild(langDiv);
        header.appendChild(actionsDiv);

        preEl.parentNode.insertBefore(wrapper, preEl);
        wrapper.appendChild(header);
        wrapper.appendChild(preEl);
    });

    // Check if raw message (without code blocks or enclosing entire text) is an HTML document
    if (!hasPreviewableHtml && isHtmlContent(rawText)) {
        let existingBanner = containerElement.parentElement ? containerElement.parentElement.querySelector('.msg-html-banner') : null;
        if (!existingBanner) {
            const banner = document.createElement('div');
            banner.className = 'msg-html-banner';
            banner.innerHTML = `
                <div class="msg-html-banner-left">
                    <span class="material-symbols-outlined msg-html-banner-icon">html</span>
                    <div>
                        <div class="msg-html-banner-text">HTML Document Detected</div>
                        <div class="msg-html-banner-sub">Test and interact with this document in the preview panel</div>
                    </div>
                </div>
                <button class="msg-html-banner-btn">
                    <span class="material-symbols-outlined">play_circle</span>
                    <span>Open in Preview</span>
                </button>
            `;
            const bannerBtn = banner.querySelector('.msg-html-banner-btn');
            bannerBtn.addEventListener('click', () => {
                loadHtmlIntoPreview(rawText, true);
            });
            containerElement.appendChild(banner);
        }
    }
    updateChatbarPreviewButtonVisibility();
}

// ============================================================================
// Main Chat Logic
// ============================================================================

async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text || isGenerating) return;
    
    if (currentAbortController) {
        currentAbortController.abort();
    }
    currentAbortController = new AbortController();

    chatInput.value = '';
    chatInput.style.height = 'auto';
    setGeneratingState(true);
    welcomeScreen.style.display = 'none';

    // Add user message
    appendMessage('user', text);
    chatHistory.push({ role: 'user', content: text });

    // Create assistant message container
    const assistantMsgDiv = createMessageContainer('assistant');
    messagesContainer.appendChild(assistantMsgDiv);
    
    // Add reasoning block (hidden initially)
    let reasoningBlock = null;
    let reasoningContent = null;
    let mainContent = document.createElement('div');
    assistantMsgDiv.querySelector('.msg-content').appendChild(mainContent);

    // Build messages payload with optional system prompt
    const messagesToSend = [];
    const sysPrompt = systemPromptInput ? systemPromptInput.value.trim() : '';
    if (sysPrompt) {
        messagesToSend.push({ role: 'system', content: sysPrompt });
    }
    messagesToSend.push(...chatHistory);

    const isThinking = thinkingEnabled ? thinkingEnabled.checked : true;
    const budgetVal = isThinking ? (thinkingBudget ? parseInt(thinkingBudget.value, 10) : 4096) : 0;
    const payload = {
        model: "deepseek-v4-flash",
        messages: messagesToSend,
        max_tokens: parseInt(tokensInput.value, 10),
        temperature: parseFloat(tempSlider.value),
        stream: true,
        thinking: { 
            type: isThinking ? "enabled" : "disabled",
            budget_tokens: budgetVal
        },
        max_thinking_tokens: budgetVal,
        reasoning_effort: isThinking ? reasoningEffort.value : "none"
    };

    let rawReasoning = "";
    let rawContent = "";
    let isReasoningDone = false;

    const startTime = performance.now();
    let firstTokenTime = null;
    let tokenCount = 0;
    let promptTokens = 0;
    let completionTokens = 0;

    try {
        const apiUrl = `${getApiBase()}/v1/chat/completions`;

        function pythonJsonDumps(obj) {
            if (obj === null) return 'null';
            if (typeof obj === 'boolean') return obj ? 'true' : 'false';
            if (typeof obj === 'number') return String(obj);
            if (typeof obj === 'string') return JSON.stringify(obj);
            if (Array.isArray(obj)) {
                return '[' + obj.map(pythonJsonDumps).join(', ') + ']';
            }
            if (typeof obj === 'object') {
                const entries = Object.entries(obj).map(([k, v]) => `${JSON.stringify(k)}: ${pythonJsonDumps(v)}`);
                return '{' + entries.join(', ') + '}';
            }
            return JSON.stringify(obj);
        }

        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream'
            },
            body: pythonJsonDumps(payload),
            signal: currentAbortController.signal
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const dataStr = line.slice(6);
                    if (dataStr === '[DONE]') break;
                    
                    try {
                        const data = JSON.parse(dataStr);

                        if (data.usage) {
                            if (data.usage.prompt_tokens !== undefined) promptTokens = data.usage.prompt_tokens;
                            if (data.usage.completion_tokens !== undefined) completionTokens = data.usage.completion_tokens;
                        }
                        
                        if (data.choices && data.choices.length > 0) {
                            const delta = data.choices[0].delta || {};
                            
                            if (delta.reasoning_content !== undefined) {
                                if (firstTokenTime === null) firstTokenTime = performance.now();
                                tokenCount++;

                                if (!reasoningBlock) {
                                    reasoningBlock = document.createElement('details');
                                    reasoningBlock.className = 'reasoning-block';
                                    reasoningBlock.open = true;
                                    
                                    const summary = document.createElement('summary');
                                    summary.innerHTML = '<span class="thinking-spinner">progress_activity</span> Thinking...';
                                    
                                    reasoningContent = document.createElement('div');
                                    reasoningContent.className = 'reasoning-content';
                                    
                                    reasoningBlock.appendChild(summary);
                                    reasoningBlock.appendChild(reasoningContent);
                                    assistantMsgDiv.querySelector('.msg-content').insertBefore(reasoningBlock, mainContent);
                                }
                                rawReasoning += delta.reasoning_content;
                                reasoningContent.textContent = rawReasoning;
                            } 
                            
                            if (delta.content !== undefined) {
                                if (firstTokenTime === null) firstTokenTime = performance.now();
                                tokenCount++;

                                if (reasoningBlock && !isReasoningDone) {
                                    isReasoningDone = true;
                                    reasoningBlock.open = false;
                                    reasoningBlock.querySelector('summary').innerHTML = 'Thought process';
                                }
                                rawContent += delta.content;
                                renderMarkdownContent(rawContent, mainContent);
                            }
                        }
                        
                        messagesContainer.scrollTo({
                            top: messagesContainer.scrollHeight,
                            behavior: 'smooth'
                        });

                    } catch (e) {
                        console.error('JSON Parse error', e, dataStr);
                    }
                }
            }
        }
        
        const endTime = performance.now();
        if (firstTokenTime === null) firstTokenTime = endTime;
        const ttftMs = firstTokenTime - startTime;
        const decodeTimeMs = Math.max(0, endTime - firstTokenTime);
        const ttftSec = ttftMs / 1000.0;
        const decodeSec = decodeTimeMs / 1000.0;

        const actualCompletionTokens = completionTokens > 0 ? completionTokens : tokenCount;
        const prefillTps = (ttftSec > 0 && promptTokens > 0) ? (promptTokens / ttftSec) : 0.0;
        const decodeTps = (decodeSec > 0 && actualCompletionTokens > 1) ? ((actualCompletionTokens - 1) / decodeSec) : 0.0;

        const lastStats = {
            ttftSec,
            decodeSec,
            promptTokens,
            completionTokens: actualCompletionTokens,
            prefillTps,
            decodeTps
        };

        sessionStats.totalTurns++;
        sessionStats.totalPromptTokens += promptTokens;
        sessionStats.totalCompletionTokens += actualCompletionTokens;
        sessionStats.totalTtftMs += ttftMs;
        sessionStats.totalDecodeTimeMs += decodeTimeMs;

        updateStatsUI(lastStats);

        // Final pass on message content
        renderMarkdownContent(rawContent, mainContent);

        chatHistory.push({
            role: 'assistant',
            content: rawContent,
            reasoning_content: rawReasoning || undefined
        });
        
    } catch (err) {
        if (err.name === 'AbortError') {
            console.log('Request aborted by user.');
            if (reasoningBlock && !isReasoningDone) {
                isReasoningDone = true;
                reasoningBlock.open = false;
                const summaryEl = reasoningBlock.querySelector('summary');
                if (summaryEl) summaryEl.innerHTML = 'Thought process (stopped)';
            }
            if (rawContent || rawReasoning) {
                renderMarkdownContent(rawContent, mainContent);
                chatHistory.push({
                    role: 'assistant',
                    content: rawContent,
                    reasoning_content: rawReasoning || undefined
                });
            }
            return;
        }
        console.error(err);
        mainContent.innerHTML += `<br><br><b>Error:</b> Failed to connect to engine. Make sure it's running.`;
    } finally {
        currentAbortController = null;
        setGeneratingState(false);
        // Refresh expert specialization profile after generation
        fetchExpertProfile();
    }
}

function createMessageContainer(role) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    
    const contentWrapper = document.createElement('div');
    contentWrapper.className = 'msg-content';
    
    div.appendChild(contentWrapper);
    return div;
}

function appendMessage(role, text) {
    const div = createMessageContainer(role);
    if (role === 'user') {
        div.querySelector('.msg-content').textContent = text;
    }
    messagesContainer.appendChild(div);
    messagesContainer.scrollTo({
        top: messagesContainer.scrollHeight,
        behavior: 'smooth'
    });
}

// ════════════════════════════════════════════════════════════════════════════════
//  Expert Specialization & Analytics UI Handlers
// ════════════════════════════════════════════════════════════════════════════════

let currentExpertProfileData = null;
let currentExpertFilter = 'all';

async function fetchExpertProfile(event) {
    if (event) {
        event.stopPropagation();
        event.preventDefault();
    }
    const apiUrl = `${getApiBase()}/v1/experts/profile`;
    try {
        const res = await fetch(apiUrl);
        if (!res.ok) throw new Error(`HTTP error ${res.status}`);
        const data = await res.json();
        currentExpertProfileData = data;
        renderExpertProfileUI(data, currentExpertFilter);
    } catch (err) {
        console.warn('Could not fetch expert profile:', err);
    }
}

function getCategoryClass(category) {
    switch (category) {
        case 'Coding / Syntax': return 'cat-code';
        case 'Math / Logic': return 'cat-math';
        case 'Reasoning': return 'cat-reasoning';
        case 'General Prose': return 'cat-prose';
        case 'Format / Syntax': return 'cat-format';
        case 'Multilingual': return 'cat-multi';
        default: return 'cat-prose';
    }
}

let currentFilterType = 'cat';

function renderExpertProfileUI(data) {
    const elTokens = document.getElementById('expert-tracked-tokens-val');
    const elActive = document.getElementById('expert-active-count-val');
    const elL1 = document.getElementById('res-l1-label');
    const elL2 = document.getElementById('res-l2-label');
    const elSSD = document.getElementById('res-ssd-label');
    const listContainer = document.getElementById('expert-list-container');

    if (data) {
        if (elL1 && data.l1_resident !== undefined) elL1.textContent = `L1: ${data.l1_resident.toLocaleString()}`;
        if (elL2 && data.l2_resident !== undefined) elL2.textContent = `L2: ${data.l2_resident.toLocaleString()}`;
        if (elSSD && data.ssd_count !== undefined) elSSD.textContent = `SSD: ${data.ssd_count.toLocaleString()}`;
    }

    if (!data || !data.experts || data.experts.length === 0) {
        if (elTokens) elTokens.textContent = '0 tokens';
        if (elActive) elActive.textContent = '0 active';
        if (listContainer) {
            listContainer.innerHTML = `
                <div class="expert-empty-hint">
                    <span class="material-symbols-outlined">info</span>
                    <span>No expert profile data loaded yet.</span>
                </div>`;
        }
        return;
    }

    if (elTokens) elTokens.textContent = `${data.total_tokens.toLocaleString()} tokens`;
    if (elActive) elActive.textContent = `${data.total_active_experts || data.experts.length} active`;

    if (!listContainer) return;

    let filtered = data.experts;
    if (currentFilterType === 'tier' && currentExpertFilter !== 'all') {
        filtered = data.experts.filter(e => (e.tier || 'l1') === currentExpertFilter);
    } else if (currentFilterType === 'cat' && currentExpertFilter !== 'all') {
        filtered = data.experts.filter(e => e.category === currentExpertFilter);
    }

    if (filtered.length === 0) {
        listContainer.innerHTML = `
            <div class="expert-empty-hint">
                <span class="material-symbols-outlined">filter_alt_off</span>
                <span>No active experts matching filter: <b>${currentExpertFilter}</b></span>
            </div>`;
        return;
    }

    // Render up to 60 top experts
    const maxShow = Math.min(filtered.length, 60);
    let html = '';

    for (let i = 0; i < maxShow; i++) {
        const exp = filtered[i];
        const catClass = getCategoryClass(exp.category);
        const hitPctClamped = Math.min(Math.max(exp.hit_pct, 0.5), 100);
        const tierClass = exp.tier || 'l1';
        const locLabel = exp.location || (tierClass === 'l1' ? 'L1 (VRAM)' : tierClass === 'l2' ? 'L2 (DRAM)' : 'SSD (Disk)');

        let pillsHtml = '';
        if (exp.top_tokens && exp.top_tokens.length > 0) {
            pillsHtml = '<div class="token-pills-row">';
            for (const t of exp.top_tokens.slice(0, 6)) {
                const cleanTok = t.token.replace(/\n/g, '\\n').replace(/\t/g, '\\t');
                const safeTok = cleanTok.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
                pillsHtml += `<span class="token-pill" title="Token: ${safeTok} (${t.count} hits)">${safeTok}</span>`;
            }
            pillsHtml += '</div>';
        }

        html += `
            <div class="expert-card" data-cat="${exp.category}" data-tier="${tierClass}">
                <div class="expert-card-top">
                    <div class="expert-badge">
                        <span class="expert-rank-num">#${exp.rank || (i + 1)}</span>
                        <span class="layer-tag">L${exp.layer}</span>
                        <span>·</span>
                        <span>E${exp.expert_id}</span>
                    </div>
                    <div style="display:flex; gap:4px; align-items:center;">
                        <span class="tier-badge ${tierClass}" title="Preload Residency: ${locLabel}">${locLabel}</span>
                        <span class="cat-badge ${catClass}">${exp.category}</span>
                    </div>
                </div>
                <div class="hit-bar-wrapper">
                    <div class="hit-bar-labels">
                        <span>Hit Rate: <b>${exp.hit_pct.toFixed(1)}%</b></span>
                        <span class="hit-count">${exp.count.toLocaleString()} hits</span>
                    </div>
                    <div class="hit-bar-bg">
                        <div class="hit-bar-fill" style="width: ${hitPctClamped}%;"></div>
                    </div>
                </div>
                ${pillsHtml}
            </div>
        `;
    }

    listContainer.innerHTML = html;
}

function initExpertProfileUI() {
    const filterContainer = document.getElementById('expert-category-filters');
    if (filterContainer) {
        filterContainer.addEventListener('click', (e) => {
            const btn = e.target.closest('.cat-filter-btn');
            if (!btn) return;

            filterContainer.querySelectorAll('.cat-filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            currentFilterType = btn.getAttribute('data-filter-type') || 'cat';
            currentExpertFilter = btn.getAttribute('data-filter') || btn.getAttribute('data-cat') || 'all';
            if (currentExpertProfileData) {
                renderExpertProfileUI(currentExpertProfileData);
            }
        });
    }

    // Initial fetch
    fetchExpertProfile();
}

function downloadExpertProfileJson() {
    if (!currentExpertProfileData) {
        alert('No expert profile data loaded yet. Run inference with --track first.');
        return;
    }
    const blob = new Blob([JSON.stringify(currentExpertProfileData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `expert_profile_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Sidebar toggle
const menuBtn = document.querySelector('.menu-btn');
const sidebar = document.querySelector('.sidebar');
if (menuBtn && sidebar) {
    menuBtn.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
    });
}

// Run initialization on DOM load
document.addEventListener('DOMContentLoaded', () => {
    initPreviewPanel();
    initExpertProfileUI();
});
