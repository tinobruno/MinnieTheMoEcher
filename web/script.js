const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const messagesContainer = document.getElementById('messages-container');
const welcomeScreen = document.getElementById('welcome-screen');

if (typeof marked !== 'undefined' && marked.setOptions) {
    marked.setOptions({
        gfm: true,
        breaks: true
    });
}

const tempSlider = document.getElementById('temp-slider');
const tempVal = document.getElementById('temp-val');
const tokensInput = document.getElementById('tokens-input');
const reasoningEffort = document.getElementById('reasoning-effort');
const thinkingBudget = document.getElementById('thinking-budget');
const thinkingEnabled = document.getElementById('thinking-enabled');
const webRetrievalEnabled = document.getElementById('web-retrieval-enabled');
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
    fetch(apiUrl, { method: 'POST' }).catch(() => { });
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

function openAgenticSettingsTab() {
    openPreviewPanel();
    switchPreviewTab('tab-agentic');
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

function loadHtmlIntoPreview(htmlCode, autoSwitchTab = true) {
    currentHtmlCode = htmlCode || '';
    if (!isPreviewOpen) openPreviewPanel();

    if (previewCodeEditor) {
        previewCodeEditor.value = currentHtmlCode;
        updateEditorLineNumbers();
    }

    clearConsoleLogs();

    // Direct YouTube video player detection for 100% reliable hardware-accelerated playback
    const ytMatch = currentHtmlCode.match(/(?:youtube-nocookie\.com\/embed\/|youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})/i);
    if (ytMatch && (currentHtmlCode.includes('youtube-nocookie.com') || currentHtmlCode.includes('YouTube Video') || currentHtmlCode.includes('yt-container') || currentHtmlCode.includes('player'))) {
        const videoId = ytMatch[1];
        if (previewIframe) {
            previewIframe.removeAttribute('srcdoc');
            previewIframe.setAttribute('allow', 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share');
            previewIframe.src = `https://www.youtube-nocookie.com/embed/${videoId}?autoplay=1&enablejsapi=1&rel=0`;
        }
    } else {
        renderPreviewIframe(currentHtmlCode);
    }

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
        if (isHtmlContent(code, lang)) {
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
        if (isHtmlContent(code, lang)) {
            return code;
        }
    }
    if (currentHtmlCode && currentHtmlCode.trim().length > 0) {
        return currentHtmlCode;
    }
    return '';
}

// Extract HTML snippet specifically generated in the current assistant message
function getTurnHtmlCode(assistantMsgEl, rawText) {
    if (assistantMsgEl) {
        const codeBlocks = assistantMsgEl.querySelectorAll('pre code');
        for (let i = codeBlocks.length - 1; i >= 0; i--) {
            const code = codeBlocks[i].textContent || '';
            const lang = (codeBlocks[i].className || '').toLowerCase();
            if (isHtmlContent(code, lang)) {
                return code;
            }
        }
    }
    if (isFullHtmlDocument(rawText)) {
        return rawText;
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

    previewIframe.removeAttribute('src');
    previewIframe.setAttribute('allow', 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share');
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

function isFullHtmlDocument(text) {
    if (!text || typeof text !== 'string') return false;
    const trimmed = text.trim();
    if (trimmed.includes('tool-activity-block') || trimmed.includes('<tool_response>') || trimmed.includes('tool-activity-card')) return false;
    return (/^<!doctype\s+html/i.test(trimmed) || /^<html[\s>]/i.test(trimmed));
}

function createYouTubePlayerHtml(videoId, title = 'YouTube Video') {
    return `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${escapeHtml(title)}</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 20px; background: #0f0f0f; color: #f1f1f1; line-height: 1.5; }
    .yt-container { max-width: 900px; margin: 0 auto; }
    .video-wrapper { position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.7); background: #000; margin-bottom: 18px; border: 1px solid rgba(255,255,255,0.12); }
    .video-wrapper iframe, .video-wrapper #player { position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none; }
    .video-title { font-size: 1.35rem; font-weight: 600; margin-bottom: 8px; color: #ffffff; }
    .video-actions a { display: inline-flex; align-items: center; gap: 6px; color: #fff; background: rgba(255,255,255,0.12); padding: 6px 14px; border-radius: 18px; text-decoration: none; font-size: 0.85rem; }
  </style>
</head>
<body>
  <div class="yt-container">
    <div class="video-wrapper">
      <iframe id="player" src="https://www.youtube-nocookie.com/embed/${videoId}?autoplay=1&enablejsapi=1&rel=0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen style="position:absolute;top:0;left:0;width:100%;height:100%;border:none;"></iframe>
    </div>
    <div class="video-title">${escapeHtml(title)}</div>
    <div class="video-actions">
      <a href="https://www.youtube.com/watch?v=${videoId}" target="_blank">Watch on YouTube &#x2197;</a>
    </div>
  </div>
  <script src="https://www.youtube.com/iframe_api"></script>
  <script>
    var player;
    function onYouTubeIframeAPIReady() {
      try {
        player = new YT.Player('player', {
          events: {
            'onReady': function(e) {
              try {
                e.target.unMute();
                e.target.setVolume(100);
                e.target.playVideo();
              } catch(err) {}
            }
          }
        });
      } catch(e) {}
    }
  </script>
</body>
</html>`;
}

// Markdown rendering and code-block post-processing
function renderMarkdownContent(rawText, containerElement) {
    containerElement.innerHTML = marked.parse(rawText);

    // Decorate code blocks
    const codeBlocks = containerElement.querySelectorAll('pre > code');
    let hasPreviewableHtml = false;

    codeBlocks.forEach(codeEl => {
        const preEl = codeEl.parentElement;
        const codeContent = codeEl.textContent || '';
        const rawClass = codeEl.className || '';
        const match = /language-(\w+)/.exec(rawClass);
        const lang = match ? match[1] : '';

        const isHtmlCandidate = isHtmlContent(codeContent, lang);
        if (isHtmlCandidate) {
            hasPreviewableHtml = true;
        }

        const wrapper = document.createElement('div');
        wrapper.className = 'code-block-wrapper';

        const header = document.createElement('div');
        header.className = 'code-block-header';

        const langDiv = document.createElement('div');
        langDiv.className = 'code-block-lang';
        langDiv.textContent = (lang || 'code').toUpperCase();

        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'code-block-actions';

        const previewBtn = document.createElement('button');
        previewBtn.className = 'code-action-btn' + (isHtmlCandidate ? ' preview-btn-highlight' : '');
        previewBtn.innerHTML = `<span class="material-symbols-outlined btn-icon">${isHtmlCandidate ? 'play_circle' : 'preview'}</span> Preview`;
        previewBtn.title = 'Test and render this snippet in the HTML preview panel';
        previewBtn.addEventListener('click', () => {
            loadHtmlIntoPreview(codeContent, true);
        });

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
        actionsDiv.appendChild(previewBtn);
        actionsDiv.appendChild(copyBtn);

        header.appendChild(langDiv);
        header.appendChild(actionsDiv);

        preEl.parentNode.insertBefore(wrapper, preEl);
        wrapper.appendChild(header);
        wrapper.appendChild(preEl);
    });

    // Check if raw message (without code blocks or enclosing entire text) is an explicit full HTML document
    if (!hasPreviewableHtml && isFullHtmlDocument(rawText)) {
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
    // Intercept rendered YouTube links in messages so clicking them plays directly in the HTML Preview Panel
    const links = containerElement.querySelectorAll('a[href]');
    links.forEach(a => {
        const href = a.getAttribute('href') || '';
        const m = href.match(/(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})/i);
        if (m) {
            a.title = 'Click to play in HTML Preview Panel';
            a.addEventListener('click', (e) => {
                e.preventDefault();
                const videoId = m[1];
                const playerHtml = createYouTubePlayerHtml(videoId, a.textContent || 'YouTube Video');
                loadHtmlIntoPreview(playerHtml, true);
            });
        }
    });

    // Detect YouTube URLs in message and auto-load interactive player preview
    const ytMatch = rawText.match(/(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})/i);
    if (ytMatch) {
        const videoId = ytMatch[1];
        const ytDocId = 'yt_' + videoId;
        if (!retrievedDocsStore[ytDocId]) {
            const ytHtml = createYouTubePlayerHtml(videoId, 'YouTube Video');
            addRetrievedDocument({
                id: ytDocId,
                url: `https://www.youtube.com/watch?v=${videoId}`,
                title: 'YouTube Video',
                html: ytHtml,
                snippet: 'YouTube interactive player'
            });
        }
    }

    updateChatbarPreviewButtonVisibility();
}

// ============================================================================
// Retrieved Documents Store & Management
// ============================================================================

let retrievedDocuments = [];
let retrievedDocsStore = {};

function addRetrievedDocument(doc) {
    if (!doc || !doc.url) return;
    const docId = doc.id || ('doc_' + Date.now() + '_' + Math.floor(Math.random() * 1000));

    // Auto-generate YouTube player HTML if this document is a YouTube video
    let docHtml = doc.html || '';
    const ytMatch = (doc.url + ' ' + (doc.title || '')).match(/(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})/i);
    if (ytMatch && (!docHtml || (!docHtml.includes('<iframe') && !docHtml.includes('YT.Player')))) {
        docHtml = createYouTubePlayerHtml(ytMatch[1], doc.title || 'YouTube Video');
    }

    // Check if already present by url
    const existingIdx = retrievedDocuments.findIndex(d => d.url === doc.url);
    const docObj = {
        id: docId,
        url: doc.url,
        title: doc.title || extractDomain(doc.url) || 'Retrieved Web Page',
        html: docHtml,
        snippet: doc.snippet || '',
        timeStr: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    if (existingIdx >= 0) {
        retrievedDocuments[existingIdx] = docObj;
    } else {
        retrievedDocuments.push(docObj);
    }
    retrievedDocsStore[docId] = docObj;
    if (doc.id) retrievedDocsStore[doc.id] = docObj;
    retrievedDocsStore[doc.url] = docObj;
    if (ytMatch) {
        retrievedDocsStore['yt_' + ytMatch[1]] = docObj;
    }

    renderRetrievedDocsList();
}

function extractDomain(url) {
    try {
        const u = new URL(url);
        return u.hostname.replace('www.', '');
    } catch (e) {
        return url;
    }
}

function renderRetrievedDocsList() {
    const container = document.getElementById('retrieved-docs-container');
    const badge = document.getElementById('retrieved-docs-badge');
    const countLabel = document.getElementById('retrieved-doc-count-label');
    if (!container) return;

    if (retrievedDocuments.length === 0) {
        container.innerHTML = `
            <div class="retrieved-empty-state" id="retrieved-empty-state">
                <span class="material-symbols-outlined empty-icon">travel_explore</span>
                <h3>No Documents Retrieved</h3>
                <p>Web pages and search results retrieved during chat turns will appear here as live interactive thumbnails.</p>
            </div>
        `;
        if (badge) { badge.style.display = 'none'; badge.textContent = '0'; }
        if (countLabel) countLabel.textContent = '0 Documents Retrieved';
        return;
    }

    if (badge) {
        badge.style.display = 'inline-block';
        badge.textContent = String(retrievedDocuments.length);
    }
    if (countLabel) {
        countLabel.textContent = `${retrievedDocuments.length} Document${retrievedDocuments.length > 1 ? 's' : ''} Retrieved`;
    }

    let html = '';
    for (let i = 0; i < retrievedDocuments.length; i++) {
        const doc = retrievedDocuments[i];
        const safeTitle = escapeHtml(doc.title);
        const safeUrl = escapeHtml(doc.url);
        const safeSnippet = escapeHtml(doc.snippet);
        const safeSrcdoc = escapeHtml(doc.html || `<!DOCTYPE html><html><body style="font-family:-apple-system,BlinkMacSystemFont,sans-serif;padding:16px;color:#333;"><h3>${safeTitle}</h3><p>${safeSnippet}</p></body></html>`);

        const isDocYouTube = (doc.url && (doc.url.includes('youtube.com') || doc.url.includes('youtu.be'))) || (doc.html && doc.html.includes('youtube-nocookie.com/embed'));
        let miniContentHtml = '';
        if (isDocYouTube) {
            let ytId = '';
            const ytMatch = ((doc.url || '') + ' ' + (doc.html || '')).match(/(?:v=|youtu\.be\/|embed\/)([a-zA-Z0-9_-]{11})/);
            if (ytMatch) ytId = ytMatch[1];

            if (ytId) {
                miniContentHtml = `
                    <div style="width:100%;height:100%;position:relative;background:#000;display:flex;align-items:center;justify-content:center;">
                        <img src="https://img.youtube.com/vi/${ytId}/mqdefault.jpg" alt="${safeTitle}" style="width:100%;height:100%;object-fit:cover;" />
                        <div style="position:absolute;width:40px;height:40px;border-radius:50%;background:rgba(255,0,0,0.9);display:flex;align-items:center;justify-content:center;color:#fff;box-shadow:0 4px 12px rgba(0,0,0,0.6);">
                            <span class="material-symbols-outlined" style="font-size:26px;margin-left:2px;">play_arrow</span>
                        </div>
                    </div>
                `;
            } else {
                miniContentHtml = `
                    <div style="width:100%;height:100%;background:#18181b;display:flex;align-items:center;justify-content:center;color:#ef4444;">
                        <span class="material-symbols-outlined" style="font-size:36px;">smart_display</span>
                    </div>
                `;
            }
        } else {
            miniContentHtml = `
                <div class="retrieved-mini-snapshot" style="width:100%;height:100%;background:linear-gradient(135deg,rgba(30,41,59,0.7),rgba(15,23,42,0.9));display:flex;flex-direction:column;justify-content:center;padding:14px;box-sizing:border-box;">
                    <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px;">
                        <span class="material-symbols-outlined" style="font-size:18px;color:#38bdf8;">language</span>
                        <span style="font-size:11px;font-weight:600;color:#94a3b8;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escapeHtml(extractDomain(doc.url))}</span>
                    </div>
                    <div style="font-size:11px;color:#cbd5e1;line-height:1.4;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;">
                        ${safeSnippet || safeTitle}
                    </div>
                </div>
            `;
        }

        html += `
            <div class="retrieved-doc-card" onclick="openDocInFullPreview('${doc.id}', event)" title="Click to open in HTML Preview">
                <div class="retrieved-doc-header">
                    <div class="retrieved-doc-title-row">
                        <span class="material-symbols-outlined retrieved-doc-icon">public</span>
                        <div class="retrieved-doc-meta">
                            <div class="retrieved-doc-title">${safeTitle}</div>
                            <a href="${safeUrl}" target="_blank" class="retrieved-doc-url" onclick="event.stopPropagation()">${safeUrl}</a>
                        </div>
                    </div>
                    <div class="retrieved-doc-actions">
                        <span class="retrieved-doc-time">${doc.timeStr}</span>
                        <button type="button" class="retrieved-preview-btn" onclick="openDocInFullPreview('${doc.id}', event)" title="Open in Full Preview">
                            <span class="material-symbols-outlined">visibility</span>
                            <span>Preview</span>
                        </button>
                    </div>
                </div>
                <div class="retrieved-mini-preview-wrap">
                    ${miniContentHtml}
                    <div class="retrieved-mini-overlay">
                        <span class="material-symbols-outlined">fullscreen</span>
                        <span>Click to Open in HTML Preview</span>
                    </div>
                </div>
                ${safeSnippet ? `<div class="retrieved-doc-snippet">${safeSnippet}</div>` : ''}
            </div>
        `;
    }
    container.innerHTML = html;
}

function openDocInFullPreview(docId, event) {
    if (event) {
        event.stopPropagation();
    }
    const doc = retrievedDocsStore[docId];
    if (!doc) return;

    // A. YouTube video player embed or direct video player HTML
    const isDocYouTube = (doc.url && (doc.url.includes('youtube.com') || doc.url.includes('youtu.be'))) || (doc.html && (doc.html.includes('youtube-nocookie.com/embed') || doc.html.includes('youtube.com')));
    if (isDocYouTube) {
        let ytMatch = (doc.url || '').match(/(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})/i);
        if (!ytMatch && doc.html) {
            ytMatch = doc.html.match(/(?:youtube-nocookie\.com\/embed\/|youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})/i);
        }
        const videoId = ytMatch ? ytMatch[1] : '';
        const playerHtml = videoId ? createYouTubePlayerHtml(videoId, doc.title || 'YouTube Video') : (doc.html || '');
        if (playerHtml) {
            loadHtmlIntoPreview(playerHtml, true);
            return;
        }
    }

    // B. External Web URLs (such as LinkedIn, GitHub, Google, Wikipedia, etc.) -> ALWAYS load via Moecher Reverse Proxy
    if (doc.url && (doc.url.startsWith('http://') || doc.url.startsWith('https://'))) {
        if (!isPreviewOpen) openPreviewPanel();
        if (previewIframe) {
            previewIframe.removeAttribute('srcdoc');
            previewIframe.setAttribute('allow', 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share');
            previewIframe.src = `${getApiBase()}/api/proxy?url=${encodeURIComponent(doc.url)}`;
        }
        if (previewCodeEditor) {
            previewCodeEditor.value = doc.html || `<!-- Live proxied page from: ${doc.url} -->\n<iframe src="${getApiBase()}/api/proxy?url=${encodeURIComponent(doc.url)}" style="width:100%;height:100%;border:none;"></iframe>`;
            updateEditorLineNumbers();
        }
        clearConsoleLogs();
        if (previewEmptyState) previewEmptyState.classList.add('hidden');
        if (docStatusBadge) {
            docStatusBadge.textContent = 'Active (Proxied Live)';
            docStatusBadge.classList.remove('modified');
        }
        switchPreviewTab('tab-preview');
        return;
    }

    // C. User-generated / local HTML documents
    if (doc.html && doc.html.trim().length > 0) {
        loadHtmlIntoPreview(doc.html, true);
        return;
    }

    const htmlToLoad = `<!DOCTYPE html><html><head><meta charset="UTF-8"><title>${escapeHtml(doc.title)}</title><style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;padding:30px;line-height:1.6;max-width:800px;margin:auto;color:#202124;}</style></head><body><h1>${escapeHtml(doc.title)}</h1><p><a href="${escapeHtml(doc.url)}" target="_blank">${escapeHtml(doc.url)}</a></p><hr/><p>${escapeHtml(doc.snippet)}</p></body></html>`;
    loadHtmlIntoPreview(htmlToLoad, true);
}

function clearRetrievedDocs() {
    retrievedDocuments = [];
    retrievedDocsStore = {};
    renderRetrievedDocsList();
}

function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

// ============================================================================
// Agentic Settings & Guardrails Management
// ============================================================================

let agenticSettings = {
    timeoutSec: 60,
    boundaryEnforced: true,
    requireAuth: true,
    authorizedPaths: [],
    maxTurns: 8,
    compactToolOutputs: true,
    omitPastReasoning: true,
    searchProvider: 'tavily',
    tavilyApiKey: '',
    searxngUrl: 'https://searx.be',
    braveApiKey: '',
    serperApiKey: '',
    googleApiKey: '',
    googleCx: '',
    fastMediaSearch: true,
    tools: {
        read_file: true,
        write_file: true,
        edit_file: true,
        execute_command: true,
        web_search: true,
        google_search: true,
        fetch_url: true
    },
    workspaceDir: ''
};

let currentPendingAuth = null;

function isMediaSearchQuery(query) {
    if (!query) return false;
    const q = query.toLowerCase();
    const kw = [
        "youtube", "youtu.be", "video", "videos", "song", "songs", "music", "play", "listen",
        "track", "tracks", "album", "clip", "clips", "audio", "soundtrack", "canto", "canzone",
        "musica", "suona", "ascolta", "videoclip"
    ];
    return kw.some(k => q.includes(k));
}

function loadAgenticSettings() {
    try {
        const savedTimeout = localStorage.getItem('moecher_agentic_timeout');
        if (savedTimeout) agenticSettings.timeoutSec = parseInt(savedTimeout, 10) || 60;

        const savedBoundary = localStorage.getItem('moecher_agentic_boundary');
        if (savedBoundary !== null) agenticSettings.boundaryEnforced = savedBoundary === 'true';

        const savedAuth = localStorage.getItem('moecher_agentic_require_auth');
        if (savedAuth !== null) agenticSettings.requireAuth = savedAuth === 'true';

        const savedMaxTurns = localStorage.getItem('moecher_agentic_max_turns');
        if (savedMaxTurns !== null) agenticSettings.maxTurns = parseInt(savedMaxTurns, 10);

        const savedCompactTools = localStorage.getItem('moecher_agentic_compact_tools');
        if (savedCompactTools !== null) agenticSettings.compactToolOutputs = savedCompactTools === 'true';

        const savedOmitReasoning = localStorage.getItem('moecher_agentic_omit_reasoning');
        if (savedOmitReasoning !== null) agenticSettings.omitPastReasoning = savedOmitReasoning === 'true';

        const savedProvider = localStorage.getItem('moecher_search_provider');
        if (savedProvider) agenticSettings.searchProvider = savedProvider;

        const savedTavilyKey = localStorage.getItem('moecher_tavily_api_key');
        if (savedTavilyKey) agenticSettings.tavilyApiKey = savedTavilyKey;

        const savedSearxngUrl = localStorage.getItem('moecher_searxng_url');
        if (savedSearxngUrl) agenticSettings.searxngUrl = savedSearxngUrl;

        const savedBraveKey = localStorage.getItem('moecher_brave_api_key');
        if (savedBraveKey) agenticSettings.braveApiKey = savedBraveKey;

        const savedSerperKey = localStorage.getItem('moecher_serper_api_key');
        if (savedSerperKey) agenticSettings.serperApiKey = savedSerperKey;

        const savedGoogleKey = localStorage.getItem('moecher_google_api_key');
        if (savedGoogleKey) agenticSettings.googleApiKey = savedGoogleKey;

        const savedGoogleCx = localStorage.getItem('moecher_google_cx');
        if (savedGoogleCx) agenticSettings.googleCx = savedGoogleCx;

        const savedFastMedia = localStorage.getItem('moecher_fast_media_search');
        if (savedFastMedia !== null) agenticSettings.fastMediaSearch = savedFastMedia === 'true';
        else agenticSettings.fastMediaSearch = true;

        const savedPaths = localStorage.getItem('moecher_agentic_auth_paths');
        if (savedPaths) {
            try { agenticSettings.authorizedPaths = JSON.parse(savedPaths); } catch (e) { }
        }

        const savedTools = localStorage.getItem('moecher_agentic_tools');
        if (savedTools) {
            try { Object.assign(agenticSettings.tools, JSON.parse(savedTools)); } catch (e) { }
        }
    } catch (e) {
        console.warn('Could not load agentic settings from localStorage', e);
    }
}

function saveAgenticSettings() {
    try {
        localStorage.setItem('moecher_agentic_timeout', agenticSettings.timeoutSec);
        localStorage.setItem('moecher_agentic_boundary', agenticSettings.boundaryEnforced);
        localStorage.setItem('moecher_agentic_require_auth', agenticSettings.requireAuth);
        localStorage.setItem('moecher_agentic_max_turns', agenticSettings.maxTurns);
        localStorage.setItem('moecher_agentic_compact_tools', agenticSettings.compactToolOutputs);
        localStorage.setItem('moecher_agentic_omit_reasoning', agenticSettings.omitPastReasoning);
        localStorage.setItem('moecher_search_provider', agenticSettings.searchProvider || 'tavily');
        localStorage.setItem('moecher_tavily_api_key', agenticSettings.tavilyApiKey || '');
        localStorage.setItem('moecher_searxng_url', agenticSettings.searxngUrl || 'https://searx.be');
        localStorage.setItem('moecher_brave_api_key', agenticSettings.braveApiKey || '');
        localStorage.setItem('moecher_serper_api_key', agenticSettings.serperApiKey || '');
        localStorage.setItem('moecher_google_api_key', agenticSettings.googleApiKey || '');
        localStorage.setItem('moecher_google_cx', agenticSettings.googleCx || '');
        localStorage.setItem('moecher_fast_media_search', agenticSettings.fastMediaSearch !== false);
        localStorage.setItem('moecher_agentic_auth_paths', JSON.stringify(agenticSettings.authorizedPaths));
        localStorage.setItem('moecher_agentic_tools', JSON.stringify(agenticSettings.tools));
    } catch (e) {
        console.warn('Could not save agentic settings to localStorage', e);
    }
}

function toggleKeyVisibility(inputId, iconId) {
    const input = document.getElementById(inputId);
    const icon = document.getElementById(iconId);
    if (!input || !icon) return;
    if (input.type === 'password') {
        input.type = 'text';
        icon.textContent = 'visibility_off';
    } else {
        input.type = 'password';
        icon.textContent = 'visibility';
    }
}

function toggleGoogleKeyVisibility() {
    toggleKeyVisibility('google-api-key-input', 'google-key-visibility-icon');
}

function updateSearchProviderStatusBadge(configured) {
    const badge = document.getElementById('search-provider-status-badge');
    if (!badge) return;
    const provider = agenticSettings.searchProvider || 'tavily';

    let isReady = false;
    let label = 'Configured';
    if (provider === 'searxng') {
        isReady = !!(agenticSettings.searxngUrl && agenticSettings.searxngUrl.trim());
        label = isReady ? 'Zero-Config Ready' : 'URL Required';
    } else if (provider === 'tavily') {
        isReady = !!(agenticSettings.tavilyApiKey && agenticSettings.tavilyApiKey.trim());
        label = isReady ? 'Tavily Ready' : 'Key Required';
    } else if (provider === 'brave') {
        isReady = !!(agenticSettings.braveApiKey && agenticSettings.braveApiKey.trim());
        label = isReady ? 'Brave Ready' : 'Key Required';
    } else if (provider === 'serper') {
        isReady = !!(agenticSettings.serperApiKey && agenticSettings.serperApiKey.trim());
        label = isReady ? 'Serper Ready' : 'Key Required';
    } else if (provider === 'google') {
        isReady = !!(agenticSettings.googleApiKey && agenticSettings.googleCx);
        label = isReady ? 'Google Ready' : 'Key/CX Required';
    }

    if (configured !== undefined && typeof configured === 'boolean') {
        isReady = configured;
    }

    if (isReady) {
        badge.textContent = label;
        badge.style.background = 'rgba(34, 197, 94, 0.2)';
        badge.style.color = '#4ade80';
    } else {
        badge.textContent = label;
        badge.style.background = 'rgba(239, 68, 68, 0.2)';
        badge.style.color = '#f87171';
    }
}

function updateGoogleSearchStatusBadge(isConfigured) {
    updateSearchProviderStatusBadge(isConfigured);
}

function onSearchProviderChange(targetProvider) {
    const select = document.getElementById('search-provider-select');
    const provider = targetProvider || (select ? select.value : 'tavily');
    agenticSettings.searchProvider = provider;
    if (select && select.value !== provider) {
        select.value = provider;
    }

    const providers = ['tavily', 'searxng', 'brave', 'serper', 'google'];
    providers.forEach(p => {
        const sec = document.getElementById(`provider-section-${p}`);
        if (sec) {
            sec.style.display = (p === provider) ? 'block' : 'none';
        }
    });

    saveAgenticSettings();
    updateSearchProviderStatusBadge();
}

function saveSearchProviderSettingsUI() {
    const select = document.getElementById('search-provider-select');
    const tavilyInput = document.getElementById('tavily-api-key-input');
    const searxngInput = document.getElementById('searxng-url-input');
    const braveInput = document.getElementById('brave-api-key-input');
    const serperInput = document.getElementById('serper-api-key-input');
    const googleKeyInput = document.getElementById('google-api-key-input');
    const googleCxInput = document.getElementById('google-cx-input');
    const fastMediaToggle = document.getElementById('fast-media-search-toggle');
    const saveMsg = document.getElementById('search-provider-save-msg');

    const provider = select ? select.value : (agenticSettings.searchProvider || 'tavily');
    const tavilyKey = tavilyInput ? tavilyInput.value.trim() : '';
    const searxngUrl = searxngInput ? searxngInput.value.trim() : 'https://searx.be';
    const braveKey = braveInput ? braveInput.value.trim() : '';
    const serperKey = serperInput ? serperInput.value.trim() : '';
    const googleKey = googleKeyInput ? googleKeyInput.value.trim() : '';
    const googleCx = googleCxInput ? googleCxInput.value.trim() : '';
    const fastMedia = fastMediaToggle ? fastMediaToggle.checked : (agenticSettings.fastMediaSearch !== false);

    agenticSettings.searchProvider = provider;
    agenticSettings.tavilyApiKey = tavilyKey;
    agenticSettings.searxngUrl = searxngUrl;
    agenticSettings.braveApiKey = braveKey;
    agenticSettings.serperApiKey = serperKey;
    agenticSettings.googleApiKey = googleKey;
    agenticSettings.googleCx = googleCx;
    agenticSettings.fastMediaSearch = fastMedia;

    saveAgenticSettings();

    // Sync to backend server
    fetch(`${getApiBase()}/api/settings/search_provider`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            search_provider: provider,
            tavily_api_key: tavilyKey,
            searxng_url: searxngUrl,
            brave_api_key: braveKey,
            serper_api_key: serperKey,
            google_search_api_key: googleKey,
            google_search_cx: googleCx,
            fast_media_search: fastMedia
        })
    })
    .then(r => r.json())
    .then(data => {
        updateSearchProviderStatusBadge(data.configured);
        if (saveMsg) {
            saveMsg.style.display = 'inline';
            saveMsg.textContent = 'Saved to server & browser!';
            setTimeout(() => { saveMsg.style.display = 'none'; }, 3000);
        }
    })
    .catch(() => {
        updateSearchProviderStatusBadge();
        if (saveMsg) {
            saveMsg.style.display = 'inline';
            saveMsg.textContent = 'Saved to browser!';
            setTimeout(() => { saveMsg.style.display = 'none'; }, 3000);
        }
    });
}

function syncFastMediaSearchToggle(enabled) {
    agenticSettings.fastMediaSearch = !!enabled;
    const sidebarToggle = document.getElementById('sidebar-fast-media-search');
    const panelToggle = document.getElementById('fast-media-search-toggle');
    if (sidebarToggle) sidebarToggle.checked = !!enabled;
    if (panelToggle) panelToggle.checked = !!enabled;
    saveAgenticSettings();

    // Sync to backend server
    fetch(`${getApiBase()}/api/settings/search_provider`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            fast_media_search: !!enabled
        })
    }).catch(() => {});
}

function onFastMediaSearchToggleChange() {
    const panelToggle = document.getElementById('fast-media-search-toggle');
    syncFastMediaSearchToggle(panelToggle ? panelToggle.checked : true);
}

function saveGoogleSearchCredentialsUI() {
    saveSearchProviderSettingsUI();
}

function fetchSearchProviderSettings() {
    fetch(`${getApiBase()}/api/settings/search_provider`)
        .then(r => r.json())
        .then(data => {
            if (data) {
                if (data.search_provider) {
                    agenticSettings.searchProvider = data.search_provider;
                }
                if (data.searxng_url) {
                    agenticSettings.searxngUrl = data.searxng_url;
                }
                if (data.google_search_cx) {
                    agenticSettings.googleCx = data.google_search_cx;
                }
                if (data.fast_media_search !== undefined) {
                    agenticSettings.fastMediaSearch = !!data.fast_media_search;
                }

                // Update UI inputs
                const select = document.getElementById('search-provider-select');
                if (select && data.search_provider) select.value = data.search_provider;

                const tavilyInput = document.getElementById('tavily-api-key-input');
                if (tavilyInput && agenticSettings.tavilyApiKey) tavilyInput.value = agenticSettings.tavilyApiKey;

                const searxngInput = document.getElementById('searxng-url-input');
                if (searxngInput && (data.searxng_url || agenticSettings.searxngUrl)) {
                    searxngInput.value = data.searxng_url || agenticSettings.searxngUrl;
                }

                const braveInput = document.getElementById('brave-api-key-input');
                if (braveInput && agenticSettings.braveApiKey) braveInput.value = agenticSettings.braveApiKey;

                const serperInput = document.getElementById('serper-api-key-input');
                if (serperInput && agenticSettings.serperApiKey) serperInput.value = agenticSettings.serperApiKey;

                const googleKeyInput = document.getElementById('google-api-key-input');
                if (googleKeyInput && agenticSettings.googleApiKey) googleKeyInput.value = agenticSettings.googleApiKey;

                const googleCxInput = document.getElementById('google-cx-input');
                if (googleCxInput && (data.google_search_cx || agenticSettings.googleCx)) {
                    googleCxInput.value = data.google_search_cx || agenticSettings.googleCx;
                }

                const fastMediaToggle = document.getElementById('fast-media-search-toggle');
                if (fastMediaToggle) fastMediaToggle.checked = agenticSettings.fastMediaSearch !== false;
                const sidebarFastMediaToggle = document.getElementById('sidebar-fast-media-search');
                if (sidebarFastMediaToggle) sidebarFastMediaToggle.checked = agenticSettings.fastMediaSearch !== false;

                onSearchProviderChange(agenticSettings.searchProvider);
                updateSearchProviderStatusBadge(data.configured);
            }
        })
        .catch(() => {
            const fastMediaToggle = document.getElementById('fast-media-search-toggle');
            if (fastMediaToggle) fastMediaToggle.checked = agenticSettings.fastMediaSearch !== false;
            const sidebarFastMediaToggle = document.getElementById('sidebar-fast-media-search');
            if (sidebarFastMediaToggle) sidebarFastMediaToggle.checked = agenticSettings.fastMediaSearch !== false;
            onSearchProviderChange(agenticSettings.searchProvider);
            updateSearchProviderStatusBadge();
        });
}

function fetchGoogleSearchSettings() {
    fetchSearchProviderSettings();
}

function initAgenticSettingsUI() {
    loadAgenticSettings();

    // Timeout slider & input sync
    const slider = document.getElementById('agentic-timeout-slider');
    const input = document.getElementById('agentic-timeout-input');
    if (slider && input) {
        slider.value = agenticSettings.timeoutSec;
        input.value = agenticSettings.timeoutSec;

        slider.addEventListener('input', (e) => {
            const val = parseInt(e.target.value, 10);
            input.value = val;
            agenticSettings.timeoutSec = val;
            saveAgenticSettings();
        });

        input.addEventListener('input', (e) => {
            let val = parseInt(e.target.value, 10);
            if (isNaN(val) || val < 5) val = 5;
            if (val > 600) val = 600;
            slider.value = Math.min(val, 300);
            agenticSettings.timeoutSec = val;
            saveAgenticSettings();
        });
    }

    // Context Optimization UI (Right Panel & Left Sidebar sync)
    const maxTurnsSlider = document.getElementById('agentic-max-turns-slider');
    const maxTurnsInput = document.getElementById('agentic-max-turns-input');
    const sideMaxTurnsSlider = document.getElementById('sidebar-max-turns-slider');
    const sideMaxTurnsVal = document.getElementById('sidebar-max-turns-val');

    const updateMaxTurnsUI = (val) => {
        agenticSettings.maxTurns = val;
        if (maxTurnsSlider) maxTurnsSlider.value = Math.min(val, 32);
        if (maxTurnsInput) maxTurnsInput.value = val;
        if (sideMaxTurnsSlider) sideMaxTurnsSlider.value = Math.min(val, 32);
        if (sideMaxTurnsVal) sideMaxTurnsVal.textContent = val === 0 ? 'Unlimited' : val;
        saveAgenticSettings();
    };

    updateMaxTurnsUI(agenticSettings.maxTurns);

    if (maxTurnsSlider) {
        maxTurnsSlider.addEventListener('input', (e) => updateMaxTurnsUI(parseInt(e.target.value, 10)));
    }
    if (maxTurnsInput) {
        maxTurnsInput.addEventListener('input', (e) => {
            let val = parseInt(e.target.value, 10);
            if (isNaN(val) || val < 0) val = 0;
            if (val > 64) val = 64;
            updateMaxTurnsUI(val);
        });
    }
    if (sideMaxTurnsSlider) {
        sideMaxTurnsSlider.addEventListener('input', (e) => updateMaxTurnsUI(parseInt(e.target.value, 10)));
    }

    // Compact Tools Toggles
    const compactToolsToggle = document.getElementById('agentic-compact-tools');
    const sideCompactToolsToggle = document.getElementById('sidebar-compact-tools');
    const updateCompactToolsUI = (checked) => {
        agenticSettings.compactToolOutputs = checked;
        if (compactToolsToggle) compactToolsToggle.checked = checked;
        if (sideCompactToolsToggle) sideCompactToolsToggle.checked = checked;
        saveAgenticSettings();
    };

    updateCompactToolsUI(agenticSettings.compactToolOutputs !== false);

    if (compactToolsToggle) {
        compactToolsToggle.addEventListener('change', (e) => updateCompactToolsUI(e.target.checked));
    }
    if (sideCompactToolsToggle) {
        sideCompactToolsToggle.addEventListener('change', (e) => updateCompactToolsUI(e.target.checked));
    }

    // Omit Reasoning Toggles
    const omitReasoningToggle = document.getElementById('agentic-omit-reasoning');
    const sideOmitReasoningToggle = document.getElementById('sidebar-omit-reasoning');
    const updateOmitReasoningUI = (checked) => {
        agenticSettings.omitPastReasoning = checked;
        if (omitReasoningToggle) omitReasoningToggle.checked = checked;
        if (sideOmitReasoningToggle) sideOmitReasoningToggle.checked = checked;
        saveAgenticSettings();
    };

    updateOmitReasoningUI(agenticSettings.omitPastReasoning !== false);

    if (omitReasoningToggle) {
        omitReasoningToggle.addEventListener('change', (e) => updateOmitReasoningUI(e.target.checked));
    }
    if (sideOmitReasoningToggle) {
        sideOmitReasoningToggle.addEventListener('change', (e) => updateOmitReasoningUI(e.target.checked));
    }

    // Boundary & Auth Toggles
    const boundaryToggle = document.getElementById('agentic-boundary-enforced');
    if (boundaryToggle) {
        boundaryToggle.checked = agenticSettings.boundaryEnforced;
        boundaryToggle.addEventListener('change', (e) => {
            agenticSettings.boundaryEnforced = e.target.checked;
            saveAgenticSettings();
        });
    }

    const requireAuthToggle = document.getElementById('agentic-require-auth');
    if (requireAuthToggle) {
        requireAuthToggle.checked = agenticSettings.requireAuth;
        requireAuthToggle.addEventListener('change', (e) => {
            agenticSettings.requireAuth = e.target.checked;
            saveAgenticSettings();
        });
    }

    // Tool Checkboxes and Master Toggles
    const masterWebToggle = document.getElementById('tool-master-web');
    const masterLocalToggle = document.getElementById('tool-master-local');

    const webToolNames = ['web_search', 'fetch_url', 'google_search'];
    const localToolNames = ['read_file', 'write_file', 'edit_file', 'execute_command'];

    const toolCheckboxes = {
        read_file: document.getElementById('tool-enable-read'),
        write_file: document.getElementById('tool-enable-write'),
        edit_file: document.getElementById('tool-enable-edit'),
        execute_command: document.getElementById('tool-enable-command'),
        web_search: document.getElementById('tool-enable-web-search'),
        google_search: document.getElementById('tool-enable-google-search'),
        fetch_url: document.getElementById('tool-enable-fetch')
    };

    function syncMasterCheckboxes() {
        if (masterWebToggle) {
            masterWebToggle.checked = webToolNames.some(t => agenticSettings.tools[t] !== false);
        }
        if (masterLocalToggle) {
            masterLocalToggle.checked = localToolNames.some(t => agenticSettings.tools[t] !== false);
        }
    }

    if (masterWebToggle) {
        masterWebToggle.addEventListener('change', (e) => {
            const val = e.target.checked;
            webToolNames.forEach(t => {
                agenticSettings.tools[t] = val;
                if (toolCheckboxes[t]) toolCheckboxes[t].checked = val;
            });
            saveAgenticSettings();
        });
    }

    if (masterLocalToggle) {
        masterLocalToggle.addEventListener('change', (e) => {
            const val = e.target.checked;
            localToolNames.forEach(t => {
                agenticSettings.tools[t] = val;
                if (toolCheckboxes[t]) toolCheckboxes[t].checked = val;
            });
            saveAgenticSettings();
        });
    }

    Object.entries(toolCheckboxes).forEach(([toolName, cb]) => {
        if (cb) {
            cb.checked = agenticSettings.tools[toolName] !== false;
            cb.addEventListener('change', (e) => {
                agenticSettings.tools[toolName] = e.target.checked;
                syncMasterCheckboxes();
                saveAgenticSettings();
            });
        }
    });

    syncMasterCheckboxes();

    // Enter key on path input
    const pathInput = document.getElementById('agentic-new-path-input');
    if (pathInput) {
        pathInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                addAuthorizedPath();
            }
        });
    }

    // Populate search inputs
    const tavilyKeyInput = document.getElementById('tavily-api-key-input');
    if (tavilyKeyInput && agenticSettings.tavilyApiKey) {
        tavilyKeyInput.value = agenticSettings.tavilyApiKey;
    }
    const searxngUrlInput = document.getElementById('searxng-url-input');
    if (searxngUrlInput && agenticSettings.searxngUrl) {
        searxngUrlInput.value = agenticSettings.searxngUrl;
    }
    const braveKeyInput = document.getElementById('brave-api-key-input');
    if (braveKeyInput && agenticSettings.braveApiKey) {
        braveKeyInput.value = agenticSettings.braveApiKey;
    }
    const serperKeyInput = document.getElementById('serper-api-key-input');
    if (serperKeyInput && agenticSettings.serperApiKey) {
        serperKeyInput.value = agenticSettings.serperApiKey;
    }
    const keyInput = document.getElementById('google-api-key-input');
    if (keyInput && agenticSettings.googleApiKey) {
        keyInput.value = agenticSettings.googleApiKey;
    }
    const cxInput = document.getElementById('google-cx-input');
    if (cxInput && agenticSettings.googleCx) {
        cxInput.value = agenticSettings.googleCx;
    }

    const fastMediaToggle = document.getElementById('fast-media-search-toggle');
    const sidebarFastMediaToggle = document.getElementById('sidebar-fast-media-search');
    if (fastMediaToggle) {
        fastMediaToggle.checked = agenticSettings.fastMediaSearch !== false;
        fastMediaToggle.addEventListener('change', (e) => syncFastMediaSearchToggle(e.target.checked));
    }
    if (sidebarFastMediaToggle) {
        sidebarFastMediaToggle.checked = agenticSettings.fastMediaSearch !== false;
        sidebarFastMediaToggle.addEventListener('change', (e) => syncFastMediaSearchToggle(e.target.checked));
    }

    onSearchProviderChange(agenticSettings.searchProvider);
    renderAuthorizedPathsTags();
    fetchWorkspaceInfo();
    fetchSearchProviderSettings();
}

function fetchWorkspaceInfo() {
    fetch(`${getApiBase()}/v1/workspace`)
        .then(res => res.json())
        .then(data => {
            if (data && data.workspace_directory) {
                agenticSettings.workspaceDir = data.workspace_directory;
                const pathBox = document.getElementById('agentic-workspace-path');
                if (pathBox) pathBox.textContent = data.workspace_directory;
                const authModalWs = document.getElementById('auth-modal-workspace');
                if (authModalWs) authModalWs.textContent = data.workspace_directory;
            }
        })
        .catch(() => {
            const pathBox = document.getElementById('agentic-workspace-path');
            if (pathBox) pathBox.textContent = 'Active Working Directory';
        });
}

function renderAuthorizedPathsTags() {
    const container = document.getElementById('agentic-path-tags');
    if (!container) return;

    if (!agenticSettings.authorizedPaths || agenticSettings.authorizedPaths.length === 0) {
        container.innerHTML = '<span style="font-size:0.75rem; color:#80868B; font-style:italic;">No external paths authorized yet. Files outside workspace are protected.</span>';
        return;
    }

    let html = '';
    agenticSettings.authorizedPaths.forEach((p, idx) => {
        html += `
            <div class="agentic-path-chip">
                <span>${escapeHtml(p)}</span>
                <button type="button" class="remove-path-btn" onclick="removeAuthorizedPath(${idx})" title="Remove Authorization">
                    <span class="material-symbols-outlined">close</span>
                </button>
            </div>
        `;
    });
    container.innerHTML = html;
}

function addAuthorizedPath() {
    const input = document.getElementById('agentic-new-path-input');
    if (!input) return;
    const path = input.value.trim();
    if (!path) return;

    if (!agenticSettings.authorizedPaths.includes(path)) {
        agenticSettings.authorizedPaths.push(path);
        saveAgenticSettings();
        renderAuthorizedPathsTags();
    }
    input.value = '';
}

function removeAuthorizedPath(idx) {
    if (idx >= 0 && idx < agenticSettings.authorizedPaths.length) {
        agenticSettings.authorizedPaths.splice(idx, 1);
        saveAgenticSettings();
        renderAuthorizedPathsTags();
    }
}

// Built-in tool definitions builder (sends lightweight tool name list to backend)
function getActiveToolsPayload() {
    const isWebRetrieval = webRetrievalEnabled ? webRetrievalEnabled.checked : true;
    const knownTools = ['web_search', 'google_search', 'fetch_url', 'read_file', 'write_file', 'edit_file', 'execute_command'];

    const activeList = knownTools.filter(name => {
        if ((name === 'web_search' || name === 'google_search' || name === 'fetch_url') && !isWebRetrieval) {
            return false;
        }
        return agenticSettings.tools[name] !== false;
    });

    return activeList;
}

// Authorization Modal Prompt
function showAuthPrompt(tool, path, id) {
    currentPendingAuth = { tool, path, id };
    const modal = document.getElementById('agentic-auth-modal');
    const toolEl = document.getElementById('auth-modal-tool');
    const pathEl = document.getElementById('auth-modal-path');
    const wsEl = document.getElementById('auth-modal-workspace');

    if (toolEl) toolEl.textContent = tool || 'file_tool';
    if (pathEl) pathEl.textContent = path || 'external_path';
    if (wsEl) wsEl.textContent = agenticSettings.workspaceDir || 'Workspace Directory';

    if (modal) modal.style.display = 'flex';
}

function resolveAuthPrompt(allow, always = false) {
    const modal = document.getElementById('agentic-auth-modal');
    if (modal) modal.style.display = 'none';

    if (!currentPendingAuth) return;
    const { tool, path } = currentPendingAuth;

    if (allow) {
        if (!agenticSettings.authorizedPaths.includes(path)) {
            agenticSettings.authorizedPaths.push(path);
            if (always) {
                saveAgenticSettings();
            }
            renderAuthorizedPathsTags();
        }
        if (chatInput && !isGenerating) {
            chatInput.value = `Authorized external path "${path}". Please proceed with the ${tool} operation.`;
            sendMessage();
        }
    } else {
        if (chatInput && !isGenerating) {
            chatInput.value = `Access to external path "${path}" was denied by the user. Please stay within the workspace boundary or suggest an alternative.`;
            sendMessage();
        }
    }
    currentPendingAuth = null;
}

// CAPTCHA / Network Verification Modal Prompt & Proxy Assistant
let currentCaptchaPending = null;
let currentCaptchaUrl = '';

function updateProxyAssistantInfo() {
    const host = window.location.hostname || 'localhost';
    const proxyPort = 8002;
    const badge = document.getElementById('proxy-assistant-host-badge');
    const btnSetup = document.getElementById('btn-download-proxy-setup');
    const btnRestore = document.getElementById('btn-download-proxy-restore');

    if (badge) badge.textContent = `Proxy: ${host}:${proxyPort}`;
    if (btnSetup) btnSetup.href = `${getApiBase()}/api/proxy/setup.bat`;
    if (btnRestore) btnRestore.href = `${getApiBase()}/api/proxy/restore.bat`;

    checkProxyConnectivity(false);
}

function checkProxyConnectivity(showFeedback = false) {
    const badge = document.getElementById('proxy-assistant-host-badge');
    fetch(`${getApiBase()}/api/proxy/status`)
        .then(r => r.json())
        .then(data => {
            if (badge) {
                badge.textContent = `Proxy Online: ${data.proxy_host}:${data.proxy_port}`;
                badge.style.background = 'rgba(34, 197, 94, 0.2)';
                badge.style.color = '#86efac';
            }
            if (showFeedback) {
                showToast(`Proxy listener online on port ${data.proxy_port}! Client: ${data.remote_addr}`);
            }
        })
        .catch(err => {
            if (badge) {
                badge.textContent = `Proxy: Offline / Unreachable`;
                badge.style.background = 'rgba(239, 68, 68, 0.2)';
                badge.style.color = '#fca5a5';
            }
            if (showFeedback) {
                showToast(`Proxy status check failed: ${err.message}`);
            }
        });
}

function copyProxyPacUrl() {
    const pacUrl = `${window.location.protocol}//${window.location.host}/proxy.pac`;
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(pacUrl).then(() => {
            showToast('PAC URL copied to clipboard!');
        }).catch(() => {
            prompt('Copy PAC URL:', pacUrl);
        });
    } else {
        prompt('Copy PAC URL:', pacUrl);
    }
}

function copyBrowserLaunchCmd() {
    const host = window.location.hostname || 'localhost';
    const webUrl = `${window.location.protocol}//${window.location.host}`;
    const cmd = `msedge.exe --proxy-server="http://${host}:8002" ${webUrl}`;
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(cmd).then(() => {
            showToast('Browser launch command copied to clipboard!');
        }).catch(() => {
            prompt('Copy Browser Command:', cmd);
        });
    } else {
        prompt('Copy Browser Command:', cmd);
    }
}

function showCaptchaPrompt(url, callback, reason = 'captcha') {
    currentCaptchaUrl = url || '';
    currentCaptchaPending = { url, callback };
    const modal = document.getElementById('captcha-verification-modal');
    const urlEl = document.getElementById('captcha-modal-url');
    const iframeEl = document.getElementById('captcha-modal-iframe');
    const iframeWrap = document.getElementById('captcha-iframe-wrapper');
    const titleEl = document.getElementById('captcha-modal-title');
    const subEl = document.getElementById('captcha-modal-sub');
    const iconEl = document.getElementById('captcha-modal-icon');
    const iconWrap = document.getElementById('captcha-modal-icon-wrap');
    const proxyCard = document.getElementById('proxy-assistant-card');
    const challengeCard = document.getElementById('challenge-assistant-card');
    const confirmTextEl = document.getElementById('captcha-modal-confirm-text');
    const confirmIconEl = document.getElementById('captcha-modal-confirm-icon');

    if (urlEl) urlEl.textContent = url || 'https://www.google.com';

    if (reason === 'cors') {
        if (proxyCard) proxyCard.style.display = 'block';
        if (challengeCard) challengeCard.style.display = 'none';
        if (iframeWrap) iframeWrap.style.display = 'none';
        updateProxyAssistantInfo();

        if (titleEl) titleEl.textContent = 'Cross-Origin (CORS) Security Restriction';
        if (subEl) subEl.textContent = 'Direct browser fetch was blocked by Cross-Origin (CORS) security policy. Configure the engine proxy to bypass CORS transparently.';
        if (iconEl) iconEl.textContent = 'shield_lock';
        if (iconWrap) {
            iconWrap.style.background = 'rgba(239, 68, 68, 0.15)';
            iconWrap.style.color = '#ef4444';
        }
        if (confirmTextEl) confirmTextEl.textContent = 'Proxy Configured - Retry Request';
        if (confirmIconEl) confirmIconEl.textContent = 'refresh';
    } else {
        if (proxyCard) proxyCard.style.display = 'none';
        if (challengeCard) challengeCard.style.display = 'block';
        if (iframeWrap) iframeWrap.style.display = 'none';

        if (titleEl) titleEl.textContent = 'Host Security Verification';
        if (subEl) subEl.textContent = 'The host (e.g. Google) requires human verification. Open in your browser to solve.';
        if (iconEl) iconEl.textContent = 'verified_user';
        if (iconWrap) {
            iconWrap.style.background = 'rgba(245, 158, 11, 0.15)';
            iconWrap.style.color = '#f59e0b';
        }
        if (confirmTextEl) confirmTextEl.textContent = 'I Solved the Challenge - Continue';
        if (confirmIconEl) confirmIconEl.textContent = 'check_circle';
    }

    if (modal) modal.style.display = 'flex';
}

function openCaptchaInBrowser() {
    if (currentCaptchaUrl) {
        window.open(currentCaptchaUrl, '_blank');
    }
}

function resolveCaptchaPrompt(retried) {
    const modal = document.getElementById('captcha-verification-modal');
    const iframeEl = document.getElementById('captcha-modal-iframe');
    if (iframeEl) iframeEl.src = 'about:blank';
    if (modal) modal.style.display = 'none';

    if (!currentCaptchaPending) return;
    const { callback } = currentCaptchaPending;
    currentCaptchaPending = null;
    if (callback) callback(retried);
}

// ============================================================================
// Main Chat Logic & Context Window Optimization
// ============================================================================

function buildOptimizedMessagesPayload() {
    const messagesToSend = [];
    const sysPrompt = systemPromptInput ? systemPromptInput.value.trim() : '';
    if (sysPrompt) {
        messagesToSend.push({ role: 'system', content: sysPrompt });
    }

    // 1. Slice history according to maxTurns limit (1 turn = 1 user + 1 assistant message)
    let historySlice = chatHistory.slice();
    const maxTurns = agenticSettings.maxTurns;
    if (maxTurns > 0 && historySlice.length > maxTurns * 2) {
        historySlice = historySlice.slice(historySlice.length - (maxTurns * 2));
    }

    // 2. Clean, prune, and compact messages
    for (let idx = 0; idx < historySlice.length; idx++) {
        const msg = historySlice[idx];
        const isCurrentActiveTurn = (idx === historySlice.length - 1);

        const cleanedMsg = {
            role: msg.role,
            content: msg.content || ''
        };

        if (msg.tool_calls) {
            cleanedMsg.tool_calls = msg.tool_calls;
        }

        // If explicitly requested, preserve past reasoning; otherwise omit past reasoning traces
        // to avoid injecting thousands of redundant <think> tokens into subsequent turns.
        if (agenticSettings.omitPastReasoning === false && msg.reasoning_content) {
            cleanedMsg.reasoning_content = msg.reasoning_content;
        }

        // Compact past bulky tool outputs from earlier turns
        if (agenticSettings.compactToolOutputs !== false && !isCurrentActiveTurn) {
            if (cleanedMsg.role === 'tool' || cleanedMsg.role === 'function') {
                if (cleanedMsg.content && cleanedMsg.content.length > 250) {
                    cleanedMsg.content = cleanedMsg.content.slice(0, 250) + `\n... [Content compacted for context window: total ${cleanedMsg.content.length} chars]`;
                }
            } else if (cleanedMsg.role === 'assistant' || cleanedMsg.role === 'user') {
                if (cleanedMsg.content && cleanedMsg.content.includes('<tool_response>')) {
                    cleanedMsg.content = cleanedMsg.content.replace(/<tool_response>([\s\S]*?)<\/tool_response>/g, (match, p1) => {
                        if (p1.length > 250) {
                            return `<tool_response>\n${p1.slice(0, 250)}\n... [Output compacted for context: ${p1.length} chars]\n</tool_response>`;
                        }
                        return match;
                    });
                }
            }
        }

        messagesToSend.push(cleanedMsg);
    }

    return messagesToSend;
}

function stripToolCallsFromText(text) {
    if (!text) return '';
    let out = text;
    out = out.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '');
    out = out.replace(/<｜tool call begin｜>[\s\S]*?<｜tool call end｜>/g, '');
    out = out.replace(/\{"name":\s*"[^"]+"[\s\S]*?\}/g, '');
    out = out.replace(/\{"function":\s*"[^"]+"[\s\S]*?\}/g, '');
    return out.trim();
}

function isRawToolCallString(text) {
    if (!text) return false;
    const trimmed = text.trim();
    if (trimmed.startsWith('<tool_call>') || trimmed.startsWith('<｜tool call begin｜>')) return true;
    if (trimmed.startsWith('{"name"') || trimmed.startsWith('{"function"') || trimmed.startsWith('{"name":') || trimmed.startsWith('{"function":')) return true;
    return false;
}


function extractFromLoadedDom(doc, url, mode = 'text', pattern = '', maxChars = 4000) {
    if (!doc) return { text: '', isBlocked: false, searchCount: 0 };
    const rawHtml = doc.documentElement ? doc.documentElement.outerHTML : (doc.body ? doc.body.innerHTML : '');
    const pageTitle = doc.title || extractDomain(url);
    const docTitle = (doc.title || '').toLowerCase();
    const rawBodyText = (doc.body ? (doc.body.innerText || doc.body.textContent || '') : '').toLowerCase();

    // Genuine bot / CAPTCHA block detection (avoids false positives from normal Google search scripts)
    const isSorry = (doc.location && typeof doc.location.pathname === 'string' && doc.location.pathname.includes('/sorry/')) ||
        (docTitle.includes('about this page') && rawBodyText.includes('unusual traffic')) ||
        (docTitle.includes('informazioni su questa pagina') && rawBodyText.includes('traffico insolito')) ||
        (rawHtml.includes('/sorry/index') || rawHtml.includes('unusual traffic from your computer network'));
    const isJsChallenge = rawHtml.includes('knitsail') || rawHtml.includes('/httpservice/retry/enablejs') || rawHtml.includes('emsg=SG_REL') || rawHtml.includes('having trouble accessing Google Search');
    const isBotBlocked = isSorry || (isJsChallenge && !doc.querySelector('div.g, div.MjjYud, div[data-sokoban-container], #rso, .BNeawe, div.kCrYT, div.Gx5Zad'));

    if (mode === 'raw') {
        if (pattern) {
            const idx = rawHtml.toLowerCase().indexOf(pattern.toLowerCase());
            if (idx >= 0) {
                const start = Math.max(0, idx - 400);
                const end = Math.min(rawHtml.length, idx + pattern.length + 400);
                return { text: `... ${rawHtml.slice(start, end)} ...`, isBlocked: isBotBlocked, searchCount: 0 };
            }
        }
        return { text: rawHtml.slice(0, maxChars), isBlocked: isBotBlocked, searchCount: 0 };
    }

    if (mode === 'scripts') {
        const scripts = Array.from(doc.querySelectorAll('script'));
        let found = [];
        for (const s of scripts) {
            const text = s.textContent || '';
            if (!pattern || text.toLowerCase().includes(pattern.toLowerCase())) {
                found.push(text.trim());
            }
        }
        return { text: found.join('\n---\n').slice(0, maxChars), isBlocked: isBotBlocked, searchCount: 0 };
    }

    if (mode === 'links') {
        const links = Array.from(doc.querySelectorAll('a[href]'));
        let found = [];
        for (const a of links) {
            const href = a.getAttribute('href');
            const text = a.textContent ? a.textContent.trim() : '';
            if (href && (!pattern || href.toLowerCase().includes(pattern.toLowerCase()) || text.toLowerCase().includes(pattern.toLowerCase()))) {
                found.push(`- [${text || href}](${href})`);
            }
        }
        return { text: found.slice(0, 50).join('\n'), isBlocked: isBotBlocked, searchCount: 0 };
    }

    // Default 'text' mode: Search engine result extraction
    const isGoogle = url.includes('google.');
    const isBing = url.includes('bing.com');
    const isDdg = url.includes('duckduckgo.com');
    const isYt = url.includes('youtube.com');

    let searchItems = [];

    if (isGoogle) {
        // 1. Standard Desktop and gbv=1 Google Search Cards
        const cards = doc.querySelectorAll('div.g, div.MjjYud, div[data-sokoban-container], div.tF2Cxc, #rso > div, div.kCrYT, div.Gx5Zad, div.ZINbbc, div.Gx5Zad > div');
        cards.forEach(c => {
            const h3 = c.querySelector('h3, h2, .vvjwJb, .BNeawe.vvjwJb, div[role="heading"]');
            const a = c.querySelector('a[href]');
            if (h3) {
                const title = h3.textContent.trim();
                let link = a ? (a.getAttribute('href') || '') : '';
                if (link.startsWith('/url?q=')) {
                    link = decodeURIComponent(link.slice(7).split('&')[0]);
                } else if (link.startsWith('/url?')) {
                    try {
                        const urlParams = new URLSearchParams(link.slice(5));
                        if (urlParams.has('url')) link = urlParams.get('url');
                        else if (urlParams.has('q')) link = urlParams.get('q');
                    } catch (e) { }
                }
                if (!link || link.startsWith('/') || link.includes('google.')) {
                    const cite = c.querySelector('cite');
                    if (cite && cite.textContent.trim()) {
                        link = cite.textContent.trim();
                    }
                }
                if (!link || link.startsWith('/')) {
                    link = `https://www.google.com/search?q=${encodeURIComponent(title)}&hl=en&gl=us&gbv=1`;
                }
                const snippetEl = c.querySelector('.VwiC3b, .yXK7lf, .IsZvec, div[data-sncf], .b_caption, .b_snippet, .BNeawe.s3v9rd, .s3v9rd');
                const snippet = snippetEl ? snippetEl.textContent.trim() : '';
                if (title && !title.startsWith('http') && title !== 'Google' && !searchItems.some(item => item.title === title)) {
                    searchItems.push({ title, link, snippet });
                }
            }
        });

        // 2. Fallback for basic Google markup with /url?q= links
        if (searchItems.length === 0) {
            const urlLinks = doc.querySelectorAll('a[href^="/url?q="]');
            urlLinks.forEach(a => {
                const href = a.getAttribute('href') || '';
                let cleanLink = decodeURIComponent(href.slice(7).split('&')[0]);
                const title = a.textContent.trim();
                if (title && cleanLink && cleanLink.startsWith('http') && !cleanLink.includes('google.') && !searchItems.some(item => item.link === cleanLink)) {
                    searchItems.push({ title, link: cleanLink, snippet: '' });
                }
            });
        }
    } else if (isBing) {
        const cards = doc.querySelectorAll('li.b_algo, #b_results > li');
        cards.forEach(c => {
            const h2a = c.querySelector('h2 a');
            if (h2a) {
                const title = h2a.textContent.trim();
                const link = h2a.getAttribute('href') || '';
                const snippetEl = c.querySelector('.b_caption p, .b_snippet');
                const snippet = snippetEl ? snippetEl.textContent.trim() : '';
                if (title && link && !searchItems.some(item => item.link === link)) {
                    searchItems.push({ title, link, snippet });
                }
            }
        });
    } else if (isDdg) {
        const cards = doc.querySelectorAll('article[data-testid="result"], .result__body');
        cards.forEach(c => {
            const a = c.querySelector('h2 a, .result__title a');
            if (a) {
                const title = a.textContent.trim();
                const link = a.getAttribute('href') || '';
                const snippetEl = c.querySelector('[data-result="snippet"], .result__snippet');
                const snippet = snippetEl ? snippetEl.textContent.trim() : '';
                if (title && link && !searchItems.some(item => item.link === link)) {
                    searchItems.push({ title, link, snippet });
                }
            }
        });
    } else if (isYt) {
        const links = doc.querySelectorAll('a[href*="/watch?v="]');
        links.forEach(a => {
            const href = a.getAttribute('href') || '';
            const title = (a.getAttribute('title') || a.textContent || '').trim();
            const fullLink = href.startsWith('http') ? href : ('https://www.youtube.com' + href);
            if (title && fullLink && !searchItems.some(item => item.link === fullLink)) {
                searchItems.push({ title, link: fullLink, snippet: 'YouTube Video' });
            }
        });
    }

    if (searchItems.length > 0) {
        let out = `Search Results for "${url}":\n\n`;
        searchItems.slice(0, 10).forEach((item, idx) => {
            out += `${idx + 1}. **[${item.title}](${item.link})**\n   URL: ${item.link}\n`;
            if (item.snippet) out += `   Snippet: ${item.snippet}\n`;
            out += '\n';
        });
        return { text: out.slice(0, maxChars), isBlocked: false, searchCount: searchItems.length };
    }

    // Fallback: Generic clean DOM text extraction (stripping out useless scripts, css, nav, headers)
    try {
        const bodyClone = doc.body ? doc.body.cloneNode(true) : null;
        if (bodyClone) {
            const unwanted = bodyClone.querySelectorAll('script, style, noscript, svg, nav, footer, header, form, iframe, aside');
            unwanted.forEach(el => el.remove());
            let text = bodyClone.innerText || bodyClone.textContent || '';
            text = text.replace(/[ \t]+/g, ' ').replace(/\n\s*\n\s*\n+/g, '\n\n').trim();
            if (pattern) {
                const idx = text.toLowerCase().indexOf(pattern.toLowerCase());
                if (idx >= 0) {
                    const start = Math.max(0, idx - 400);
                    const end = Math.min(text.length, idx + pattern.length + 400);
                    return { text: `... ${text.slice(start, end)} ...`, isBlocked: isBotBlocked, searchCount: 0 };
                }
            }
            return { text: `Page Title: ${pageTitle}\nURL: ${url}\n\n${text.slice(0, maxChars)}`, isBlocked: isBotBlocked, searchCount: 0 };
        }
    } catch (e) {
        console.warn('DOM clone text extraction error', e);
    }
    return { text: `Page Title: ${pageTitle}\nURL: ${url}`, isBlocked: isBotBlocked, searchCount: 0 };
}

// ============================================================================
// Direct JavaScript Tavily AI API Client (SDK Equivalent: @tavily/core)
// ============================================================================

async function executeTavilySearchJS(query, apiKey, maxResults = 5) {
    const key = (apiKey || agenticSettings.tavilyApiKey || '').trim();
    if (!key) {
        return {
            output: "[Tavily Search API Key Not Configured]\n\n" +
                    "Tavily provides 1,000 free web searches monthly without requiring a credit card.\n" +
                    "1. Get your free API key at: https://tavily.com\n" +
                    "2. Save your API key in Settings.",
            retrieved_document: null
        };
    }

    const isMedia = (agenticSettings.fastMediaSearch !== false) && isMediaSearchQuery(query);
    const effectiveMaxResults = isMedia ? Math.min(maxResults || 3, 3) : (maxResults || 5);
    const effectiveQuery = isMedia && !query.toLowerCase().includes('youtube') && !query.toLowerCase().includes('video') ? `${query} youtube` : query;

    const payload = {
        api_key: key,
        query: effectiveQuery,
        search_depth: "basic",
        include_answer: !isMedia,
        max_results: effectiveMaxResults
    };

    const res = await fetch('https://api.tavily.com/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
    });

    if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.error || `HTTP ${res.status} ${res.statusText}`);
    }

    const data = await res.json();
    const results = data.results || [];
    const answer = data.answer || '';

    if (isMedia) {
        let textOut = `[Media Search Results: "${query}"]\n\n`;
        let topYtDoc = null;
        let count = 0;
        for (const item of results) {
            count++;
            const title = item.title || 'Video Link';
            const link = item.url || '';
            const snippet = item.content || '';
            textOut += `${count}. [${title}](${link})\n`;
            if (snippet) textOut += `   ${snippet}\n\n`;
            if (!topYtDoc && link) {
                const m = link.match(/(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})/i);
                if (m) {
                    topYtDoc = {
                        id: 'yt_' + m[1],
                        url: `https://www.youtube.com/watch?v=${m[1]}`,
                        title: title,
                        html: createYouTubePlayerHtml(m[1], title),
                        snippet: item.content || 'Interactive YouTube Player'
                    };
                    addRetrievedDocument(topYtDoc);
                }
            }
            if (count >= 3) break;
        }
        if (count === 0) {
            textOut += `No media results found for query: "${query}"`;
        }
        return {
            output: textOut,
            retrieved_document: topYtDoc
        };
    }

    let textOut = `[Web Search Results (Tavily AI): "${query}"]\n\n`;
    if (answer) {
        textOut += `Direct AI Summary: ${answer}\n\n`;
    }

    let htmlCards = `<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; color:#e2e8f0; background:#0f172a; padding:20px; border-radius:12px; max-width:860px; margin:0 auto;">`;
    htmlCards += `<h2 style="margin:0 0 16px 0; font-size:18px; color:#60a5fa;">Tavily AI Search: ${escapeHtml(query)}</h2>`;
    if (answer) {
        htmlCards += `<div style="background:rgba(59,130,246,0.15); border:1px solid rgba(59,130,246,0.3); border-radius:8px; padding:12px 14px; margin-bottom:16px; font-size:14px; color:#93c5fd;"><strong>AI Summary:</strong> ${escapeHtml(answer)}</div>`;
    }

    results.slice(0, maxResults || 5).forEach((item, idx) => {
        const title = item.title || 'Source';
        const link = item.url || '';
        const snippet = item.content || '';
        const score = item.score !== undefined ? ` (Score: ${(item.score * 100).toFixed(0)}%)` : '';

        textOut += `${idx + 1}. **${title}**\n   URL: ${link}\n   Snippet: ${snippet}\n\n`;

        htmlCards += `
            <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:14px 16px; margin-bottom:12px;">
                <a href="${escapeHtml(link)}" target="_blank" style="font-size:16px; font-weight:600; color:#93c5fd; text-decoration:none; display:inline-block; margin-bottom:6px;">${escapeHtml(title)}${score} &rarr;</a>
                <p style="font-size:13px; color:#cbd5e1; margin:0; line-height:1.5;">${escapeHtml(snippet)}</p>
            </div>
        `;
    });
    htmlCards += `</div>`;

    if (results.length === 0) {
        textOut += `No search results found for query: "${query}"`;
    }

    let ytDocFromSearch = null;
    for (const item of results) {
        const link = item.url || '';
        const m = link.match(/(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})/i);
        if (m) {
            ytDocFromSearch = {
                id: 'yt_' + m[1],
                url: `https://www.youtube.com/watch?v=${m[1]}`,
                title: item.title || 'YouTube Video',
                html: createYouTubePlayerHtml(m[1], item.title || 'YouTube Video'),
                snippet: item.content || 'Interactive YouTube Player'
            };
            addRetrievedDocument(ytDocFromSearch);
            break;
        }
    }

    const primaryUrl = (results.length > 0 && results[0].url) ? results[0].url : 'https://tavily.com';

    return {
        output: textOut,
        retrieved_document: ytDocFromSearch || {
            id: 'tavily_' + Date.now(),
            url: primaryUrl,
            title: (results.length > 0 && results[0].title) ? results[0].title : `Tavily Search: ${query}`,
            html: htmlCards,
            snippet: answer || (results.length > 0 ? results[0].content : '')
        }
    };
}

function executeBrowserFetch(url, mode = 'text', pattern = '', maxChars = 4000) {
    return new Promise(async (resolve) => {
        // 1. Direct Media / YouTube playback detection (Instant 0-latency player registration)
        const ytMatch = url.match(/(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})/i);
        if (ytMatch) {
            const videoId = ytMatch[1];
            const ytDocId = 'yt_' + videoId;
            const ytHtml = createYouTubePlayerHtml(videoId, 'YouTube Video');
            const ytDoc = {
                id: ytDocId,
                url: `https://www.youtube.com/watch?v=${videoId}`,
                title: 'YouTube Video',
                html: ytHtml,
                snippet: 'Interactive YouTube Player'
            };
            addRetrievedDocument(ytDoc);
            resolve({
                output: `[Retrieved YouTube Video: https://www.youtube.com/watch?v=${videoId} | Video player registered for preview]`,
                retrieved_document: ytDoc
            });
            return;
        }

        // 2. Audio / Video Direct File Playback detection
        if (/\.(mp4|webm|ogv|mp3|wav|ogg|m4a|aac)(\?.*)?$/i.test(url)) {
            const isVideo = /\.(mp4|webm|ogv)(\?.*)?$/i.test(url);
            const mediaHtml = `<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Media Playback</title><style>body{margin:0;padding:24px;background:#0f0f0f;color:#fff;display:flex;flex-direction:column;align-items:center;justify-content:center;font-family:sans-serif;} video,audio{max-width:90%;border-radius:12px;box-shadow:0 8px 30px rgba(0,0,0,0.7);}</style></head><body><h3>Media Stream</h3>${isVideo ? `<video controls autoplay src="${escapeHtml(url)}" style="max-height:480px;"></video>` : `<audio controls autoplay src="${escapeHtml(url)}"></audio>`}<p><a href="${escapeHtml(url)}" target="_blank" style="color:#60a5fa;">${escapeHtml(url)}</a></p></body></html>`;
            const mediaDoc = {
                id: 'media_' + Date.now(),
                url: url,
                title: 'Media Playback',
                html: mediaHtml,
                snippet: 'Media Stream Player'
            };
            addRetrievedDocument(mediaDoc);
            resolve({
                output: `[Retrieved Media Stream: ${url} | Media player registered]`,
                retrieved_document: mediaDoc
            });
            return;
        }

        // 3. Direct document / web page retrieval via backend engine tool executor
        try {
            const res = await fetch(`${getApiBase()}/api/tool/execute`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: 'fetch_url',
                    arguments: JSON.stringify({ url, mode, pattern, max_chars: maxChars }),
                    timeout_ms: (agenticSettings.timeoutSec || 60) * 1000
                })
            });
            const data = await res.json();
            if (data.retrieved_document) {
                const docText = data.output || data.retrieved_document.clean_text || '';
                const ytMatchInDoc = docText.match(/(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})/i);
                if (ytMatchInDoc && !data.retrieved_document.html) {
                    data.retrieved_document.html = createYouTubePlayerHtml(ytMatchInDoc[1], data.retrieved_document.title || 'YouTube Video');
                }
                addRetrievedDocument(data.retrieved_document);
            }
            resolve({
                output: data.output || JSON.stringify(data),
                retrieved_document: data.retrieved_document
            });
            return;
        } catch (err) {
            console.warn('[executeBrowserFetch] Backend tool execute failed, falling back to embedded proxy:', err);
        }

        // 4. Fallback via embedded Moecher proxy (/api/proxy?url=...)
        try {
            const proxyRes = await fetch(`${getApiBase()}/api/proxy?url=${encodeURIComponent(url)}`);
            const htmlStr = await proxyRes.text();
            const parser = new DOMParser();
            const parsedDoc = parser.parseFromString(htmlStr, 'text/html');
            const res = extractFromLoadedDom(parsedDoc, url, mode, pattern, maxChars);
            const finalText = (res && res.text) ? res.text : htmlStr.slice(0, maxChars);

            let ytId = null;
            const ytMatch = (htmlStr + ' ' + url).match(/(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})/i);
            if (ytMatch) ytId = ytMatch[1];

            const fetchedDoc = {
                id: 'doc_' + Date.now(),
                url: url,
                title: parsedDoc.title || extractDomain(url),
                html: ytId ? createYouTubePlayerHtml(ytId, parsedDoc.title || 'YouTube Video') : htmlStr,
                snippet: finalText.slice(0, 300)
            };
            addRetrievedDocument(fetchedDoc);
            resolve({
                output: finalText,
                retrieved_document: fetchedDoc
            });
        } catch (proxyErr) {
            resolve({
                output: `[Fetch error for ${url}: ${proxyErr.message}]`,
                retrieved_document: null
            });
        }
    });
}

async function executeClientToolCall(tc, turnRetrievedDocs = null) {
    if (tc.name === 'fetch_url') {
        let args = {};
        try {
            args = typeof tc.arguments === 'string' ? JSON.parse(tc.arguments) : (tc.arguments || {});
        } catch (e) {
            args = { url: tc.arguments };
        }
        let url = args.url || '';
        if (!url.startsWith('http://') && !url.startsWith('https://')) {
            url = `https://www.google.com/search?q=${encodeURIComponent(url)}&hl=en&gl=us`;
        }
        const mode = args.mode || 'text';
        const pattern = args.pattern || '';
        const maxChars = args.max_chars || 4000;

        const fetchRes = await executeBrowserFetch(url, mode, pattern, maxChars);
        if (typeof fetchRes === 'object' && fetchRes !== null) {
            if (fetchRes.retrieved_document && turnRetrievedDocs) {
                turnRetrievedDocs.push(fetchRes.retrieved_document);
            }
            return fetchRes.output || JSON.stringify(fetchRes);
        }
        return fetchRes;
    } else if (tc.name === 'web_search' || tc.name === 'google_search' || tc.name === 'youtube_search') {
        let args = {};
        try {
            args = typeof tc.arguments === 'string' ? JSON.parse(tc.arguments) : (tc.arguments || {});
        } catch (e) {
            args = { query: tc.arguments };
        }
        const query = args.query || '';
        const isMedia = (tc.name === 'youtube_search') || ((agenticSettings.fastMediaSearch !== false) && isMediaSearchQuery(query));

        // 1. If fast media search is enabled or tool is youtube_search, try direct YouTube search first (0 API cost)
        if (isMedia) {
            try {
                const ytRes = await fetch(`${getApiBase()}/api/tool/execute`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name: 'youtube_search',
                        arguments: JSON.stringify({ query: query, num_results: Math.min(args.num_results || 3, 3) }),
                        timeout_ms: 10000
                    })
                });
                const ytData = await ytRes.json();
                if (ytData && ytData.output && ytData.retrieved_document && ytData.retrieved_document.url && ytData.retrieved_document.url.includes('youtube.com/watch')) {
                    addRetrievedDocument(ytData.retrieved_document);
                    if (turnRetrievedDocs) turnRetrievedDocs.push(ytData.retrieved_document);
                    return ytData.output;
                }
            } catch (ytErr) {
                console.warn('[YouTube Direct Search] Fallback to configured provider:', ytErr);
            }
        }

        const provider = args.provider || agenticSettings.searchProvider || 'tavily';
        const tavilyKey = args.tavily_api_key || agenticSettings.tavilyApiKey || '';

        // Direct JavaScript Tavily AI execution (SDK Equivalent: @tavily/core)
        if (provider === 'tavily' && tavilyKey) {
            try {
                const tavilyResult = await executeTavilySearchJS(query, tavilyKey, args.num_results || 5);
                if (tavilyResult && !tavilyResult.error) {
                    if (tavilyResult.retrieved_document) {
                        addRetrievedDocument(tavilyResult.retrieved_document);
                        if (turnRetrievedDocs) turnRetrievedDocs.push(tavilyResult.retrieved_document);
                    }
                    return tavilyResult.output;
                }
            } catch (jsErr) {
                console.warn('[Tavily JS Client] Direct fetch fallback to backend:', jsErr);
            }
        }

        if (!args.provider && agenticSettings.searchProvider) args.provider = agenticSettings.searchProvider;
        if (!args.tavily_api_key && agenticSettings.tavilyApiKey) args.tavily_api_key = agenticSettings.tavilyApiKey;
        if (!args.searxng_url && agenticSettings.searxngUrl) args.searxng_url = agenticSettings.searxngUrl;
        if (!args.brave_api_key && agenticSettings.braveApiKey) args.brave_api_key = agenticSettings.braveApiKey;
        if (!args.serper_api_key && agenticSettings.serperApiKey) args.serper_api_key = agenticSettings.serperApiKey;
        if (!args.api_key && agenticSettings.googleApiKey) args.api_key = agenticSettings.googleApiKey;
        if (!args.google_search_api_key && agenticSettings.googleApiKey) args.google_search_api_key = agenticSettings.googleApiKey;
        if (!args.cx && agenticSettings.googleCx) args.cx = agenticSettings.googleCx;
        if (!args.google_search_cx && agenticSettings.googleCx) args.google_search_cx = agenticSettings.googleCx;

        try {
            const res = await fetch(`${getApiBase()}/api/tool/execute`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: 'web_search',
                    arguments: JSON.stringify(args),
                    timeout_ms: (agenticSettings.timeoutSec || 60) * 1000
                })
            });
            const data = await res.json();
            if (data.retrieved_document) {
                addRetrievedDocument(data.retrieved_document);
                if (turnRetrievedDocs) turnRetrievedDocs.push(data.retrieved_document);
            }
            return data.output || JSON.stringify(data);
        } catch (err) {
            return `[Failed to execute ${tc.name}: ${err.message}]`;
        }
    } else {
        try {
            const res = await fetch(`${getApiBase()}/api/tool/execute`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: tc.name,
                    arguments: tc.arguments,
                    timeout_ms: (agenticSettings.timeoutSec || 60) * 1000,
                    require_external_authorization: agenticSettings.requireAuth !== false,
                    workspace_boundary_enforced: agenticSettings.boundaryEnforced !== false,
                    authorized_paths: agenticSettings.authorizedPaths || []
                })
            });
            const data = await res.json();
            if (data.authorization_required) {
                showAuthPrompt(data.authorization_required.tool, data.authorization_required.path, tc.id);
            }
            if (data.retrieved_document) {
                addRetrievedDocument(data.retrieved_document);
                if (turnRetrievedDocs) turnRetrievedDocs.push(data.retrieved_document);
            }
            return data.output || JSON.stringify(data);
        } catch (err) {
            return `[Failed to execute ${tc.name}: ${err.message}]`;
        }
    }
}

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

    // Index where this turn starts in chatHistory
    const turnHistoryStartIndex = chatHistory.length;

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

    // Initial visual feedback while engine starts thinking/elaborating
    const liveIndicator = document.createElement('div');
    liveIndicator.className = 'elaboration-status-badge';
    liveIndicator.id = 'live-status-indicator';
    liveIndicator.innerHTML = `
        <span class="tool-pulse-spinner"></span>
        <span class="status-msg-text">Thinking and preparing response</span>
        <div class="elaboration-dots"><span></span><span></span><span></span></div>
    `;
    mainContent.appendChild(liveIndicator);

    const isThinking = thinkingEnabled ? thinkingEnabled.checked : true;
    const budgetVal = isThinking ? (thinkingBudget ? parseInt(thinkingBudget.value, 10) : 4096) : 0;
    const activeTools = getActiveToolsPayload();

    const startTime = performance.now();
    let firstTokenTime = null;
    let totalPromptTokens = 0;
    let totalCompletionTokens = 0;
    let turnRetrievedDocs = [];
    let isReasoningDone = false;

    const maxRounds = 6;
    let round = 0;

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

        const clientUa = navigator.userAgent || '';
        const clientLang = (navigator.languages && navigator.languages.length) ? navigator.languages.join(',') : (navigator.language || 'en-US,en');
        const clientSecUa = (navigator.userAgentData && navigator.userAgentData.brands) ? navigator.userAgentData.brands.map(b => `"${b.brand}";v="${b.version}"`).join(', ') : '"Chromium";v="133", "Not(A:Brand";v="99", "Microsoft Edge";v="133"';
        const clientMobile = (navigator.userAgentData && navigator.userAgentData.mobile) ? '?1' : '?0';
        const clientPlatform = (navigator.userAgentData && navigator.userAgentData.platform) ? `"${navigator.userAgentData.platform}"` : '"Windows"';
        const clientCookies = document.cookie || '';

        while (round < maxRounds) {
            round++;
            const messagesToSend = buildOptimizedMessagesPayload();

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
                reasoning_effort: isThinking ? reasoningEffort.value : "none",
                execution_timeout_sec: agenticSettings.timeoutSec || 60,
                workspace_boundary_enforced: agenticSettings.boundaryEnforced !== false,
                require_external_authorization: agenticSettings.requireAuth !== false,
                authorized_paths: agenticSettings.authorizedPaths || [],
                client_tool_execution: true,
                client_context: {
                    user_agent: clientUa,
                    languages: clientLang,
                    sec_ch_ua: clientSecUa,
                    mobile: clientMobile,
                    platform: clientPlatform,
                    cookies: clientCookies
                }
            };

            if (activeTools.length > 0) {
                const allTools = ['web_search', 'google_search', 'fetch_url', 'read_file', 'write_file', 'edit_file', 'execute_command'];
                if (activeTools.length === allTools.length && activeTools.every((t, i) => t === allTools[i])) {
                    payload.tools = "default";
                } else {
                    payload.tools = activeTools;
                }
            }

            let roundReasoning = "";
            let roundContent = "";
            let roundToolCalls = [];
            let roundFinishReason = "stop";

            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'text/event-stream',
                    'X-Client-Tool-Execution': 'true',
                    'X-Client-User-Agent': clientUa,
                    'X-Client-Accept-Language': clientLang,
                    'X-Client-Cookie': clientCookies
                },
                body: pythonJsonDumps(payload),
                signal: currentAbortController.signal
            });

            if (!response.ok) {
                throw new Error(`Server returned error ${response.status}: ${response.statusText}`);
            }

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
                                if (data.usage.prompt_tokens !== undefined) totalPromptTokens += data.usage.prompt_tokens;
                                if (data.usage.completion_tokens !== undefined) totalCompletionTokens += data.usage.completion_tokens;
                            }

                            if (data.choices && data.choices.length > 0) {
                                const choice = data.choices[0];
                                const delta = choice.delta || {};
                                if (choice.finish_reason) {
                                    roundFinishReason = choice.finish_reason;
                                }

                                if (delta.retrieved_document) {
                                    turnRetrievedDocs.push(delta.retrieved_document);
                                    addRetrievedDocument(delta.retrieved_document);
                                }

                                if (delta.authorization_required) {
                                    showAuthPrompt(delta.authorization_required.tool, delta.authorization_required.path, delta.authorization_required.id);
                                }

                                if (delta.tool_calls && Array.isArray(delta.tool_calls)) {
                                    for (const tc of delta.tool_calls) {
                                        const idx = tc.index !== undefined ? tc.index : roundToolCalls.length;
                                        if (!roundToolCalls[idx]) {
                                            roundToolCalls[idx] = { id: tc.id || ('tc_' + idx), type: tc.type || 'function', name: '', arguments: '' };
                                        }
                                        if (tc.id) roundToolCalls[idx].id = tc.id;
                                        if (tc.function) {
                                            if (tc.function.name) roundToolCalls[idx].name = tc.function.name;
                                            if (tc.function.arguments) roundToolCalls[idx].arguments += tc.function.arguments;
                                        }
                                    }
                                }

                                if (delta.reasoning_content !== undefined) {
                                    if (firstTokenTime === null) firstTokenTime = performance.now();

                                    const statusText = liveIndicator ? liveIndicator.querySelector('.status-msg-text') : null;
                                    if (statusText) statusText.textContent = 'Reasoning and analyzing query...';

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
                                    } else {
                                        reasoningBlock.open = true;
                                        const summary = reasoningBlock.querySelector('summary');
                                        if (summary && !summary.querySelector('.thinking-spinner')) {
                                            summary.innerHTML = '<span class="thinking-spinner">progress_activity</span> Thinking...';
                                        }
                                    }

                                    const toolIdMatch = delta.reasoning_content.match(/id="(tool-act-[^"]+)"/);
                                    let replaced = false;
                                    if (toolIdMatch) {
                                        const actId = toolIdMatch[1];
                                        const regex = new RegExp('<div class="tool-activity-block active"[^>]*id="' + actId + '"[\\s\\S]*?<\\/div>', 'g');
                                        if (regex.test(roundReasoning)) {
                                            roundReasoning = roundReasoning.replace(regex, delta.reasoning_content.trim());
                                            replaced = true;
                                        }
                                    }
                                    if (!replaced) {
                                        roundReasoning += delta.reasoning_content;
                                    }
                                    reasoningContent.innerHTML = marked.parse(roundReasoning);
                                }

                                if (delta.content !== undefined) {
                                    if (firstTokenTime === null) firstTokenTime = performance.now();

                                    if (liveIndicator && liveIndicator.parentElement) {
                                        liveIndicator.remove();
                                    }

                                    const toolIdMatch = delta.content.match(/id="(tool-act-[^"]+)"/);
                                    let replaced = false;
                                    if (toolIdMatch) {
                                        const actId = toolIdMatch[1];
                                        const regex = new RegExp('<div class="tool-activity-block active"[^>]*id="' + actId + '"[\\s\\S]*?<\\/div>', 'g');
                                        if (regex.test(roundContent)) {
                                            roundContent = roundContent.replace(regex, delta.content.trim());
                                            replaced = true;
                                            renderMarkdownContent(roundContent, mainContent);
                                        }
                                    }

                                    if (!replaced) {
                                        if (reasoningBlock && !isReasoningDone) {
                                            isReasoningDone = true;
                                            reasoningBlock.open = false;
                                            const summary = reasoningBlock.querySelector('summary');
                                            if (summary) summary.innerHTML = 'Thought process';
                                        }
                                        roundContent += delta.content;

                                        // Filter out raw tool call JSON so it never pollutes the chat UI
                                        const displayContent = stripToolCallsFromText(roundContent);
                                        if (displayContent.length > 0) {
                                            renderMarkdownContent(displayContent, mainContent);
                                        } else if (isRawToolCallString(roundContent)) {
                                            mainContent.innerHTML = '';
                                        }
                                    }
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

            const validToolCalls = roundToolCalls.filter(tc => tc && tc.name);

            if (validToolCalls.length > 0 && (roundFinishReason === 'tool_calls' || roundFinishReason === 'stop')) {
                const cleanRoundContent = stripToolCallsFromText(roundContent);
                if (cleanRoundContent.length > 0) {
                    renderMarkdownContent(cleanRoundContent, mainContent);
                } else {
                    mainContent.innerHTML = '';
                }

                chatHistory.push({
                    role: 'assistant',
                    content: cleanRoundContent,
                    reasoning_content: roundReasoning || undefined,
                    tool_calls: validToolCalls.map(tc => ({
                        id: tc.id,
                        type: tc.type || 'function',
                        function: { name: tc.name, arguments: tc.arguments }
                    }))
                });

                for (const tc of validToolCalls) {
                    let toolTarget = '';
                    let actionLabel = 'Executing tool';
                    let doneIcon = 'done';
                    let completedLabel = 'Completed';

                    try {
                        const parsedArgs = typeof tc.arguments === 'string' ? JSON.parse(tc.arguments) : tc.arguments;
                        if (tc.name === 'web_search' || tc.name === 'google_search') {
                            toolTarget = parsedArgs.query || '';
                            const prov = agenticSettings.searchProvider || 'web';
                            let provName = 'Web';
                            if (prov === 'tavily') provName = 'Tavily';
                            else if (prov === 'searxng') provName = 'SearXNG';
                            else if (prov === 'brave') provName = 'Brave';
                            else if (prov === 'serper') provName = 'Serper';
                            else if (prov === 'google') provName = 'Google';
                            actionLabel = `${provName} Searching`;
                            doneIcon = 'travel_explore';
                            completedLabel = `${provName} Searched`;
                        } else if (tc.name === 'fetch_url') {
                            toolTarget = parsedArgs.url || '';
                            actionLabel = 'Browsing web page';
                            doneIcon = 'travel_explore';
                            completedLabel = 'Rendered DOM';
                        } else if (tc.name === 'read_file') {
                            toolTarget = parsedArgs.path || '';
                            actionLabel = 'Reading file';
                            doneIcon = 'description';
                            completedLabel = 'Read file';
                        } else if (tc.name === 'write_file') {
                            toolTarget = parsedArgs.path || '';
                            actionLabel = 'Writing file';
                            doneIcon = 'edit_document';
                            completedLabel = 'Wrote file';
                        } else if (tc.name === 'edit_file') {
                            toolTarget = parsedArgs.path || '';
                            actionLabel = 'Editing file';
                            doneIcon = 'find_replace';
                            completedLabel = 'Edited file';
                        } else if (tc.name === 'execute_command') {
                            toolTarget = parsedArgs.command || '';
                            actionLabel = 'Running command';
                            doneIcon = 'terminal';
                            completedLabel = 'Executed';
                        }
                    } catch (e) {
                        toolTarget = tc.arguments || '';
                    }

                    const activeCardHtml = `
<div class="tool-activity-block active" id="tool-act-${tc.id}">
  <span class="thinking-spinner">progress_activity</span>
  <span class="tool-action-label">${escapeHtml(actionLabel)}</span>
  <span class="tool-target-subtle">${escapeHtml(toolTarget)}</span>
</div>
`;
                    const toolTargetContainer = (reasoningContent && !isReasoningDone) ? reasoningContent : mainContent;
                    toolTargetContainer.insertAdjacentHTML('beforeend', activeCardHtml);

                    const toolOutput = await executeClientToolCall(tc, turnRetrievedDocs);

                    const activeCardEl = document.getElementById(`tool-act-${tc.id}`);
                    if (activeCardEl) {
                        activeCardEl.className = 'tool-activity-block completed';
                        activeCardEl.innerHTML = `
  <span class="material-symbols-outlined tool-done-icon">${doneIcon}</span>
  <span class="tool-action-label">${escapeHtml(completedLabel)}</span>
  <span class="tool-target-subtle">${escapeHtml(toolTarget)}</span>
`;
                    }

                    chatHistory.push({
                        role: 'tool',
                        tool_call_id: tc.id,
                        content: toolOutput
                    });
                }

                continue;
            }

            chatHistory.push({
                role: 'assistant',
                content: roundContent,
                reasoning_content: roundReasoning || undefined
            });

            // --- Turn Media & Preview Detection ---
            let turnMediaDoc = null;

            // 1. Check all turnRetrievedDocs from this turn
            for (let i = turnRetrievedDocs.length - 1; i >= 0; i--) {
                const d = turnRetrievedDocs[i];
                if (!d) continue;
                if (d.html && (d.html.includes('youtube-nocookie.com/embed') || d.html.includes('youtube.com/iframe_api') || d.html.includes('<video') || d.html.includes('<audio'))) {
                    turnMediaDoc = d;
                    break;
                }
                if (d.url && (d.url.includes('watch?v=') || d.url.includes('youtu.be') || d.url.includes('youtube.com/embed'))) {
                    const m = d.url.match(/(?:watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})/i);
                    if (m) {
                        turnMediaDoc = {
                            id: 'yt_' + m[1],
                            url: `https://www.youtube.com/watch?v=${m[1]}`,
                            title: d.title || 'YouTube Video',
                            html: createYouTubePlayerHtml(m[1], d.title || 'YouTube Video'),
                            snippet: 'Interactive YouTube Player'
                        };
                        addRetrievedDocument(turnMediaDoc);
                        break;
                    }
                }
            }

            // 2. If not found in turnRetrievedDocs, check tool outputs ONLY from THIS turn (in reverse order)
            if (!turnMediaDoc) {
                const currentTurnToolOutputs = chatHistory.slice(turnHistoryStartIndex).filter(m => m.role === 'tool');
                for (let i = currentTurnToolOutputs.length - 1; i >= 0; i--) {
                    const content = currentTurnToolOutputs[i].content || '';
                    const ytMatch = content.match(/(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})/i);
                    if (ytMatch) {
                        const videoId = ytMatch[1];
                        const ytHtml = createYouTubePlayerHtml(videoId, 'YouTube Video');
                        turnMediaDoc = {
                            id: 'yt_' + videoId,
                            url: `https://www.youtube.com/watch?v=${videoId}`,
                            title: 'YouTube Video',
                            html: ytHtml,
                            snippet: 'Interactive YouTube Player'
                        };
                        addRetrievedDocument(turnMediaDoc);
                        break;
                    }
                }
            }

            // 3. If not found in tool outputs, check the assistant response (roundContent)
            if (!turnMediaDoc) {
                const ytMatch = roundContent.match(/(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})/i);
                if (ytMatch) {
                    const videoId = ytMatch[1];
                    const ytHtml = createYouTubePlayerHtml(videoId, 'YouTube Video');
                    turnMediaDoc = {
                        id: 'yt_' + videoId,
                        url: `https://www.youtube.com/watch?v=${videoId}`,
                        title: 'YouTube Video',
                        html: ytHtml,
                        snippet: 'Interactive YouTube Player'
                    };
                    addRetrievedDocument(turnMediaDoc);
                }
            }

            // 4. Check if turn generated an explicit HTML code block or web app
            const turnHtml = getTurnHtmlCode(assistantMsgDiv, roundContent);

            if (turnMediaDoc && turnMediaDoc.html) {
                loadHtmlIntoPreview(turnMediaDoc.html, true);
            } else if (turnHtml && turnHtml.trim().length > 0) {
                loadHtmlIntoPreview(turnHtml, false);
            }

            break;
        }

        const endTime = performance.now();
        if (firstTokenTime === null) firstTokenTime = endTime;
        const ttftMs = firstTokenTime - startTime;
        const decodeTimeMs = Math.max(0, endTime - firstTokenTime);
        const ttftSec = ttftMs / 1000.0;
        const decodeSec = decodeTimeMs / 1000.0;

        const actualCompletionTokens = totalCompletionTokens > 0 ? totalCompletionTokens : 1;
        const prefillTps = (ttftSec > 0 && totalPromptTokens > 0) ? (totalPromptTokens / ttftSec) : 0.0;
        const decodeTps = (decodeSec > 0 && actualCompletionTokens > 1) ? ((actualCompletionTokens - 1) / decodeSec) : 0.0;

        const lastStats = {
            ttftSec,
            decodeSec,
            promptTokens: totalPromptTokens,
            completionTokens: actualCompletionTokens,
            prefillTps,
            decodeTps
        };

        sessionStats.totalTurns++;
        sessionStats.totalPromptTokens += totalPromptTokens;
        sessionStats.totalCompletionTokens += actualCompletionTokens;
        sessionStats.totalTtftMs += ttftMs;
        sessionStats.totalDecodeTimeMs += decodeTimeMs;

        updateStatsUI(lastStats);

        if (reasoningBlock) {
            const summary = reasoningBlock.querySelector('summary');
            if (summary && summary.querySelector('.thinking-spinner')) {
                summary.innerHTML = 'Thought process';
            }
        }

    } catch (err) {
        if (err.name === 'AbortError') {
            console.log('Request aborted by user.');
            if (reasoningBlock && !isReasoningDone) {
                isReasoningDone = true;
                reasoningBlock.open = false;
                const summaryEl = reasoningBlock.querySelector('summary');
                if (summaryEl) summaryEl.innerHTML = 'Thought process (stopped)';
            }
            return;
        }
        console.error(err);
        mainContent.innerHTML += `<br><br><b>Error:</b> ${escapeHtml(err.message || "Failed to connect to engine. Make sure it's running.")}`;
    } finally {
        currentAbortController = null;
        setGeneratingState(false);
        const liveEl = assistantMsgDiv.querySelector('#live-status-indicator');
        if (liveEl && liveEl.parentElement) liveEl.remove();
        if (reasoningBlock) {
            const summary = reasoningBlock.querySelector('summary');
            if (summary && summary.querySelector('.thinking-spinner')) {
                summary.innerHTML = 'Thought process';
            }
        }
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

// Clean up any legacy Service Worker interceptors
function initProxyServiceWorker() {
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.getRegistrations().then(registrations => {
            for (const r of registrations) {
                r.unregister();
            }
        }).catch(err => {
            console.debug('[SW] ServiceWorker unregister error:', err);
        });
    }
}

// Run initialization on DOM load
document.addEventListener('DOMContentLoaded', () => {
    initProxyServiceWorker();
    initPreviewPanel();
    initExpertProfileUI();
    initAgenticSettingsUI();
});
