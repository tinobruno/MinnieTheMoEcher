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
    // Detect YouTube URLs in message and auto-load interactive player preview
    const ytMatch = rawText.match(/(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})/i);
    if (ytMatch) {
        const videoId = ytMatch[1];
        const ytDocId = 'yt_' + videoId;
        if (!retrievedDocsStore[ytDocId]) {
            const ytHtml = `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>YouTube Video Player</title>
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
      <div id="player"></div>
    </div>
    <div class="video-title">YouTube Video</div>
    <div class="video-actions">
      <a href="https://www.youtube.com/watch?v=${videoId}" target="_blank">Watch on YouTube &#x2197;</a>
    </div>
  </div>
  <script src="https://www.youtube.com/iframe_api"></script>
  <script>
    var player;
    function onYouTubeIframeAPIReady() {
      player = new YT.Player('player', {
        videoId: '${videoId}',
        playerVars: {
          'autoplay': 1,
          'playsinline': 1,
          'enablejsapi': 1,
          'rel': 0
        },
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
    }
    setTimeout(function() {
      var container = document.getElementById('player');
      if (container && container.tagName !== 'IFRAME') {
        container.innerHTML = '<iframe src="https://www.youtube-nocookie.com/embed/${videoId}?autoplay=1&enablejsapi=1&rel=0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen style="position:absolute;top:0;left:0;width:100%;height:100%;border:none;"></iframe>';
      }
    }, 2500);
  </script>
</body>
</html>`;
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
    
    // Check if already present by url
    const existingIdx = retrievedDocuments.findIndex(d => d.url === doc.url);
    const docObj = {
        id: docId,
        url: doc.url,
        title: doc.title || extractDomain(doc.url) || 'Retrieved Web Page',
        html: doc.html || '',
        snippet: doc.snippet || '',
        timeStr: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    if (existingIdx >= 0) {
        retrievedDocuments[existingIdx] = docObj;
    } else {
        retrievedDocuments.push(docObj);
    }
    retrievedDocsStore[docId] = docObj;

    renderRetrievedDocsList();

    // Automatically open preview and start autoplay for YouTube videos or media embeds
    const isYouTube = doc.url.includes('youtube.com') || doc.url.includes('youtu.be') || (doc.html && doc.html.includes('youtube-nocookie.com/embed'));
    if (isYouTube) {
        openDocInFullPreview(docId);
    }
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

        const isDocYouTube = doc.url.includes('youtube.com') || doc.url.includes('youtu.be') || (doc.html && doc.html.includes('youtube-nocookie.com/embed'));
        let miniContentHtml = '';
        if (isDocYouTube) {
            let ytId = '';
            const ytMatch = (doc.url + ' ' + (doc.html || '')).match(/(?:v=|youtu\.be\/|embed\/)([a-zA-Z0-9_-]{11})/);
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
            miniContentHtml = `<iframe class="retrieved-mini-iframe" sandbox="allow-same-origin" srcdoc="${safeSrcdoc}"></iframe>`;
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

    const htmlToLoad = doc.html || `<!DOCTYPE html><html><head><meta charset="UTF-8"><title>${escapeHtml(doc.title)}</title><style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;padding:30px;line-height:1.6;max-width:800px;margin:auto;color:#202124;}</style></head><body><h1>${escapeHtml(doc.title)}</h1><p><a href="${escapeHtml(doc.url)}" target="_blank">${escapeHtml(doc.url)}</a></p><hr/><p>${escapeHtml(doc.snippet)}</p></body></html>`;
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
    tools: {
        read_file: true,
        write_file: true,
        edit_file: true,
        execute_command: true,
        fetch_url: true
    },
    workspaceDir: ''
};

let currentPendingAuth = null;

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

        const savedPaths = localStorage.getItem('moecher_agentic_auth_paths');
        if (savedPaths) {
            try { agenticSettings.authorizedPaths = JSON.parse(savedPaths); } catch (e) {}
        }

        const savedTools = localStorage.getItem('moecher_agentic_tools');
        if (savedTools) {
            try { Object.assign(agenticSettings.tools, JSON.parse(savedTools)); } catch (e) {}
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
        localStorage.setItem('moecher_agentic_auth_paths', JSON.stringify(agenticSettings.authorizedPaths));
        localStorage.setItem('moecher_agentic_tools', JSON.stringify(agenticSettings.tools));
    } catch (e) {
        console.warn('Could not save agentic settings to localStorage', e);
    }
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

    // Tool Checkboxes
    const toolCheckboxes = {
        read_file: document.getElementById('tool-enable-read'),
        write_file: document.getElementById('tool-enable-write'),
        edit_file: document.getElementById('tool-enable-edit'),
        execute_command: document.getElementById('tool-enable-command'),
        fetch_url: document.getElementById('tool-enable-fetch')
    };

    Object.entries(toolCheckboxes).forEach(([toolName, cb]) => {
        if (cb) {
            cb.checked = agenticSettings.tools[toolName] !== false;
            cb.addEventListener('change', (e) => {
                agenticSettings.tools[toolName] = e.target.checked;
                saveAgenticSettings();
            });
        }
    });

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

    renderAuthorizedPathsTags();
    fetchWorkspaceInfo();
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

// Built-in tool definitions builder
function getActiveToolsPayload() {
    const isWebRetrieval = webRetrievalEnabled ? webRetrievalEnabled.checked : true;
    if (!isWebRetrieval) return [];

    const allToolDefs = [
        {
            name: "fetch_url",
            type: "function",
            function: {
                name: "fetch_url",
                description: "Fetch and extract readable text content from a public web URL (HTTP/HTTPS), search DuckDuckGo, or trigger YouTube playback.",
                parameters: {
                    type: "object",
                    properties: {
                        url: {
                            type: "string",
                            description: "The complete HTTP or HTTPS URL to fetch (e.g. https://html.duckduckgo.com/html/?q=query or https://example.com)"
                        }
                    },
                    required: ["url"]
                }
            }
        },
        {
            name: "read_file",
            type: "function",
            function: {
                name: "read_file",
                description: "Read the text contents of a file on the local filesystem. Supports line numbering and viewing line ranges.",
                parameters: {
                    type: "object",
                    properties: {
                        path: {
                            type: "string",
                            description: "The relative or absolute file path to read (e.g. 'src/main.cpp' or 'config.json')."
                        },
                        start_line: {
                            type: "integer",
                            description: "Optional 1-indexed starting line number (default: 1)."
                        },
                        end_line: {
                            type: "integer",
                            description: "Optional 1-indexed ending line number (default: -1 for entire file)."
                        }
                    },
                    required: ["path"]
                }
            }
        },
        {
            name: "write_file",
            type: "function",
            function: {
                name: "write_file",
                description: "Create a new file or completely overwrite an existing file with the provided text content.",
                parameters: {
                    type: "object",
                    properties: {
                        path: {
                            type: "string",
                            description: "The relative or absolute file path to write."
                        },
                        content: {
                            type: "string",
                            description: "The complete text content to write to the file."
                        },
                        overwrite: {
                            type: "boolean",
                            description: "Whether to overwrite if file already exists (default: true)."
                        }
                    },
                    required: ["path", "content"]
                }
            }
        },
        {
            name: "edit_file",
            type: "function",
            function: {
                name: "edit_file",
                description: "Perform a precise search-and-replace on a unique block of text within an existing file.",
                parameters: {
                    type: "object",
                    properties: {
                        path: {
                            type: "string",
                            description: "The relative or absolute file path to edit."
                        },
                        target_content: {
                            type: "string",
                            description: "The exact, unique block of lines to replace, matching whitespace."
                        },
                        replacement_content: {
                            type: "string",
                            description: "The new content that replaces the target block."
                        }
                    },
                    required: ["path", "target_content", "replacement_content"]
                }
            }
        },
        {
            name: "execute_command",
            type: "function",
            function: {
                name: "execute_command",
                description: "Execute a terminal/shell command on the local system (e.g. dir, ls, git, cargo, msbuild, cmake) and return its output.",
                parameters: {
                    type: "object",
                    properties: {
                        command: {
                            type: "string",
                            description: "The exact shell command line to execute."
                        }
                    },
                    required: ["command"]
                }
            }
        }
    ];

    return allToolDefs.filter(t => agenticSettings.tools[t.name] !== false).map(t => ({
        type: t.type,
        function: t.function
    }));
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

    // Build optimized messages payload with context pruning applied
    const messagesToSend = buildOptimizedMessagesPayload();

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
        reasoning_effort: isThinking ? reasoningEffort.value : "none",
        execution_timeout_sec: agenticSettings.timeoutSec || 60,
        workspace_boundary_enforced: agenticSettings.boundaryEnforced !== false,
        require_external_authorization: agenticSettings.requireAuth !== false,
        authorized_paths: agenticSettings.authorizedPaths || []
    };

    const activeTools = getActiveToolsPayload();
    if (activeTools.length > 0) {
        payload.tools = activeTools;
    }

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

                            if (delta.retrieved_document) {
                                addRetrievedDocument(delta.retrieved_document);
                            }

                            if (delta.authorization_required) {
                                showAuthPrompt(delta.authorization_required.tool, delta.authorization_required.path, delta.authorization_required.id);
                            }
                            
                            if (delta.reasoning_content !== undefined) {
                                if (firstTokenTime === null) firstTokenTime = performance.now();
                                tokenCount++;

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

                                // Check if delta is an in-place update to an existing active tool card
                                const toolIdMatch = delta.reasoning_content.match(/id="(tool-act-[^"]+)"/);
                                let replaced = false;
                                if (toolIdMatch) {
                                    const actId = toolIdMatch[1];
                                    const regex = new RegExp('<div class="tool-activity-block active"[^>]*id="' + actId + '"[\\s\\S]*?<\\/div>', 'g');
                                    if (regex.test(rawReasoning)) {
                                        rawReasoning = rawReasoning.replace(regex, delta.reasoning_content.trim());
                                        replaced = true;
                                    }
                                }
                                if (!replaced) {
                                    rawReasoning += delta.reasoning_content;
                                }
                                reasoningContent.innerHTML = marked.parse(rawReasoning);
                            } 
                            
                            if (delta.content !== undefined) {
                                if (firstTokenTime === null) firstTokenTime = performance.now();
                                tokenCount++;

                                if (liveIndicator && liveIndicator.parentElement) {
                                    liveIndicator.remove();
                                }

                                // Check if delta is an in-place update to an existing active tool card (when thinking is disabled)
                                const toolIdMatch = delta.content.match(/id="(tool-act-[^"]+)"/);
                                let replaced = false;
                                if (toolIdMatch) {
                                    const actId = toolIdMatch[1];
                                    const regex = new RegExp('<div class="tool-activity-block active"[^>]*id="' + actId + '"[\\s\\S]*?<\\/div>', 'g');
                                    if (regex.test(rawContent)) {
                                        rawContent = rawContent.replace(regex, delta.content.trim());
                                        replaced = true;
                                        renderMarkdownContent(rawContent, mainContent);
                                    }
                                }

                                if (!replaced) {
                                    // When final answer content arrives, close the reasoning block if open
                                    if (reasoningBlock && !isReasoningDone) {
                                        isReasoningDone = true;
                                        reasoningBlock.open = false;
                                        const summary = reasoningBlock.querySelector('summary');
                                        if (summary) summary.innerHTML = 'Thought process';
                                    }
                                    rawContent += delta.content;
                                    renderMarkdownContent(rawContent, mainContent);
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

        if (reasoningBlock) {
            const summary = reasoningBlock.querySelector('summary');
            if (summary && summary.querySelector('.thinking-spinner')) {
                summary.innerHTML = 'Thought process';
            }
        }

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
        const liveEl = assistantMsgDiv.querySelector('#live-status-indicator');
        if (liveEl && liveEl.parentElement) liveEl.remove();
        if (reasoningBlock) {
            const summary = reasoningBlock.querySelector('summary');
            if (summary && summary.querySelector('.thinking-spinner')) {
                summary.innerHTML = 'Thought process';
            }
        }
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
    initAgenticSettingsUI();
});
