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
    totalPrefillMs: 0,
    totalDecodeTimeMs: 0,
    totalToolTimeMs: 0,
    totalToolCalls: 0
};

function updateStatsUI(lastStats) {
    const elLastTtft = document.getElementById('stat-last-ttft');
    const elLastPrefill = document.getElementById('stat-last-prefill');
    const elLastDecode = document.getElementById('stat-last-decode');
    const elLastTokens = document.getElementById('stat-last-tokens');
    const elLastToolTime = document.getElementById('stat-last-tool-time');
    const elLastToolCalls = document.getElementById('stat-last-tool-calls');

    const elAvgTtft = document.getElementById('stat-avg-ttft');
    const elAvgPrefill = document.getElementById('stat-avg-prefill');
    const elAvgDecode = document.getElementById('stat-avg-decode');
    const elTotalTokens = document.getElementById('stat-total-tokens');
    const elAvgToolTime = document.getElementById('stat-avg-tool-time');
    const elTotalToolCalls = document.getElementById('stat-total-tool-calls');

    if (lastStats) {
        if (elLastTtft) elLastTtft.textContent = `${lastStats.ttftSec.toFixed(2)}s`;
        if (elLastPrefill) elLastPrefill.textContent = lastStats.prefillTps > 0 ? `${lastStats.prefillTps.toFixed(1)} t/s` : '-';
        if (elLastDecode) elLastDecode.textContent = lastStats.decodeTps > 0 ? `${lastStats.decodeTps.toFixed(1)} t/s` : '-';
        if (elLastTokens) elLastTokens.textContent = `${lastStats.completionTokens} tok`;
        if (elLastToolTime) {
            elLastToolTime.textContent = lastStats.toolCalls > 0 ? `${lastStats.toolTimeSec.toFixed(2)}s` : '0.00s';
        }
        if (elLastToolCalls) {
            elLastToolCalls.textContent = `${lastStats.toolCalls} call${lastStats.toolCalls === 1 ? '' : 's'}`;
        }
    } else {
        if (elLastTtft) elLastTtft.textContent = '-';
        if (elLastPrefill) elLastPrefill.textContent = '-';
        if (elLastDecode) elLastDecode.textContent = '-';
        if (elLastTokens) elLastTokens.textContent = '-';
        if (elLastToolTime) elLastToolTime.textContent = '-';
        if (elLastToolCalls) elLastToolCalls.textContent = '-';
    }

    if (sessionStats.totalTurns > 0) {
        const avgTtftSec = (sessionStats.totalTtftMs / sessionStats.totalTurns) / 1000.0;
        const totalPrefillSec = (sessionStats.totalPrefillMs || sessionStats.totalTtftMs) / 1000.0;
        const totalDecodeSec = sessionStats.totalDecodeTimeMs / 1000.0;
        const totalToolSec = (sessionStats.totalToolTimeMs || 0) / 1000.0;
        const avgToolSecPerTurn = totalToolSec / sessionStats.totalTurns;

        const avgPrefillTps = (totalPrefillSec > 0 && sessionStats.totalPromptTokens > 0)
            ? (sessionStats.totalPromptTokens / totalPrefillSec)
            : 0;
        const avgDecodeTps = (totalDecodeSec > 0 && sessionStats.totalCompletionTokens > 0)
            ? (sessionStats.totalCompletionTokens / totalDecodeSec)
            : 0;

        if (elAvgTtft) elAvgTtft.textContent = `${avgTtftSec.toFixed(2)}s`;
        if (elAvgPrefill) elAvgPrefill.textContent = avgPrefillTps > 0 ? `${avgPrefillTps.toFixed(1)} t/s` : '-';
        if (elAvgDecode) elAvgDecode.textContent = avgDecodeTps > 0 ? `${avgDecodeTps.toFixed(1)} t/s` : '-';
        if (elTotalTokens) elTotalTokens.textContent = `${sessionStats.totalCompletionTokens} tok (${sessionStats.totalTurns} turn${sessionStats.totalTurns === 1 ? '' : 's'})`;
        if (elAvgToolTime) {
            elAvgToolTime.textContent = sessionStats.totalToolCalls > 0
                ? `${totalToolSec.toFixed(2)}s (${avgToolSecPerTurn.toFixed(2)}s/t)`
                : '0.00s';
        }
        if (elTotalToolCalls) {
            elTotalToolCalls.textContent = `${sessionStats.totalToolCalls || 0} call${sessionStats.totalToolCalls === 1 ? '' : 's'}`;
        }
    } else {
        if (elAvgTtft) elAvgTtft.textContent = '-';
        if (elAvgPrefill) elAvgPrefill.textContent = '-';
        if (elAvgDecode) elAvgDecode.textContent = '-';
        if (elTotalTokens) elTotalTokens.textContent = '-';
        if (elAvgToolTime) elAvgToolTime.textContent = '-';
        if (elTotalToolCalls) elTotalToolCalls.textContent = '-';
    }
}

let isGenerating = false;

function setGeneratingState(generating) {
    isGenerating = generating;
    const previewBtn = document.getElementById('preview-btn-chatbar');
    const inputBox = document.querySelector('.input-box');
    const micBtn = document.getElementById('mic-btn');

    if (generating) {
        if (typeof isVoiceRecording !== 'undefined' && isVoiceRecording) {
            stopVoiceRecognition();
        }
        if (typeof stopTtsAudio === 'function') {
            stopTtsAudio();
        }
        sendBtn.classList.add('stop-mode');
        sendBtn.innerHTML = '<span class="material-symbols-outlined">stop</span>';
        sendBtn.title = 'Stop generation';
        sendBtn.disabled = false;
        if (micBtn) micBtn.disabled = true;
        chatInput.disabled = true;
        chatInput.placeholder = 'Generating response...';
        if (inputBox) inputBox.classList.add('disabled');
        if (previewBtn) previewBtn.classList.add('hidden');
    } else {
        sendBtn.classList.remove('stop-mode');
        sendBtn.innerHTML = '<span class="material-symbols-outlined">send</span>';
        sendBtn.title = 'Send message';
        if (micBtn) micBtn.disabled = false;
        chatInput.disabled = false;
        chatInput.placeholder = 'Message Minnie...';
        if (inputBox) inputBox.classList.remove('disabled');
        sendBtn.disabled = chatInput.value.trim() === '';
        updateChatbarPreviewButtonVisibility();
        chatInput.focus();
    }
}

function getApiBase() {
    if (typeof window !== 'undefined' && window.location && window.location.protocol.startsWith('http')) {
        if (window.location.port === '8001' || !window.location.port) {
            return window.location.origin;
        }
        return `${window.location.protocol}//${window.location.hostname}:8001`;
    }
    return 'http://localhost:8001';
}

function stopGeneration() {
    if (typeof stopTtsAudio === 'function') {
        stopTtsAudio();
    }
    if (typeof isVoiceRecording !== 'undefined' && isVoiceRecording) {
        stopVoiceRecognition();
    }
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

const repPenaltySlider = document.getElementById('rep-penalty-slider');
const repPenaltyVal = document.getElementById('rep-penalty-val');
if (repPenaltySlider && repPenaltyVal) {
    repPenaltySlider.addEventListener('input', (e) => {
        repPenaltyVal.textContent = parseFloat(e.target.value).toFixed(2);
    });
}

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
    const apiBase = getApiBase();
    fetch(`${apiBase}/api/kv/reset`, { method: 'POST' }).catch(() => { });
    chatHistory = [];
    sessionStats = {
        totalTurns: 0,
        totalPromptTokens: 0,
        totalCompletionTokens: 0,
        totalTtftMs: 0,
        totalPrefillMs: 0,
        totalDecodeTimeMs: 0,
        totalToolTimeMs: 0,
        totalToolCalls: 0
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

// ════════════════════════════════════════════════════════════════════════════════
//  Global Toast Notifications
// ════════════════════════════════════════════════════════════════════════════════

function showToast(message, type = 'info', duration = 4000) {
    if (!message || typeof document === 'undefined') return;
    let container = document.getElementById('moecher-toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'moecher-toast-container';
        container.className = 'moecher-toast-container';
        document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = `moecher-toast toast-${type}`;

    let iconName = 'info';
    if (type === 'error') iconName = 'error';
    else if (type === 'warn' || type === 'warning') iconName = 'warning';
    else if (type === 'success') iconName = 'check_circle';

    const safeMsg = (typeof escapeHtml === 'function') ? escapeHtml(message) : message;

    toast.innerHTML = `
        <span class="material-symbols-outlined toast-icon">${iconName}</span>
        <span class="toast-message">${safeMsg}</span>
        <button type="button" class="toast-close" title="Dismiss">&times;</button>
    `;

    const closeBtn = toast.querySelector('.toast-close');
    const dismiss = () => {
        toast.classList.remove('visible');
        setTimeout(() => { if (toast.parentElement) toast.remove(); }, 250);
    };
    if (closeBtn) closeBtn.onclick = (e) => { e.stopPropagation(); dismiss(); };
    toast.onclick = dismiss;

    container.appendChild(toast);
    requestAnimationFrame(() => {
        toast.classList.add('visible');
    });

    setTimeout(dismiss, duration);
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

// Dedicated Settings Panel Elements
const settingsPanel = document.getElementById('settings-panel');
const settingsToggleBtn = document.getElementById('settings-toggle-btn');
const settingsMaximizeBtn = document.getElementById('settings-maximize-btn');

let isPreviewOpen = false;
let isMaximized = false;
let isSettingsOpen = false;
let isSettingsMaximized = false;
let currentHtmlCode = '';
let consoleLogs = [];
let activeConsoleFilter = 'all';

// Initialize Panel State from localStorage
function initPreviewPanel() {
    const savedWidth = localStorage.getItem('moecher_preview_width');
    const panelWidth = (savedWidth && parseInt(savedWidth, 10) > 300) ? `${parseInt(savedWidth, 10)}px` : '540px';
    if (previewPanel) previewPanel.style.width = panelWidth;
    if (settingsPanel) settingsPanel.style.width = panelWidth;

    // Default closed unless explicitly opened
    if (previewPanel) previewPanel.classList.add('collapsed');
    if (settingsPanel) settingsPanel.classList.add('collapsed');
    if (resizerHandle) resizerHandle.classList.add('hidden');
    if (previewToggleBtn) previewToggleBtn.classList.remove('active');
    if (settingsToggleBtn) settingsToggleBtn.classList.remove('active');
    isPreviewOpen = false;
    isSettingsOpen = false;

    setupResizer();
    setupTabs();
    setupSettingsTabs();
    setupViewportControls();
    setupCodeEditor();
    setupConsoleListener();
}

function openPreviewPanel() {
    // Mutual exclusivity: Close settings panel if open
    if (isSettingsOpen) {
        closeSettingsPanel();
    }
    isPreviewOpen = true;
    if (previewPanel) previewPanel.classList.remove('collapsed');
    if (resizerHandle) resizerHandle.classList.remove('hidden');
    if (previewToggleBtn) previewToggleBtn.classList.add('active');
    localStorage.setItem('moecher_preview_open', 'true');
}

function closePreviewPanel() {
    isPreviewOpen = false;
    if (isMaximized) toggleMaximizePreview();
    if (previewPanel) previewPanel.classList.add('collapsed');
    if (!isSettingsOpen && resizerHandle) {
        resizerHandle.classList.add('hidden');
    }
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
        if (previewPanel) previewPanel.classList.add('maximized');
        if (resizerHandle) resizerHandle.classList.add('hidden');
        if (previewMaximizeBtn) {
            previewMaximizeBtn.innerHTML = '<span class="material-symbols-outlined">fullscreen_exit</span>';
            previewMaximizeBtn.title = 'Restore Panel Size';
        }
    } else {
        if (previewPanel) previewPanel.classList.remove('maximized');
        if (resizerHandle) resizerHandle.classList.remove('hidden');
        if (previewMaximizeBtn) {
            previewMaximizeBtn.innerHTML = '<span class="material-symbols-outlined">fullscreen</span>';
            previewMaximizeBtn.title = 'Maximize Panel';
        }
    }
}

// Dedicated Settings Panel Controls
function openSettingsPanel(tabId = null) {
    // Mutual exclusivity: Close HTML preview panel if open
    if (isPreviewOpen) {
        closePreviewPanel();
    }
    isSettingsOpen = true;
    if (settingsPanel) settingsPanel.classList.remove('collapsed');
    if (resizerHandle) resizerHandle.classList.remove('hidden');
    if (settingsToggleBtn) settingsToggleBtn.classList.add('active');
    if (tabId) {
        switchSettingsTab(tabId);
    }
    localStorage.setItem('moecher_settings_open', 'true');
}

function closeSettingsPanel() {
    isSettingsOpen = false;
    if (isSettingsMaximized) toggleMaximizeSettings();
    if (settingsPanel) settingsPanel.classList.add('collapsed');
    if (settingsToggleBtn) settingsToggleBtn.classList.remove('active');
    if (!isPreviewOpen && resizerHandle) {
        resizerHandle.classList.add('hidden');
    }
    localStorage.setItem('moecher_settings_open', 'false');
}

function toggleSettingsPanel() {
    if (isSettingsOpen) {
        closeSettingsPanel();
    } else {
        openSettingsPanel();
    }
}

function toggleMaximizeSettings() {
    if (!settingsPanel) return;
    isSettingsMaximized = !isSettingsMaximized;
    if (isSettingsMaximized) {
        settingsPanel.classList.add('maximized');
        if (resizerHandle) resizerHandle.classList.add('hidden');
        if (settingsMaximizeBtn) {
            settingsMaximizeBtn.innerHTML = '<span class="material-symbols-outlined">fullscreen_exit</span>';
            settingsMaximizeBtn.title = 'Restore Panel Size';
        }
    } else {
        settingsPanel.classList.remove('maximized');
        if (resizerHandle) resizerHandle.classList.remove('hidden');
        if (settingsMaximizeBtn) {
            settingsMaximizeBtn.innerHTML = '<span class="material-symbols-outlined">fullscreen</span>';
            settingsMaximizeBtn.title = 'Maximize Panel';
        }
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
        const activePanel = isSettingsOpen ? settingsPanel : previewPanel;
        startWidth = activePanel ? activePanel.getBoundingClientRect().width : 540;
        document.body.classList.add('resizing-active');
        if (resizerHandle) resizerHandle.classList.add('is-resizing');

        window.addEventListener('mousemove', onMouseMove);
        window.addEventListener('mouseup', onMouseUp);
        e.preventDefault();
    }

    function onMouseMove(e) {
        if (!isDragging) return;
        const delta = startX - e.clientX;
        const newWidth = Math.min(Math.max(startWidth + delta, 320), window.innerWidth - 320);
        if (previewPanel) previewPanel.style.width = `${newWidth}px`;
        if (settingsPanel) settingsPanel.style.width = `${newWidth}px`;
    }

    function onMouseUp() {
        if (!isDragging) return;
        isDragging = false;
        document.body.classList.remove('resizing-active');
        if (resizerHandle) resizerHandle.classList.remove('is-resizing');
        const activePanel = isSettingsOpen ? settingsPanel : previewPanel;
        if (activePanel) {
            localStorage.setItem('moecher_preview_width', parseInt(activePanel.style.width, 10));
        }
        window.removeEventListener('mousemove', onMouseMove);
        window.removeEventListener('mouseup', onMouseUp);
    }

    if (resizerHandle) {
        resizerHandle.addEventListener('mousedown', onMouseDown);
    }
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
    document.querySelectorAll('.preview-panel .tab-pane').forEach(p => p.classList.remove('active'));

    const activeBtn = document.querySelector(`.preview-tab[data-tab="${tabId}"]`);
    const activePane = document.getElementById(tabId);

    if (activeBtn) activeBtn.classList.add('active');
    if (activePane) activePane.classList.add('active');

    if (tabId === 'tab-code') {
        updateEditorLineNumbers();
    } else if (tabId === 'tab-3d') {
        if (typeof ThreeStudio !== 'undefined') {
            ThreeStudio.onTabActivated();
            if (ThreeStudio.isModelGroupEmpty()) {
                const code = currentHtmlCode || getLastTurnPreviewCode();
                if (code && is3DContent(code)) {
                    ThreeStudio.loadModelCode(code);
                }
            }
        }
    }
}

// Dedicated Settings Tabs
function setupSettingsTabs() {
    const tabButtons = document.querySelectorAll('.settings-tab');
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            switchSettingsTab(targetTab);
        });
    });
}

function switchSettingsTab(tabId) {
    document.querySelectorAll('.settings-tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.settings-tab-pane').forEach(p => p.classList.remove('active'));

    const activeBtn = document.querySelector(`.settings-tab[data-tab="${tabId}"]`);
    const activePane = document.getElementById(tabId);

    if (activeBtn) activeBtn.classList.add('active');
    if (activePane) activePane.classList.add('active');
}

function openAgenticSettingsTab() {
    openSettingsPanel('settings-tab-agentic');
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

    const is3d = is3DContent(currentHtmlCode);
    if (is3d && typeof ThreeStudio !== 'undefined') {
        ThreeStudio.loadModelCode(currentHtmlCode);
    }

    // Direct YouTube video player detection for 100% reliable hardware-accelerated playback
    const ytMatch = currentHtmlCode.match(/(?:youtube-nocookie\.com\/embed\/|youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})/i);
    if (ytMatch && (currentHtmlCode.includes('youtube-nocookie.com') || currentHtmlCode.includes('YouTube Video') || currentHtmlCode.includes('yt-container') || currentHtmlCode.includes('player'))) {
        const videoId = ytMatch[1];
        if (previewIframe) {
            const targetSrc = `https://www.youtube-nocookie.com/embed/${videoId}?autoplay=1&enablejsapi=1&rel=0`;
            const currentSrc = previewIframe.src || '';
            if (!currentSrc.includes(`/embed/${videoId}`)) {
                previewIframe.removeAttribute('srcdoc');
                previewIframe.setAttribute('allow', 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share');
                previewIframe.src = targetSrc;
            }
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
        if (is3d) {
            switchPreviewTab('tab-3d');
        } else {
            switchPreviewTab('tab-preview');
        }
    }
}

// Extract HTML or previewable code specifically from the LAST assistant turn
function getLastTurnPreviewCode() {
    const allAssistantMsgs = document.querySelectorAll('#messages-container .message.assistant');
    if (allAssistantMsgs.length === 0) return '';
    const lastAssistantMsg = allAssistantMsgs[allAssistantMsgs.length - 1];

    // Check code blocks in the last assistant turn
    const codeBlocks = lastAssistantMsg.querySelectorAll('pre code');
    for (let i = codeBlocks.length - 1; i >= 0; i--) {
        const code = codeBlocks[i].textContent || '';
        const lang = (codeBlocks[i].className || '').toLowerCase();
        if (isHtmlContent(code, lang) || is3DContent(code, lang)) {
            return code;
        }
    }

    // Check for explicit preview banner in the last assistant turn
    const banner = lastAssistantMsg.querySelector('.msg-html-banner');
    if (banner && currentHtmlCode && currentHtmlCode.trim().length > 0) {
        return currentHtmlCode;
    }

    // Check for interactive YouTube player link in the last assistant turn
    const ytLink = lastAssistantMsg.querySelector('a[href*="youtube.com"], a[href*="youtu.be"]');
    if (ytLink) {
        const href = ytLink.getAttribute('href') || '';
        const m = href.match(/(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})/i);
        if (m) {
            return createYouTubePlayerHtml(m[1], ytLink.textContent || 'YouTube Video');
        }
    }

    return '';
}

// Check if previewable content exists in the LAST assistant turn
function hasLastTurnPreviewableContent() {
    const code = getLastTurnPreviewCode();
    return !!(code && code.trim().length > 0);
}

// Backward compatibility aliases
function hasHtmlSnippet() {
    return hasLastTurnPreviewableContent();
}

function getLatestHtmlCode() {
    return getLastTurnPreviewCode();
}

// Extract HTML or 3D snippet specifically generated in the current assistant message
function getTurnHtmlCode(assistantMsgEl, rawText) {
    if (assistantMsgEl) {
        const codeBlocks = assistantMsgEl.querySelectorAll('pre code');
        for (let i = codeBlocks.length - 1; i >= 0; i--) {
            const code = codeBlocks[i].textContent || '';
            const lang = (codeBlocks[i].className || '').toLowerCase();
            if (isHtmlContent(code, lang) || is3DContent(code, lang)) {
                return code;
            }
        }
    }
    if (isFullHtmlDocument(rawText) || is3DContent(rawText)) {
        return rawText;
    }
    return '';
}

// Update visibility of the chat bar preview button (only show when previewable content exists in the LAST assistant turn)
function updateChatbarPreviewButtonVisibility() {
    const btn = document.getElementById('preview-btn-chatbar');
    if (!btn) return;
    if (!isGenerating && hasLastTurnPreviewableContent()) {
        btn.classList.remove('hidden');
    } else {
        btn.classList.add('hidden');
    }
}

// Chatbar Preview Button Click Handler - strictly previews content from the LAST assistant turn
function previewLatestHtmlSnippet() {
    const code = getLastTurnPreviewCode();
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

    const importMap = `
<script type="importmap">
{
  "imports": {
    "three": "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/jsm/",
    "three/examples/jsm/": "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/jsm/"
  }
}
</script>
`;

    let finalHtml = htmlCode;
    const injectContent = consoleBridge + (!finalHtml.includes('type="importmap"') && !finalHtml.includes("type='importmap'") ? importMap : '');
    if (finalHtml.includes('<head>')) {
        finalHtml = finalHtml.replace('<head>', '<head>' + injectContent);
    } else if (finalHtml.includes('<html>')) {
        finalHtml = finalHtml.replace('<html>', '<html><head>' + injectContent + '</head>');
    } else {
        finalHtml = injectContent + finalHtml;
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
    const is3d = is3DContent(currentHtmlCode);
    if (is3d && typeof ThreeStudio !== 'undefined') {
        ThreeStudio.loadModelCode(currentHtmlCode);
    }
    renderPreviewIframe(currentHtmlCode);
    if (previewEmptyState) previewEmptyState.classList.add('hidden');
    if (docStatusBadge) {
        docStatusBadge.textContent = 'Active';
        docStatusBadge.classList.remove('modified');
    }
    if (is3d) {
        switchPreviewTab('tab-3d');
    } else {
        switchPreviewTab('tab-preview');
    }
}

function run3DFromEditor() {
    if (!previewCodeEditor) return;
    currentHtmlCode = previewCodeEditor.value;
    if (typeof ThreeStudio !== 'undefined') {
        ThreeStudio.openWithCode(currentHtmlCode);
    }
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

        const is3dCandidate = is3DContent(codeContent, lang);

        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'code-block-actions';

        if (is3dCandidate) {
            const studio3dBtn = document.createElement('button');
            studio3dBtn.className = 'code-action-btn preview-btn-highlight btn-view-3d-model';
            studio3dBtn.innerHTML = `<span class="material-symbols-outlined btn-icon">view_in_ar</span> 3D Studio`;
            studio3dBtn.title = 'Open and interact with this 3D model in the 3D Studio';
            studio3dBtn.addEventListener('click', () => {
                ThreeStudio.openWithCode(codeContent);
            });
            actionsDiv.appendChild(studio3dBtn);
        }

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
            const is3d = is3DContent(rawText);
            banner.innerHTML = `
                <div class="msg-html-banner-left">
                    <span class="material-symbols-outlined msg-html-banner-icon">${is3d ? 'view_in_ar' : 'html'}</span>
                    <div>
                        <div class="msg-html-banner-text">${is3d ? '3D Model Scene Detected' : 'HTML Document Detected'}</div>
                        <div class="msg-html-banner-sub">${is3d ? 'Interact, rotate, and edit this 3D model in the 3D Studio' : 'Test and interact with this document in the preview panel'}</div>
                    </div>
                </div>
                <div style="display: flex; gap: 8px; align-items: center;">
                    ${is3d ? `
                    <button class="msg-html-banner-btn btn-open-3d">
                        <span class="material-symbols-outlined">view_in_ar</span>
                        <span>Open in 3D Studio</span>
                    </button>
                    ` : ''}
                    <button class="msg-html-banner-btn btn-open-preview">
                        <span class="material-symbols-outlined">play_circle</span>
                        <span>Open in Preview</span>
                    </button>
                </div>
            `;
            const previewBtn = banner.querySelector('.btn-open-preview');
            if (previewBtn) {
                previewBtn.addEventListener('click', () => {
                    loadHtmlIntoPreview(rawText, true);
                });
            }
            if (is3d) {
                const btn3d = banner.querySelector('.btn-open-3d');
                if (btn3d) {
                    btn3d.addEventListener('click', () => {
                        ThreeStudio.openWithCode(rawText);
                    });
                }
            }
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
    maxToolRounds: 10,
    compactToolOutputs: true,
    omitPastReasoning: false,
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
        youtube_search: true,
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
        "track", "tracks", "album", "clip", "clips", "audio", "soundtrack", "ost", "theme",
        "canto", "canzone", "canzoni", "musica", "suona", "ascolta", "videoclip", "trailer",
        "teaser", "movie", "podcast", "live", "concert", "concerto", "remix", "cover", "lyrics",
        "testo", "band", "singer", "artist", "cantante", "cantautore", "orchestra", "symphony",
        "instrumental", "acoustic", "stream", "show", "performance", "discography", "chords",
        "tab", "vlog", "gameplay", "tutorial", "walkthrough", "scene", "highlight", "highlights",
        "documentary", "short", "shorts", "official video", "music video", "ep", "lp", "single"
    ];
    if (kw.some(k => q.includes(k))) return true;

    // Check if the user's latest prompt in the active turn was requesting media playback
    if (Array.isArray(chatHistory) && chatHistory.length > 0) {
        for (let i = chatHistory.length - 1; i >= 0; i--) {
            const msg = chatHistory[i];
            if (msg.role === 'user') {
                const uContent = (msg.content || '').toLowerCase();
                const mediaTriggers = [
                    "play", "suona", "canzone", "song", "music", "musica", "listen", "ascolta",
                    "fammi sentire", "fammi ascoltare", "metti la canzone", "metti il pezzo", "video", "youtube"
                ];
                if (mediaTriggers.some(t => uContent.includes(t))) {
                    return true;
                }
                break;
            }
        }
    }

    return false;
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

        const savedMaxToolRounds = localStorage.getItem('moecher_agentic_max_tool_rounds');
        if (savedMaxToolRounds !== null) agenticSettings.maxToolRounds = parseInt(savedMaxToolRounds, 10) || 10;

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
        localStorage.setItem('moecher_agentic_max_tool_rounds', agenticSettings.maxToolRounds || 10);
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
    if (input.classList.contains('revealed')) {
        input.classList.remove('revealed');
        input.style.webkitTextSecurity = 'disc';
        icon.textContent = 'visibility';
    } else {
        input.classList.add('revealed');
        input.style.webkitTextSecurity = 'none';
        icon.textContent = 'visibility_off';
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

const webToolNames = ['web_search', 'youtube_search', 'fetch_url', 'google_search', 'create_3d_model'];
const localToolNames = ['read_file', 'write_file', 'edit_file', 'execute_command'];

function syncMasterCheckboxes() {
    const masterWebToggle = document.getElementById('tool-master-web');
    const masterLocalToggle = document.getElementById('tool-master-local');
    if (masterWebToggle) {
        masterWebToggle.checked = webToolNames.some(t => agenticSettings.tools[t] !== false);
    }
    if (masterLocalToggle) {
        masterLocalToggle.checked = localToolNames.some(t => agenticSettings.tools[t] !== false);
    }
}

function syncFastMediaSearchToggle(enabled) {
    agenticSettings.fastMediaSearch = !!enabled;
    agenticSettings.tools['youtube_search'] = !!enabled;

    const sidebarToggle = document.getElementById('sidebar-fast-media-search');
    const panelToggle = document.getElementById('fast-media-search-toggle');
    const ytCb = document.getElementById('tool-enable-youtube-search');

    if (sidebarToggle) sidebarToggle.checked = !!enabled;
    if (panelToggle) panelToggle.checked = !!enabled;
    if (ytCb) ytCb.checked = !!enabled;

    syncMasterCheckboxes();
    saveAgenticSettings();
    syncToolSettingsWithBackend();

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

    // Max Tool Rounds UI (Right Panel & Left Sidebar sync)
    const maxRoundsSlider = document.getElementById('agentic-max-rounds-slider');
    const maxRoundsInput = document.getElementById('agentic-max-rounds-input');
    const sideMaxRoundsSlider = document.getElementById('sidebar-max-tool-rounds-slider');
    const sideMaxRoundsVal = document.getElementById('sidebar-max-tool-rounds-val');

    const updateMaxToolRoundsUI = (val) => {
        agenticSettings.maxToolRounds = val;
        if (maxRoundsSlider) maxRoundsSlider.value = Math.min(val, 50);
        if (maxRoundsInput) maxRoundsInput.value = val;
        if (sideMaxRoundsSlider) sideMaxRoundsSlider.value = Math.min(val, 50);
        if (sideMaxRoundsVal) sideMaxRoundsVal.textContent = val;
        saveAgenticSettings();
    };

    updateMaxToolRoundsUI(agenticSettings.maxToolRounds || 10);

    if (maxRoundsSlider) {
        maxRoundsSlider.addEventListener('input', (e) => updateMaxToolRoundsUI(parseInt(e.target.value, 10)));
    }
    if (maxRoundsInput) {
        maxRoundsInput.addEventListener('input', (e) => {
            let val = parseInt(e.target.value, 10);
            if (isNaN(val) || val < 1) val = 1;
            if (val > 100) val = 100;
            updateMaxToolRoundsUI(val);
        });
    }
    if (sideMaxRoundsSlider) {
        sideMaxRoundsSlider.addEventListener('input', (e) => updateMaxToolRoundsUI(parseInt(e.target.value, 10)));
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

    const toolCheckboxes = {
        read_file: document.getElementById('tool-enable-read'),
        write_file: document.getElementById('tool-enable-write'),
        edit_file: document.getElementById('tool-enable-edit'),
        execute_command: document.getElementById('tool-enable-command'),
        web_search: document.getElementById('tool-enable-web-search'),
        youtube_search: document.getElementById('tool-enable-youtube-search'),
        google_search: document.getElementById('tool-enable-google-search'),
        fetch_url: document.getElementById('tool-enable-fetch'),
        create_3d_model: document.getElementById('tool-enable-3d-model')
    };

    if (masterWebToggle) {
        masterWebToggle.addEventListener('change', (e) => {
            const val = e.target.checked;
            webToolNames.forEach(t => {
                agenticSettings.tools[t] = val;
                if (toolCheckboxes[t]) toolCheckboxes[t].checked = val;
            });
            agenticSettings.fastMediaSearch = val;
            const sidebarToggle = document.getElementById('sidebar-fast-media-search');
            const panelToggle = document.getElementById('fast-media-search-toggle');
            if (sidebarToggle) sidebarToggle.checked = val;
            if (panelToggle) panelToggle.checked = val;
            saveAgenticSettings();
            syncToolSettingsWithBackend();
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
            syncToolSettingsWithBackend();
        });
    }

    Object.entries(toolCheckboxes).forEach(([toolName, cb]) => {
        if (cb) {
            cb.checked = agenticSettings.tools[toolName] !== false;
            cb.addEventListener('change', (e) => {
                agenticSettings.tools[toolName] = e.target.checked;
                if (toolName === 'youtube_search') {
                    agenticSettings.fastMediaSearch = e.target.checked;
                    const sidebarToggle = document.getElementById('sidebar-fast-media-search');
                    const panelToggle = document.getElementById('fast-media-search-toggle');
                    if (sidebarToggle) sidebarToggle.checked = e.target.checked;
                    if (panelToggle) panelToggle.checked = e.target.checked;
                }
                syncMasterCheckboxes();
                saveAgenticSettings();
                syncToolSettingsWithBackend();
            });
        }
    });

    if (webRetrievalEnabled) {
        webRetrievalEnabled.addEventListener('change', () => {
            syncToolSettingsWithBackend();
        });
    }

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
    if (window.serverToolsDisabled) return [];
    const isWebRetrieval = webRetrievalEnabled ? webRetrievalEnabled.checked : true;
    const knownTools = ['web_search', 'youtube_search', 'google_search', 'fetch_url', 'create_3d_model', 'read_file', 'write_file', 'edit_file', 'execute_command'];

    const activeList = knownTools.filter(name => {
        if ((name === 'web_search' || name === 'youtube_search' || name === 'google_search' || name === 'fetch_url') && !isWebRetrieval) {
            return false;
        }
        if (name === 'youtube_search' && agenticSettings.fastMediaSearch === false) {
            return false;
        }
        return agenticSettings.tools[name] !== false;
    });

    return activeList;
}

// ============================================================================
// System Prompt & Dynamic Tooling Prefix Synchronization
// ============================================================================

function setSystemPromptSyncStatus(status, text) {
    const el = document.getElementById('system-prompt-sync-status');
    if (!el) return;
    if (status === 'syncing') {
        el.innerHTML = `<span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #eab308;"></span> ${text || 'Updating engine prefix...'}`;
    } else if (status === 'synced') {
        el.innerHTML = `<span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #22c55e;"></span> ${text || 'Synced with engine'}`;
    } else if (status === 'error') {
        el.innerHTML = `<span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #ef4444;"></span> ${text || 'Sync error'}`;
    } else {
        el.innerHTML = `<span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #3b82f6;"></span> ${text || status}`;
    }
}

let isSyncingSystemPrompt = false;
let systemPromptDebounceTimer = null;
let toolSyncDebounceTimer = null;
let lastServerInstanceId = null;

function renderServerRestartNotice() {
    if (document.getElementById('server-reboot-notice')) return;
    const noticeDiv = document.createElement('div');
    noticeDiv.id = 'server-reboot-notice';
    noticeDiv.style.cssText = 'margin: 12px 16px; padding: 10px 14px; background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 8px; font-size: 12px; color: var(--text-primary); display: flex; align-items: center; justify-content: space-between; gap: 10px; z-index: 10;';
    noticeDiv.innerHTML = `
        <div style="display: flex; align-items: center; gap: 8px;">
            <span class="material-symbols-outlined" style="color: #38bdf8; font-size: 18px;">restart_alt</span>
            <span><strong>Engine Restarted:</strong> Conversation KV cache was cleared on server. Click <em>New Chat</em> for 0ms instant prefill, or continue to re-evaluate history.</span>
        </div>
        <div style="display: flex; gap: 6px; align-items: center;">
            <button onclick="clearChat(); const n = document.getElementById('server-reboot-notice'); if (n) n.remove();" class="btn btn-secondary" style="padding: 4px 10px; font-size: 11px; cursor: pointer; border-radius: 4px; border: 1px solid #38bdf8; color: #38bdf8;">New Chat (0ms)</button>
            <button onclick="this.closest('#server-reboot-notice').remove();" style="background: transparent; border: none; color: var(--text-secondary); cursor: pointer; font-size: 16px;">&times;</button>
        </div>
    `;
    if (messagesContainer) {
        messagesContainer.appendChild(noticeDiv);
        messagesContainer.scrollTo({ top: messagesContainer.scrollHeight, behavior: 'smooth' });
    }
}

async function fetchBackendSystemPrompt() {
    try {
        setSystemPromptSyncStatus('syncing', 'Fetching engine prompt...');
        const res = await fetch(`${getApiBase()}/api/system/prompt`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (data && data.tools_enabled === false) {
            window.serverToolsDisabled = true;
        } else if (data && data.tools_enabled === true) {
            window.serverToolsDisabled = false;
        }
        if (data && data.system_prompt !== undefined) {
            if (systemPromptInput) {
                systemPromptInput.value = data.system_prompt;
            }
            const count = (data.tokens_count !== undefined) ? data.tokens_count : data.num_tokens;
            const tokStr = count ? ` (${count} tokens)` : '';
            setSystemPromptSyncStatus('synced', `Synced with engine${tokStr}`);

            if (data.server_instance_id) {
                if (lastServerInstanceId && lastServerInstanceId !== data.server_instance_id && chatHistory && chatHistory.length > 0) {
                    console.info('[Engine] Server reboot detected. Conversation turn cache was reset.');
                    renderServerRestartNotice();
                }
                lastServerInstanceId = data.server_instance_id;
            }
        }
    } catch (e) {
        console.warn('Failed to fetch backend system prompt:', e);
        setSystemPromptSyncStatus('error', 'Sync offline');
    }
}

function syncToolSettingsWithBackend() {
    if (window.serverToolsDisabled) return;
    setSystemPromptSyncStatus('syncing', 'Updating prefix & tools...');
    if (toolSyncDebounceTimer) {
        clearTimeout(toolSyncDebounceTimer);
    }
    toolSyncDebounceTimer = setTimeout(async () => {
        try {
            isSyncingSystemPrompt = true;
            const activeTools = getActiveToolsPayload();
            const res = await fetch(`${getApiBase()}/api/system/configure`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    tools: activeTools
                })
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            if (data && data.system_prompt !== undefined) {
                if (systemPromptInput) {
                    systemPromptInput.value = data.system_prompt;
                }
            }
            const count = (data.tokens_count !== undefined) ? data.tokens_count : data.num_tokens;
            const tokStr = count ? ` (${count} tokens)` : '';
            setSystemPromptSyncStatus('synced', `Synced with engine${tokStr}`);
        } catch (err) {
            console.warn('Failed to sync tools with backend:', err);
            setSystemPromptSyncStatus('error', 'Sync failed');
        } finally {
            isSyncingSystemPrompt = false;
        }
    }, 150);
}

function handleSystemPromptInput() {
    if (isSyncingSystemPrompt) return;
    setSystemPromptSyncStatus('syncing', 'Syncing custom prompt...');
    if (systemPromptDebounceTimer) {
        clearTimeout(systemPromptDebounceTimer);
    }
    systemPromptDebounceTimer = setTimeout(async () => {
        try {
            const promptVal = systemPromptInput ? systemPromptInput.value : '';
            const activeTools = getActiveToolsPayload();
            const res = await fetch(`${getApiBase()}/api/system/configure`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    system_prompt: promptVal,
                    tools: activeTools
                })
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            const count = (data.tokens_count !== undefined) ? data.tokens_count : data.num_tokens;
            const tokStr = count ? ` (${count} tokens)` : '';
            setSystemPromptSyncStatus('synced', `Synced with engine${tokStr}`);
        } catch (err) {
            console.warn('Failed to sync custom system prompt:', err);
            setSystemPromptSyncStatus('error', 'Sync failed');
        }
    }, 500);
}

async function initSystemPromptSync() {
    if (systemPromptInput) {
        systemPromptInput.addEventListener('input', handleSystemPromptInput);
    }
    await fetchBackendSystemPrompt();

    if (!window.serverToolsDisabled) {
        const activeTools = getActiveToolsPayload();
        const allTools = ['web_search', 'youtube_search', 'google_search', 'fetch_url', 'create_3d_model', 'read_file', 'write_file', 'edit_file', 'execute_command'];
        const hasCustomToolConfig = (activeTools.length !== allTools.length);
        if (hasCustomToolConfig) {
            syncToolSettingsWithBackend();
        }
    }
    window.addEventListener('focus', () => fetchBackendSystemPrompt());
    setInterval(() => fetchBackendSystemPrompt(), 30000);
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
    let sysPrompt = systemPromptInput ? systemPromptInput.value.trim() : '';
    const activeTools = getActiveToolsPayload();

    if (window.serverToolsDisabled) {
        sysPrompt = "You are an assistant.";
        if (systemPromptInput) systemPromptInput.value = sysPrompt;
        messagesToSend.push({ role: "system", content: sysPrompt });
    } else if (sysPrompt) {
        // Strip any stale # Tools or <tools> block if activeTools has changed or youtube_search was disabled
        if (sysPrompt.includes('# Tools') || sysPrompt.includes('<tools>')) {
            const hasYt = activeTools.includes('youtube_search');
            if (activeTools.length === 0 || (!hasYt && sysPrompt.includes('youtube_search'))) {
                let toolsIdx = sysPrompt.indexOf('\n\n# Tools');
                if (toolsIdx === -1) toolsIdx = sysPrompt.indexOf('# Tools');
                if (toolsIdx === -1) toolsIdx = sysPrompt.indexOf('<tools>');
                if (toolsIdx !== -1) {
                    sysPrompt = sysPrompt.substring(0, toolsIdx).trim();
                }
            }
        }
        if (window.lastUserPromptWas3D && sysPrompt) {
            // Strip any short/stale 3D instructions from base system prompt
            let basicIdx = sysPrompt.indexOf('\n\n# 3D Modeling Instructions');
            if (basicIdx === -1) basicIdx = sysPrompt.indexOf('# 3D Modeling Instructions');
            if (basicIdx === -1) basicIdx = sysPrompt.indexOf('\n\n## 3D Modeling Instructions');
            if (basicIdx === -1) basicIdx = sysPrompt.indexOf('## 3D Modeling Instructions');
            if (basicIdx !== -1) {
                sysPrompt = sysPrompt.substring(0, basicIdx).trim();
            }
            sysPrompt += "\n\n# High-Fidelity Analytic 3D Modeling Instructions\n" +
                "When asked to create, model, or reconstruct in 3D (or reconstruct an object from an image), always output Three.js representation code defining `function createModel(scene, THREE, inputImage, helpers) { ... }` inside a ```javascript code block (never output a full HTML file), adding all meshes to `scene`.\n\n" +
                "CRITICAL RULES FOR ANALYTIC MODELING, PART-SPECIFIC TEXTURING & UV MAPPING:\n" +
                "1. SEMANTIC & GEOMETRIC PART DECOMPOSITION:\n" +
                "   - Never model an object as a single monolithic blob. Analytically decompose the subject into its distinct structural and anatomical parts (e.g. for an apple: body, stem, leaf; for a teapot: vessel body, lid, knob, handle, spout, foot ring; for furniture/vehicles/shoes: separate functional components).\n" +
                "   - Model EVERY part as a separate named `THREE.Mesh` with its own geometry and material. Name every mesh descriptively (`mesh.name = 'apple_body'`, `mesh.name = 'apple_stem'`) or use `helpers.createPart(name, geometry, material)` and add all parts to `scene`.\n\n" +
                "2. PART-SPECIFIC TEXTURES & COLORS (DO NOT NAIVELY APPLY THE ENTIRE PHOTO!):\n" +
                "   - NEVER blindly apply the entire input photo onto the whole model or primary mesh (which stretches background, shadows, and unrelated features across surfaces).\n" +
                "   - For parts requiring surface textures: use `helpers.cropTexture(inputImage, uMin, vMin, uMax, vMax, options)` to crop clean sub-regions of the photo matching that part (e.g. cropped skin texture for body, cropped leaf texture for leaf, cropped label for bottle). Coordinates are normalized 0..1.\n" +
                "   - For uniform, solid, or metallic parts: use `helpers.sampleColor(inputImage, u, v)` to sample the exact realistic color from the image and set appropriate PBR properties (roughness, metalness, clearcoat).\n\n" +
                "3. ACCURATE UV MAPPING:\n" +
                "   - Revolved / lathe / cylindrical parts: use cylindrical/lathe UVs (`helpers.applyCylindricalUV(geometry)` or standard Lathe UVs) with `wrapS: THREE.RepeatWrapping`.\n" +
                "   - Flat / curved / extruded parts (leaves, labels, panels, wings): use `helpers.applyPlanarUV(geometry, 'z')` so cropped sub-textures map cleanly without distortion.\n" +
                "   - Multi-sided / cubic parts: use `helpers.applyBoxUV(geometry)`.\n\n" +
                "4. WATERTIGHT GEOMETRY & SEAMLESS INTEGRATION:\n" +
                "   - Revolved profiles must start and end at x=0 (use `helpers.createWatertightLathe` or `helpers.createHollowVessel` for vessels with solid wall thickness).\n" +
                "   - Tubes, pipes, stems, and handles must have sealed end caps (use `helpers.createCappedTube`).\n" +
                "   - Attachments (stems, leaves, handles, spouts) must penetrate 5-10% deep into parent meshes to prevent floating seams or gaps.";
        }
        if (sysPrompt) {
            messagesToSend.push({ role: 'system', content: sysPrompt });
        }
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

        if (msg.tool_calls && Array.isArray(msg.tool_calls)) {
            // Only preserve tool calls for tools that are currently enabled
            const allowedCalls = msg.tool_calls.filter(tc => {
                const name = (tc && tc.function && tc.function.name) ? tc.function.name : (tc && tc.name ? tc.name : '');
                if (!name) return true;
                return activeTools.includes(name) || name.startsWith('mcp__') || name.startsWith('tinobruno-');
            });
            if (allowedCalls.length > 0) {
                cleanedMsg.tool_calls = allowedCalls;
            }
        } else if (msg.tool_calls) {
            cleanedMsg.tool_calls = msg.tool_calls;
        }

        // If explicitly requested, preserve past reasoning; otherwise omit past reasoning traces
        // to avoid injecting thousands of redundant <think> tokens into subsequent turns.
        if (agenticSettings.omitPastReasoning === false && msg.reasoning_content) {
            cleanedMsg.reasoning_content = msg.reasoning_content;
        }

        // Compact past bulky tool outputs from older turns (skip the immediate prior turn to preserve KV cache prefix)
        const isImmediatePriorTurn = (idx >= historySlice.length - 3);
        if (agenticSettings.compactToolOutputs !== false && !isCurrentActiveTurn && !isImmediatePriorTurn) {
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
    // Standard tool calls
    out = out.replace(/<\s*tool_call>[\s\S]*?<\/\s*tool_call>/gi, '');
    out = out.replace(/<\/?\s*tool_calls?>/gi, '');

    // DeepSeek special tokens & DSML
    out = out.replace(/<\s*[｜|]?tool call begin[｜|]>[\s\S]*?<\s*[｜|]?tool call end[｜|]>/gi, '');
    out = out.replace(/<\s*[｜|]?DSML[｜|]?[^>]*>[\s\S]*?<\/\s*[｜|]?DSML[｜|]?[^>]*>/gi, '');
    out = out.replace(/<\/?\s*[｜|]?DSML[｜|]?[^>]*>/gi, '');
    out = out.replace(/<\s*[｜|]?tool (?:call begin|call end|sep|outputs begin|outputs end)[｜|]?>/gi, '');

    // Raw function call JSON
    out = out.replace(/\{"name":\s*"[^"]+"[\s\S]*?\}/g, '');
    out = out.replace(/\{"function":\s*"[^"]+"[\s\S]*?\}/g, '');
    return out.trim();
}

function isRawToolCallString(text) {
    if (!text) return false;
    const trimmed = text.trim();
    if (trimmed.startsWith('<tool_call>') || trimmed.startsWith('<｜tool call begin｜>') || trimmed.startsWith('<|tool call begin|>')) return true;
    if (trimmed.startsWith('<｜DSML') || trimmed.startsWith('<|DSML') || trimmed.startsWith('<DSML') || trimmed.startsWith('< DSML') || trimmed.startsWith('< tool')) return true;
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
            // Defers preview playback until actual content streaming starts
            resolve({
                output: `[YouTube Video: https://www.youtube.com/watch?v=${videoId} | Video player registered for preview panel. Do not fetch web page.]`,
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
    const activeTools = getActiveToolsPayload();
    if (tc && tc.name && !activeTools.includes(tc.name) && !tc.name.startsWith('mcp__') && !tc.name.startsWith('tinobruno-')) {
        console.warn(`[Tool Execution Guard] Tool '${tc.name}' is disabled in settings. Refusing execution.`);
        return `Error: Tool '${tc.name}' is disabled in user settings and cannot be executed.`;
    }

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
        let isMedia = (tc.name === 'youtube_search') || ((agenticSettings.fastMediaSearch !== false) && isMediaSearchQuery(query));

        // Disambiguate accidental youtube_search calls for general search requests (e.g. "search Tino Bruno", "who is ...")
        if (tc.name === 'youtube_search' && !isMediaSearchQuery(query)) {
            const latestUserMsg = (chatHistory.filter(m => m.role === 'user').slice(-1)[0]?.content || '').trim().toLowerCase();
            const hasMediaKeywords = isMediaSearchQuery(latestUserMsg);
            if (!hasMediaKeywords && (latestUserMsg.startsWith('search') || latestUserMsg.startsWith('cerca') || latestUserMsg.startsWith('who is') || latestUserMsg.startsWith('chi è') || latestUserMsg.startsWith('find'))) {
                console.log('[Tool Dispatcher] Overriding accidental youtube_search -> web_search for general search query:', query);
                isMedia = false;
            }
        }

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
                if (ytData && ytData.output) {
                    if (ytData.retrieved_document) {
                        addRetrievedDocument(ytData.retrieved_document);
                        if (turnRetrievedDocs) turnRetrievedDocs.push(ytData.retrieved_document);
                        // Defers preview playback until actual content streaming starts
                    }
                    return ytData.output;
                }
            } catch (ytErr) {
                console.warn('[YouTube Direct Search] Execution error:', ytErr);
            }

            // Dedicated youtube_search tool should NEVER fall through to general web search provider (e.g. Tavily)
            if (tc.name === 'youtube_search') {
                return `No YouTube videos found for query: "${query}".`;
            }
        }

        const provider = args.provider || agenticSettings.searchProvider || 'tavily';
        const tavilyKey = args.tavily_api_key || args.api_key || agenticSettings.tavilyApiKey || '';

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
        if (!args.api_key && agenticSettings.tavilyApiKey) args.api_key = agenticSettings.tavilyApiKey;
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
    } else if (tc.name === 'create_3d_model' || tc.name === 'model_3d') {
        let args = {};
        try {
            args = typeof tc.arguments === 'string' ? JSON.parse(tc.arguments) : (tc.arguments || {});
        } catch (e) {
            args = { code: tc.arguments };
        }
        const modelCode = args.code || '';
        const modelName = args.name || '3D Model';
        if (modelCode) {
            window.lastUserPromptWas3D = true;
            loadHtmlIntoPreview(modelCode, true);
            return `Successfully generated and loaded 3D model "${modelName}" into 3D Studio.`;
        }
        return `Error: No 3D model code provided.`;
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

    if (typeof isVoiceRecording !== 'undefined' && isVoiceRecording) {
        stopVoiceRecognition();
    }
    if (typeof stopTtsAudio === 'function') {
        stopTtsAudio();
    }

    // Immediately hide inline preview button on sending a new message
    const previewBtn = document.getElementById('preview-btn-chatbar');
    if (previewBtn) previewBtn.classList.add('hidden');

    if (currentAbortController) {
        currentAbortController.abort();
    }
    currentAbortController = new AbortController();

    chatInput.value = '';
    chatInput.style.height = 'auto';
    setGeneratingState(true);
    // Detect if user prompt is asking for 3D model creation
    const is3dPrompt = /(?:\b3d\b|three\.?js|\bmodel\b|\bmesh\b|\bobj\b|\bgltf\b)/i.test(text) ||
                       (activeImageAttachment && /(?:model|3d|mesh|render|create)/i.test(text));
    window.lastUserPromptWas3D = is3dPrompt;
    if (is3dPrompt) {
        if (!isPreviewOpen) openPreviewPanel();
        switchPreviewTab('tab-3d');
    }

    // Index where this turn starts in chatHistory
    const turnHistoryStartIndex = chatHistory.length;

    // Add user message (with optional image attachment)
    let userMsgContent = text;
    let attachedImg = null;
    if (activeImageAttachment) {
        attachedImg = Object.assign({}, activeImageAttachment);
        window.__lastUploadedImage = attachedImg.dataUrl;
        if (!window.__lastUploadedImageElement || window.__lastUploadedImageElement.src !== attachedImg.dataUrl) {
            const cachedImg = new Image();
            cachedImg.crossOrigin = 'anonymous';
            cachedImg.src = attachedImg.dataUrl;
            window.__lastUploadedImageElement = cachedImg;
        }
        let sendText = text ? text.trim() : "";
        let lower = sendText.toLowerCase();
        if (!sendText || lower === "describe this image" || lower === "describe the image" || lower === "describe image" || lower === "what is this" || lower === "what is this?" || lower === "what's this") {
            sendText = "What is shown in this image?";
        }
        userMsgContent = [
            { type: "text", text: sendText },
            { type: "image_url", image_url: { url: attachedImg.dataUrl } }
        ];
        removeImageAttachment();
        if (typeof ThreeStudio !== 'undefined' && ThreeStudio.updatePhotoTextureButtons) {
            ThreeStudio.updatePhotoTextureButtons();
        }
    }
    appendMessage('user', text || "Describe this image", attachedImg);
    chatHistory.push({ role: 'user', content: userMsgContent });

    // Create assistant message container
    const assistantMsgDiv = createMessageContainer('assistant');
    messagesContainer.appendChild(assistantMsgDiv);

    // Add reasoning block (hidden initially)
    let reasoningBlock = null;
    let reasoningContent = null;
    let mainContent = document.createElement('div');
    assistantMsgDiv.querySelector('.msg-content').appendChild(mainContent);

    const isThinking = attachedImg ? false : (thinkingEnabled ? thinkingEnabled.checked : true);

    // Initial visual feedback while engine starts thinking/elaborating
    const liveIndicator = document.createElement('div');
    liveIndicator.className = 'elaboration-status-badge';
    liveIndicator.id = 'live-status-indicator';
    liveIndicator.innerHTML = `
        <span class="tool-pulse-spinner"></span>
        <span class="status-msg-text">${isThinking ? 'Thinking and preparing response' : (attachedImg ? 'Analyzing image and preparing response' : 'Preparing response')}</span>
        <div class="elaboration-dots"><span></span><span></span><span></span></div>
    `;
    mainContent.appendChild(liveIndicator);
    const budgetVal = isThinking ? (thinkingBudget ? parseInt(thinkingBudget.value, 10) : 4096) : 0;
    const activeTools = getActiveToolsPayload();

    const startTime = performance.now();
    let totalPromptTokens = 0;
    let totalCompletionTokens = 0;
    let turnRetrievedDocs = [];
    let isReasoningDone = false;
    let turnMediaPreviewLoaded = false;
    let lastRoundContent = "";

    // Engine performance & tool execution timing tracking
    let round1TtftMs = null;
    let totalPrefillMs = 0;
    let totalDecodeMs = 0;
    let turnToolTimeMs = 0;
    let turnToolCallsCount = 0;

    const maxRounds = agenticSettings.maxToolRounds || 10;
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
            const roundStartTime = performance.now();
            let roundFirstTokenTime = null;
            const messagesToSend = buildOptimizedMessagesPayload();

            const payload = {
                model: currentModelId || "deepseek-v4-flash",
                messages: messagesToSend,
                max_tokens: parseInt(tokensInput ? tokensInput.value : 20000, 10) || 20000,
                temperature: parseFloat(tempSlider.value),
                repetition_penalty: parseFloat(repPenaltySlider ? repPenaltySlider.value : 1.10),
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
                max_tool_rounds: agenticSettings.maxToolRounds || 10,
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

            payload.tools = activeTools;

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

                                if (delta.tool_limit_reached || delta.tool_warning) {
                                    const procBadge = assistantMsgDiv.querySelector('#tool-proc-indicator');
                                    if (procBadge) procBadge.remove();
                                    showToast(delta.tool_warning || `Maximum tool execution rounds reached (${delta.max_tool_rounds || 10}).`);
                                }

                                if (delta.processing_status) {
                                    const target = (reasoningContent && !isReasoningDone) ? reasoningContent : mainContent;
                                    let procBadge = assistantMsgDiv.querySelector('#tool-proc-indicator');
                                    if (!procBadge) {
                                        procBadge = document.createElement('div');
                                        procBadge.className = 'tool-activity-block active tool-processing-badge';
                                        procBadge.id = 'tool-proc-indicator';
                                        target.appendChild(procBadge);
                                    }
                                    procBadge.innerHTML = `
                                        <span class="thinking-spinner">progress_activity</span>
                                        <span class="tool-action-label">${escapeHtml(delta.processing_status)}</span>
                                        <div class="elaboration-dots"><span></span><span></span><span></span></div>
                                    `;
                                    messagesContainer.scrollTo({
                                        top: messagesContainer.scrollHeight,
                                        behavior: 'smooth'
                                    });
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

                                if (delta.status !== undefined) {
                                    const statusText = liveIndicator ? liveIndicator.querySelector('.status-msg-text') : null;
                                    if (statusText) statusText.textContent = delta.status;
                                    let procBadge = assistantMsgDiv.querySelector('#tool-proc-indicator');
                                    if (!procBadge) {
                                        procBadge = document.createElement('div');
                                        procBadge.className = 'tool-activity-block active tool-processing-badge';
                                        procBadge.id = 'tool-proc-indicator';
                                        const targetContainer = (reasoningBlock && !isReasoningDone) ? reasoningContent : mainContent;
                                        targetContainer.appendChild(procBadge);
                                    }
                                    procBadge.innerHTML = `
                                        <span class="thinking-spinner">progress_activity</span>
                                        <span class="tool-action-label">${escapeHtml(delta.status)}</span>
                                    `;
                                    messagesContainer.scrollTo({
                                        top: messagesContainer.scrollHeight,
                                        behavior: 'smooth'
                                    });
                                }

                                if (delta.reasoning_content !== undefined) {
                                    const procBadge = assistantMsgDiv.querySelector('#tool-proc-indicator');
                                    if (procBadge) procBadge.remove();

                                    if (roundFirstTokenTime === null) {
                                        roundFirstTokenTime = performance.now();
                                        if (round1TtftMs === null) {
                                            round1TtftMs = Math.max(0, roundFirstTokenTime - roundStartTime);
                                        }
                                    }

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
                                    reasoningContent.innerHTML = marked.parse(stripToolCallsFromText(roundReasoning));
                                }

                                if (delta.content !== undefined) {
                                    const procBadge = assistantMsgDiv.querySelector('#tool-proc-indicator');
                                    if (procBadge) procBadge.remove();

                                    // If media was found by tools in this turn, launch preview and autoplay ONLY NOW (on content, not in reasoning)
                                    if (!turnMediaPreviewLoaded && turnRetrievedDocs.length > 0) {
                                        for (let i = turnRetrievedDocs.length - 1; i >= 0; i--) {
                                            const d = turnRetrievedDocs[i];
                                            if (d && d.html && (d.html.includes('youtube-nocookie.com') || d.html.includes('<video') || d.html.includes('<audio') || d.html.includes('YouTube Video'))) {
                                                loadHtmlIntoPreview(d.html, true);
                                                turnMediaPreviewLoaded = true;
                                                break;
                                            }
                                        }
                                    }

                                    if (roundFirstTokenTime === null) {
                                        roundFirstTokenTime = performance.now();
                                        if (round1TtftMs === null) {
                                            round1TtftMs = Math.max(0, roundFirstTokenTime - roundStartTime);
                                        }
                                    }

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

            const roundStreamEndTime = performance.now();
            const roundPrefillMs = (roundFirstTokenTime !== null ? roundFirstTokenTime : roundStreamEndTime) - roundStartTime;
            const roundDecodeMs = roundFirstTokenTime !== null ? Math.max(0, roundStreamEndTime - roundFirstTokenTime) : 0;
            totalPrefillMs += roundPrefillMs;
            totalDecodeMs += roundDecodeMs;

            const validToolCalls = roundToolCalls.filter(tc => tc && tc.name);

            if (validToolCalls.length > 0 && (roundFinishReason === 'tool_calls' || roundFinishReason === 'stop')) {
                const toolExecStartTime = performance.now();
                turnToolCallsCount += validToolCalls.length;
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
                        name: tc.name,
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
                        const parsedArgs = typeof tc.arguments === 'string' ? JSON.parse(tc.arguments) : (tc.arguments || {});
                        const queryStr = parsedArgs.query || parsedArgs.q || '';
                        let isDirectMedia = (tc.name === 'youtube_search' && activeTools.includes('youtube_search') && agenticSettings.fastMediaSearch !== false) ||
                            ((tc.name === 'web_search' || tc.name === 'google_search') && activeTools.includes(tc.name) && (agenticSettings.fastMediaSearch !== false) && isMediaSearchQuery(queryStr));
                        if (tc.name === 'youtube_search' && !isMediaSearchQuery(queryStr)) {
                            const latestUserMsg = (chatHistory.filter(m => m.role === 'user').slice(-1)[0]?.content || '').trim().toLowerCase();
                            const hasMediaKeywords = isMediaSearchQuery(latestUserMsg);
                            if (!hasMediaKeywords && (latestUserMsg.startsWith('search') || latestUserMsg.startsWith('cerca') || latestUserMsg.startsWith('who is') || latestUserMsg.startsWith('chi è') || latestUserMsg.startsWith('find'))) {
                                isDirectMedia = false;
                            }
                        }
                        if (isDirectMedia) {
                            toolTarget = queryStr;
                            actionLabel = 'YouTube Searching';
                            doneIcon = 'smart_display';
                            completedLabel = 'YouTube Searched';
                        } else if (tc.name === 'web_search' || tc.name === 'google_search' || tc.name === 'youtube_search') {
                            toolTarget = queryStr;
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
                        } else if (tc.name && tc.name.startsWith('mcp__')) {
                            const parts = tc.name.split('__');
                            const sId = parts[1] || 'mcp';
                            const rawTName = parts.slice(2).join('__') || tc.name;
                            toolTarget = typeof parsedArgs === 'object' ? JSON.stringify(parsedArgs) : String(parsedArgs || '');
                            if (toolTarget.length > 80) toolTarget = toolTarget.substring(0, 80) + '...';
                            actionLabel = `[MCP: ${sId}] ${rawTName}`;
                            doneIcon = 'hub';
                            completedLabel = `[MCP: ${sId}] Completed`;
                        } else if (tc.name) {
                            toolTarget = typeof parsedArgs === 'object' ? JSON.stringify(parsedArgs) : String(parsedArgs || '');
                            if (toolTarget.length > 80) toolTarget = toolTarget.substring(0, 80) + '...';
                            actionLabel = `Calling ${tc.name}`;
                            doneIcon = 'build';
                            completedLabel = `Executed ${tc.name}`;
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

                const toolExecEndTime = performance.now();
                turnToolTimeMs += Math.max(0, toolExecEndTime - toolExecStartTime);

                // Show active elaboration badge while the backend prefill/encodes the tool output into KV cache
                const activeToolTargetContainer = (reasoningContent && !isReasoningDone) ? reasoningContent : mainContent;
                let procBadge = assistantMsgDiv.querySelector('#tool-proc-indicator');
                if (!procBadge) {
                    procBadge = document.createElement('div');
                    procBadge.className = 'tool-activity-block active tool-processing-badge';
                    procBadge.id = 'tool-proc-indicator';
                    activeToolTargetContainer.appendChild(procBadge);
                }
                procBadge.innerHTML = `
                    <span class="thinking-spinner">progress_activity</span>
                    <span class="tool-action-label">Encoding tool output &amp; elaborating response</span>
                    <div class="elaboration-dots"><span></span><span></span><span></span></div>
                `;
                messagesContainer.scrollTo({
                    top: messagesContainer.scrollHeight,
                    behavior: 'smooth'
                });

                continue;
            }

            chatHistory.push({
                role: 'assistant',
                content: roundContent,
                reasoning_content: roundReasoning || undefined
            });

            // --- Turn Media & Preview Detection ---
            let turnMediaDoc = null;
            const isWebEnabled = webRetrievalEnabled ? webRetrievalEnabled.checked : true;
            const canPlayMedia = (agenticSettings.fastMediaSearch !== false) && (agenticSettings.tools['youtube_search'] !== false) && isWebEnabled;

            if (canPlayMedia) {
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
            }

            // 4. Check if turn generated an explicit HTML code block or web app
            const turnHtml = getTurnHtmlCode(assistantMsgDiv, roundContent);

            if (turnMediaDoc && turnMediaDoc.html) {
                loadHtmlIntoPreview(turnMediaDoc.html, true);
            } else if (turnHtml && turnHtml.trim().length > 0) {
                const is3dTurn = is3DContent(turnHtml);
                loadHtmlIntoPreview(turnHtml, is3dTurn && window.lastUserPromptWas3D);
            }

            lastRoundContent = roundContent;
            break;
        }

        const ttftMs = (round1TtftMs !== null ? round1TtftMs : totalPrefillMs);
        const ttftSec = ttftMs / 1000.0;
        const prefillSec = totalPrefillMs / 1000.0;
        const decodeSec = totalDecodeMs / 1000.0;
        const toolTimeSec = turnToolTimeMs / 1000.0;

        const actualCompletionTokens = totalCompletionTokens > 0 ? totalCompletionTokens : 1;
        const prefillTps = (prefillSec > 0 && totalPromptTokens > 0) ? (totalPromptTokens / prefillSec) : 0.0;
        const decodeTps = (decodeSec > 0 && actualCompletionTokens > 1) 
            ? ((actualCompletionTokens - 1) / decodeSec) 
            : (decodeSec > 0 ? (actualCompletionTokens / decodeSec) : 0.0);

        const lastStats = {
            ttftSec,
            prefillSec,
            decodeSec,
            promptTokens: totalPromptTokens,
            completionTokens: actualCompletionTokens,
            prefillTps,
            decodeTps,
            toolTimeSec,
            toolCalls: turnToolCallsCount
        };

        sessionStats.totalTurns++;
        sessionStats.totalPromptTokens += totalPromptTokens;
        sessionStats.totalCompletionTokens += actualCompletionTokens;
        sessionStats.totalTtftMs += ttftMs;
        sessionStats.totalPrefillMs += totalPrefillMs;
        sessionStats.totalDecodeTimeMs += totalDecodeMs;
        sessionStats.totalToolTimeMs += turnToolTimeMs;
        sessionStats.totalToolCalls += turnToolCallsCount;

        updateStatsUI(lastStats);

        if (reasoningBlock) {
            const summary = reasoningBlock.querySelector('summary');
            if (summary && summary.querySelector('.thinking-spinner')) {
                summary.innerHTML = 'Thought process';
            }
        }

        // Attach interactive message actions (Read aloud & Copy)
        if (typeof attachAssistantMessageActions === 'function') {
            attachAssistantMessageActions(assistantMsgDiv, lastRoundContent);
        }

        // If Voice Mode / Auto-read is enabled, speak the assistant's answer
        if (typeof voiceSettings !== 'undefined' && voiceSettings.autoRead && lastRoundContent) {
            speakAssistantMessage(lastRoundContent, assistantMsgDiv);
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
        const procEl = assistantMsgDiv.querySelector('#tool-proc-indicator');
        if (procEl && procEl.parentElement) procEl.remove();
        if (reasoningBlock) {
            const summary = reasoningBlock.querySelector('summary');
            if (summary && summary.querySelector('.thinking-spinner')) {
                summary.innerHTML = 'Thought process';
            }
        }
        updateChatbarPreviewButtonVisibility();
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

function appendMessage(role, text, attachedImg = null) {
    const div = createMessageContainer(role);
    if (role === 'user') {
        const contentEl = div.querySelector('.msg-content');
        contentEl.textContent = '';
        if (attachedImg && attachedImg.dataUrl) {
            const thumbCard = document.createElement('div');
            thumbCard.className = 'attachment-thumb-card user-msg-attachment';
            thumbCard.style.marginBottom = '8px';
            thumbCard.innerHTML = `
                <img src="${attachedImg.dataUrl}" alt="Attached thumbnail" style="width: 56px; height: 56px; object-fit: cover; border-radius: 8px;">
                <div class="attachment-meta">
                    <span style="font-size: 12px; font-weight: 500; color: #fff;">${escapeHtml(attachedImg.filename || 'image.png')}</span>
                    <span style="font-size: 11px; color: var(--text-muted);">${escapeHtml(attachedImg.filesize || '')}</span>
                </div>
            `;
            contentEl.appendChild(thumbCard);
        }
        const textSpan = document.createElement('div');
        textSpan.textContent = text;
        contentEl.appendChild(textSpan);
    } else {
        const contentEl = div.querySelector('.msg-content');
        renderMarkdownContent(text, contentEl);
        if (typeof attachAssistantMessageActions === 'function') {
            attachAssistantMessageActions(div, text);
        }
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

// ── Model Selector & Dropdown ──────────────────────────────────────────────────

let currentModelId = "deepseek-v4-flash";
let currentModelName = "DeepSeek V4-Flash";
let currentModelData = null;

async function initModelSelector() {
    const selectorEl = document.getElementById('model-selector');
    const dropdownEl = document.getElementById('model-dropdown');
    const modelNameEl = document.getElementById('model-name');
    const dropdownModelName = document.getElementById('dropdown-model-name');
    const dropdownModelArch = document.getElementById('dropdown-model-arch');
    const dropdownModelId = document.getElementById('dropdown-model-id');
    const dropdownModelCtx = document.getElementById('dropdown-model-ctx');
    const dropdownVisionBadge = document.getElementById('dropdown-model-vision');
    const dropdownVisionText = document.getElementById('dropdown-vision-text');
    const dropdownVramText = document.getElementById('dropdown-vram-text');
    const dropdownRamText = document.getElementById('dropdown-ram-text');
    const dropdownSsdText = document.getElementById('dropdown-ssd-text');
    const dropdownResVram = document.getElementById('dropdown-res-vram');
    const dropdownResRam = document.getElementById('dropdown-res-ram');
    const dropdownResSsd = document.getElementById('dropdown-res-ssd');

    if (!selectorEl || !modelNameEl) return;

    function formatGb(val) {
        if (val === undefined || val === null || isNaN(val)) return '--';
        if (val >= 1000) {
            return (val / 1024).toFixed(1) + ' TB';
        }
        return val.toFixed(1) + ' GB';
    }

    async function fetchModelInfo() {
        try {
            const res = await fetch(`${getApiBase()}/v1/models`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            if (data && data.data && data.data.length > 0) {
                const active = data.data.find(m => m.active) || data.data[0];
                if (active) {
                    currentModelId = active.id;
                    currentModelName = active.display_name || active.name || active.id;
                    currentModelData = active;

                    modelNameEl.textContent = currentModelName;
                    if (dropdownModelName) dropdownModelName.textContent = currentModelName;
                    if (dropdownModelId) dropdownModelId.textContent = active.id;
                    if (dropdownModelArch && active.architecture) dropdownModelArch.textContent = active.architecture;
                    if (dropdownModelCtx && active.max_context_length) {
                        dropdownModelCtx.textContent = `${Number(active.max_context_length).toLocaleString()} ctx`;
                    }
                    currentModelHasVision = (active.has_vision === true || active.has_vision === 'true');
                    updateVisionUploadVisibility();

                    // Update Vision capabilities badge
                    if (dropdownVisionBadge && dropdownVisionText) {
                        if (currentModelHasVision) {
                            dropdownVisionBadge.className = 'model-badge-vision enabled';
                            dropdownVisionText.textContent = 'Vision Enabled';
                            dropdownVisionBadge.title = 'Multimodal visual perception and OCR enabled';
                        } else {
                            dropdownVisionBadge.className = 'model-badge-vision disabled';
                            dropdownVisionText.textContent = 'Vision Disabled';
                            dropdownVisionBadge.title = 'Text-only model profile';
                        }
                    }

                    // Update System Resources (VRAM, RAM, SSD)
                    let resources = active.resources || data.resources;
                    if (!resources) {
                        try {
                            const sysRes = await fetch(`${getApiBase()}/api/system/resources`);
                            if (sysRes.ok) resources = await sysRes.json();
                        } catch (_) {}
                    }

                    if (resources) {
                        if (resources.vram && dropdownVramText) {
                            dropdownVramText.textContent = `${formatGb(resources.vram.free_gb)} free`;
                            if (dropdownResVram) {
                                dropdownResVram.title = `GPU VRAM: ${resources.vram.free_gb.toFixed(1)} GB free / ${resources.vram.total_gb.toFixed(1)} GB total (${resources.vram.used_gb.toFixed(1)} GB used)`;
                            }
                        }
                        if (resources.ram && dropdownRamText) {
                            dropdownRamText.textContent = `${formatGb(resources.ram.available_gb)} free`;
                            if (dropdownResRam) {
                                dropdownResRam.title = `System RAM: ${resources.ram.available_gb.toFixed(1)} GB available / ${resources.ram.total_gb.toFixed(1)} GB total`;
                            }
                        }
                        if (resources.storage && dropdownSsdText) {
                            dropdownSsdText.textContent = `${formatGb(resources.storage.available_gb)} free`;
                            if (dropdownResSsd) {
                                dropdownResSsd.title = `SSD Storage: ${formatGb(resources.storage.available_gb)} available / ${formatGb(resources.storage.total_gb)} total`;
                            }
                        }
                    }
                }
            }
        } catch (e) {
            console.debug('[Model] Could not fetch active model from /v1/models:', e);
        }
    }

    // Toggle dropdown
    selectorEl.addEventListener('click', (e) => {
        e.stopPropagation();
        if (!dropdownEl) return;
        const isHidden = dropdownEl.classList.contains('hidden');
        if (isHidden) {
            dropdownEl.classList.remove('hidden');
            selectorEl.classList.add('open');
            fetchModelInfo();
        } else {
            dropdownEl.classList.add('hidden');
            selectorEl.classList.remove('open');
        }
    });

    selectorEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            selectorEl.click();
        } else if (e.key === 'Escape' && dropdownEl && !dropdownEl.classList.contains('hidden')) {
            dropdownEl.classList.add('hidden');
            selectorEl.classList.remove('open');
        }
    });

    document.addEventListener('click', (e) => {
        if (dropdownEl && !dropdownEl.classList.contains('hidden')) {
            if (!dropdownEl.contains(e.target) && !selectorEl.contains(e.target)) {
                dropdownEl.classList.add('hidden');
                selectorEl.classList.remove('open');
            }
        }
    });

    // Initial fetch
    await fetchModelInfo();
}

// ── MCP (Model Context Protocol) UI Management ────────────────────────────────

let mcpServersList = [];

async function fetchMCPServers() {
    try {
        const res = await fetch(`${getApiBase()}/v1/mcp/servers`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (data && data.data) {
            mcpServersList = data.data;
            renderMCPServers(mcpServersList);
        }
    } catch (e) {
        console.debug('[MCP UI] Could not fetch servers:', e);
    }
}

function renderMCPServers(servers) {
    const listEl = document.getElementById('mcp-server-list');
    const activeCountEl = document.getElementById('mcp-active-servers-count');
    const totalToolsEl = document.getElementById('mcp-total-tools-count');

    if (!listEl) return;

    let activeCount = 0;
    let totalTools = 0;

    servers.forEach(s => {
        if (s.running) activeCount++;
        totalTools += (s.tools_count || (s.tools ? s.tools.length : 0));
    });

    if (activeCountEl) activeCountEl.textContent = String(activeCount);
    if (totalToolsEl) totalToolsEl.textContent = String(totalTools);

    if (!servers || servers.length === 0) {
        listEl.innerHTML = `
            <div class="mcp-empty-state">
                <span class="material-symbols-outlined" style="font-size: 32px; color: var(--text-muted); opacity: 0.6;">dns</span>
                <p>No MCP servers configured yet.</p>
                <span style="font-size: 11px; color: var(--text-muted);">Click "Add Server" to connect standard MCP servers (Filesystem, SQLite, Memory, GitHub, etc.)</span>
            </div>
        `;
        return;
    }

    let html = '';
    for (const s of servers) {
        const statusClass = !s.enabled ? 'disabled' : (s.running ? 'running' : 'stopped');
        const statusTitle = !s.enabled ? 'Disabled' : (s.running ? (s.pid > 0 ? `Running (PID ${s.pid})` : 'Connected (Remote)') : (s.error ? `Error: ${s.error}` : 'Stopped'));
        const argsStr = Array.isArray(s.args) ? s.args.join(' ') : String(s.args || '');
        const fullCmd = s.url ? `[${s.transport_type || 'streamable-http'}] ${s.url}` : `${s.command} ${argsStr}`.trim();
        const tools = s.tools || [];

        let toolsHtml = '';
        if (tools.length > 0) {
            toolsHtml = `
                <div class="mcp-server-tools-section">
                    <div class="mcp-tools-label">
                        <span class="material-symbols-outlined" style="font-size: 14px;">build</span>
                        <span>Exposed Tools (${tools.length})</span>
                    </div>
                    <div class="mcp-tools-grid">
                        ${tools.map(t => `<span class="mcp-tool-pill" title="${escapeHtml(t.description || '')}">${escapeHtml(t.name)}</span>`).join('')}
                    </div>
                </div>
            `;
        }

        html += `
            <div class="mcp-server-card ${statusClass}" id="mcp-card-${escapeHtml(s.id)}" onclick="editMCPServer('${escapeHtml(s.id)}')" title="Click to view or edit server parameters">
                <div class="mcp-server-header">
                    <div class="mcp-server-title-group">
                        <span class="mcp-status-dot ${statusClass}" title="${escapeHtml(statusTitle)}"></span>
                        <span class="mcp-server-name">${escapeHtml(s.id)}</span>
                        ${s.running && s.pid > 0 ? `<span class="mcp-server-pid">PID ${s.pid}</span>` : (s.running && s.url ? `<span class="mcp-server-pid" style="color: #93c5fd; background: rgba(59, 130, 246, 0.15);">Remote</span>` : '')}
                        ${s.error ? `<span class="mcp-server-pid" style="color: #fca5a5; background: rgba(239, 68, 68, 0.15);" title="${escapeHtml(s.error)}">Error</span>` : ''}
                    </div>
                    <div class="mcp-server-actions">
                        <button type="button" class="mcp-action-icon-btn edit" onclick="event.stopPropagation(); editMCPServer('${escapeHtml(s.id)}')" title="Edit Parameters & Settings">
                            <span class="material-symbols-outlined" style="font-size: 16px;">tune</span>
                        </button>
                        <button type="button" class="mcp-action-icon-btn" onclick="event.stopPropagation(); restartMCPServer('${escapeHtml(s.id)}')" title="Restart Server">
                            <span class="material-symbols-outlined" style="font-size: 16px;">restart_alt</span>
                        </button>
                        <input type="checkbox" class="agentic-toggle" ${s.enabled ? 'checked' : ''} onclick="event.stopPropagation();" onchange="toggleMCPServer('${escapeHtml(s.id)}', this.checked)" title="${s.enabled ? 'Disable Server' : 'Enable Server'}">
                        <button type="button" class="mcp-action-icon-btn delete" onclick="event.stopPropagation(); deleteMCPServer('${escapeHtml(s.id)}')" title="Delete Server">
                            <span class="material-symbols-outlined" style="font-size: 16px;">delete</span>
                        </button>
                    </div>
                </div>
                ${s.description ? `<div class="mcp-server-desc">${escapeHtml(s.description)}</div>` : ''}
                <div class="mcp-server-cmd" title="${escapeHtml(fullCmd)}">${escapeHtml(fullCmd)}</div>
                ${toolsHtml}
                <div class="mcp-card-footer-hint">
                    <span class="material-symbols-outlined" style="font-size: 13px;">tune</span>
                    <span>Click card to configure parameters</span>
                </div>
            </div>
        `;
    }

    listEl.innerHTML = html;
}

async function refreshMCPServersUI() {
    await fetchMCPServers();
}

async function toggleMCPServer(id, enabled) {
    try {
        const res = await fetch(`${getApiBase()}/v1/mcp/servers/toggle`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id, enabled })
        });
        const data = await res.json();
        if (data && data.servers) {
            mcpServersList = data.servers;
            renderMCPServers(mcpServersList);
        } else {
            await fetchMCPServers();
        }
    } catch (e) {
        console.error('[MCP UI] Error toggling server:', e);
        await fetchMCPServers();
    }
}

async function restartMCPServer(id) {
    try {
        const res = await fetch(`${getApiBase()}/v1/mcp/servers/restart`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id })
        });
        const data = await res.json();
        if (data && data.servers) {
            mcpServersList = data.servers;
            renderMCPServers(mcpServersList);
        } else {
            await fetchMCPServers();
        }
    } catch (e) {
        console.error('[MCP UI] Error restarting server:', e);
        await fetchMCPServers();
    }
}

async function deleteMCPServer(id) {
    if (!confirm(`Are you sure you want to remove MCP server '${id}'?`)) return;
    try {
        const res = await fetch(`${getApiBase()}/v1/mcp/servers?id=${encodeURIComponent(id)}`, {
            method: 'DELETE'
        });
        const data = await res.json();
        if (data && data.servers) {
            mcpServersList = data.servers;
            renderMCPServers(mcpServersList);
        } else {
            await fetchMCPServers();
        }
    } catch (e) {
        console.error('[MCP UI] Error deleting server:', e);
        await fetchMCPServers();
    }
}

let currentMCPModalMode = 'form';
let currentParsedMCPConfig = null;

const SAMPLE_TEAMS_MCP_JSON = `{
  "server": {
    "$schema": "https://static.modelcontextprotocol.io/schemas/2025-10-17/server.schema.json",
    "name": "com.microsoft/workiq-teamsserver",
    "description": "Manage Microsoft Teams chats, channels, users, and messages via Graph API.",
    "title": "Work IQ Teams MCP Server",
    "repository": {
      "url": "https://github.com/bap-microsoft/MCP-Platform/",
      "source": "github"
    },
    "version": "1.0.0",
    "remotes": [
      {
        "type": "streamable-http",
        "url": "https://agent365.svc.cloud.microsoft/agents/tenants/{tenant_id}/servers/mcp_TeamsServer",
        "variables": {
          "tenant_id": {
            "description": "Microsoft Entra tenant ID",
            "isRequired": true
          }
        }
      }
    ]
  },
  "_meta": {
    "io.modelcontextprotocol.registry/official": {
      "status": "active",
      "statusChangedAt": "2026-03-25T09:26:37.371959Z",
      "publishedAt": "2026-03-25T09:26:37.371959Z",
      "updatedAt": "2026-03-25T09:26:37.371959Z",
      "isLatest": true
    }
  }
}`;

let currentRegistryServers = [];
let searchRegistryDebounceTimer = null;

function switchMCPModalMode(mode) {
    currentMCPModalMode = mode;
    const searchTabBtn = document.getElementById('mcp-tab-btn-search');
    const formTabBtn = document.getElementById('mcp-tab-btn-form');
    const jsonTabBtn = document.getElementById('mcp-tab-btn-json');
    const searchPane = document.getElementById('mcp-modal-search-pane');
    const formPane = document.getElementById('mcp-modal-form-pane');
    const jsonPane = document.getElementById('mcp-modal-json-pane');
    const jsonInput = document.getElementById('mcp-server-json-input');
    const saveBtn = document.getElementById('mcp-save-btn');

    // Reset tab active states
    if (searchTabBtn) searchTabBtn.classList.remove('active');
    if (formTabBtn) formTabBtn.classList.remove('active');
    if (jsonTabBtn) jsonTabBtn.classList.remove('active');

    // Reset pane display states
    if (searchPane) searchPane.style.display = 'none';
    if (formPane) formPane.style.display = 'none';
    if (jsonPane) jsonPane.style.display = 'none';

    if (mode === 'search') {
        if (searchTabBtn) searchTabBtn.classList.add('active');
        if (searchPane) searchPane.style.display = 'block';
        if (saveBtn) saveBtn.style.display = 'none';

        // Auto trigger search if container is empty
        const resultsContainer = document.getElementById('mcp-registry-results-container');
        if (resultsContainer && resultsContainer.children.length === 0) {
            performMCPRegistrySearch();
        }
    } else if (mode === 'json') {
        if (jsonTabBtn) jsonTabBtn.classList.add('active');
        if (jsonPane) jsonPane.style.display = 'block';
        if (saveBtn) saveBtn.style.display = 'inline-flex';

        // If json input is empty, populate from form fields
        if (jsonInput && !jsonInput.value.trim()) {
            const id = document.getElementById('mcp-server-id-input')?.value.trim() || 'custom-server';
            const transport = document.getElementById('mcp-transport-type-select')?.value || 'stdio';
            const cmd = document.getElementById('mcp-server-cmd-input')?.value.trim() || '';
            const rawArgs = document.getElementById('mcp-server-args-input')?.value.trim() || '';
            const url = document.getElementById('mcp-server-url-input')?.value.trim() || '';
            const desc = document.getElementById('mcp-server-desc-input')?.value.trim() || '';
            let args = [];
            if (rawArgs.startsWith('[') && rawArgs.endsWith(']')) {
                try { args = JSON.parse(rawArgs); } catch (e) { args = rawArgs.split(/\s+/).filter(Boolean); }
            } else if (rawArgs) {
                args = rawArgs.split(/\s+/).filter(Boolean);
            }

            let env = {};
            try {
                const rawEnv = document.getElementById('mcp-server-env-input')?.value.trim();
                if (rawEnv) env = JSON.parse(rawEnv);
            } catch (e) {}

            let sampleObj;
            if (transport === 'streamable-http' || transport === 'sse' || url) {
                sampleObj = {
                    server: {
                        name: id,
                        title: desc || id,
                        description: desc,
                        remotes: [
                            {
                                type: transport,
                                url: url || "https://..."
                            }
                        ]
                    }
                };
            } else {
                sampleObj = {
                    mcpServers: {
                        [id]: {
                            command: cmd || "npx",
                            args: args.length > 0 ? args : ["-y", "@modelcontextprotocol/server-filesystem", "."],
                            env: env,
                            description: desc
                        }
                    }
                };
            }
            jsonInput.value = JSON.stringify(sampleObj, null, 2);
        }
        onMCPJsonInputChange();
    } else {
        // mode === 'form'
        if (formTabBtn) formTabBtn.classList.add('active');
        if (formPane) formPane.style.display = 'block';
        if (saveBtn) saveBtn.style.display = 'inline-flex';

        // If JSON input has valid parsed data, sync to form fields
        if (currentParsedMCPConfig) {
            populateFormFromParsedMCP(currentParsedMCPConfig);
        }
    }
}

async function performMCPRegistrySearch(query) {
    const searchInput = document.getElementById('mcp-registry-search-input');
    const statusEl = document.getElementById('mcp-registry-search-status');
    const statusTextEl = document.getElementById('mcp-registry-search-text');
    const statusIconEl = document.getElementById('mcp-registry-search-icon');
    const container = document.getElementById('mcp-registry-results-container');

    if (query === undefined) {
        query = searchInput ? searchInput.value.trim() : '';
    } else if (searchInput && searchInput.value !== query) {
        searchInput.value = query;
    }

    // If user pasted a full URL from registry.modelcontextprotocol.io
    if (query.includes('?q=')) {
        query = query.split('?q=')[1].split('&')[0];
    } else if (query.includes('&q=')) {
        query = query.split('&q=')[1].split('&')[0];
    } else if (query.startsWith('http')) {
        try {
            const urlObj = new URL(query);
            query = urlObj.searchParams.get('q') || urlObj.searchParams.get('search') || '';
        } catch (e) {}
    }
    query = decodeURIComponent(query).trim();

    if (statusEl) {
        statusEl.style.display = 'flex';
        statusEl.className = 'mcp-json-status info';
        if (statusIconEl) statusIconEl.textContent = 'progress_activity';
        if (statusTextEl) statusTextEl.textContent = query 
            ? `Searching registry for "${query}"...` 
            : 'Loading featured MCP servers from registry...';
    }

    try {
        const res = await fetch(`${getApiBase()}/v1/mcp/registry/search?q=${encodeURIComponent(query)}`);
        if (!res.ok) {
            if (res.status === 404) {
                throw new Error(`Server endpoint /v1/mcp/registry/search returned HTTP 404. Please restart the backend server with the newly built binary.`);
            }
            throw new Error(`Registry search service responded with HTTP ${res.status}`);
        }
        const data = await res.json();
        const servers = data.servers || (Array.isArray(data) ? data : []);
        currentRegistryServers = servers;

        if (statusEl) {
            statusEl.className = 'mcp-json-status valid';
            if (statusIconEl) statusIconEl.textContent = 'check_circle';
            if (statusTextEl) {
                statusTextEl.textContent = `Found ${servers.length} server${servers.length === 1 ? '' : 's'} in registry${query ? ` matching "${query}"` : ''}`;
            }
        }

        renderMCPRegistryResults(servers);
    } catch (e) {
        console.error('[MCP Registry Search Error]', e);
        if (statusEl) {
            statusEl.className = 'mcp-json-status error';
            if (statusIconEl) statusIconEl.textContent = 'error';
            if (statusTextEl) statusTextEl.textContent = `Search error: ${e.message}. Check network connection.`;
        }
        if (container) {
            container.innerHTML = `
                <div style="text-align: center; padding: 20px; color: #f87171; font-size: 12px;">
                    Failed to contact registry.modelcontextprotocol.io (${escapeHtml(e.message)}).
                    <div style="margin-top: 8px;">
                        <button type="button" class="editor-btn" onclick="performMCPRegistrySearch()" style="font-size: 11px;">Retry</button>
                    </div>
                </div>
            `;
        }
    }
}

function onMCPRegistrySearchInput() {
    clearTimeout(searchRegistryDebounceTimer);
    searchRegistryDebounceTimer = setTimeout(() => {
        performMCPRegistrySearch();
    }, 350);
}

function searchRegistryTag(tag) {
    const input = document.getElementById('mcp-registry-search-input');
    if (input) input.value = tag;
    performMCPRegistrySearch(tag);
}

function renderMCPRegistryResults(servers) {
    const container = document.getElementById('mcp-registry-results-container');
    if (!container) return;

    if (!Array.isArray(servers) || servers.length === 0) {
        container.innerHTML = `
            <div style="text-align: center; padding: 28px 16px; color: var(--text-muted);">
                <span class="material-symbols-outlined" style="font-size: 36px; opacity: 0.5; margin-bottom: 6px;">search_off</span>
                <p style="font-size: 13px; margin: 0 0 4px; color: var(--text-secondary);">No MCP servers found.</p>
                <span style="font-size: 11px;">Try searching for popular servers like "teams", "github", "filesystem", "sqlite", or "fetch".</span>
            </div>
        `;
        return;
    }

    let html = '';
    servers.forEach((item, index) => {
        const s = item.server || item;
        const meta = item._meta || {};
        const title = s.title || s.name || 'Unnamed Server';
        const name = s.name || '';
        const desc = s.description || 'No description provided';
        const version = s.version ? `v${s.version}` : '';

        // Determine transport badges
        let badgesHtml = '';
        if (Array.isArray(s.remotes) && s.remotes.length > 0) {
            s.remotes.forEach(r => {
                const t = r.type || 'remote';
                const cls = t === 'streamable-http' ? 'http' : (t === 'sse' ? 'sse' : 'http');
                badgesHtml += `<span class="mcp-badge-transport ${cls}">${escapeHtml(t)}</span>`;
            });
        }
        if (Array.isArray(s.packages) && s.packages.length > 0) {
            s.packages.forEach(p => {
                const reg = (p.registryType || 'cli').toLowerCase();
                badgesHtml += `<span class="mcp-badge-transport stdio">${escapeHtml(reg)}</span>`;
            });
        }
        if (!badgesHtml) {
            badgesHtml = `<span class="mcp-badge-transport stdio">stdio</span>`;
        }

        const repoUrl = s.repository?.url || '';
        const repoHtml = repoUrl ? `
            <a href="${escapeHtml(repoUrl)}" target="_blank" rel="noopener noreferrer" class="mcp-reg-link" title="Open repository in new tab">
                <span class="material-symbols-outlined" style="font-size: 13px;">open_in_new</span>
                <span>GitHub / Repo</span>
            </a>
        ` : '<span></span>';

        html += `
            <div class="mcp-registry-card" id="mcp-reg-card-${index}">
                <div class="mcp-registry-card-header">
                    <div>
                        <span class="mcp-reg-title">${escapeHtml(title)}</span>
                        ${version ? `<span style="font-size: 10px; color: var(--text-muted); margin-left: 6px; font-family: monospace;">${escapeHtml(version)}</span>` : ''}
                        <div class="mcp-reg-name">${escapeHtml(name)}</div>
                    </div>
                    <div class="mcp-reg-badges">
                        ${badgesHtml}
                    </div>
                </div>
                <div class="mcp-reg-desc">${escapeHtml(desc)}</div>
                <div class="mcp-reg-footer">
                    ${repoHtml}
                    <button type="button" class="mcp-reg-select-btn" onclick="selectMCPRegistryServer(${index})" title="Configure and install this server">
                        <span class="material-symbols-outlined" style="font-size: 14px;">check_circle</span>
                        <span>Select & Configure</span>
                    </button>
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}

function selectMCPRegistryServer(index) {
    if (!currentRegistryServers || !currentRegistryServers[index]) return;
    const item = currentRegistryServers[index];

    // Create complete official registry manifest structure
    const manifest = {
        server: item.server || item,
        _meta: item._meta || {
            "io.modelcontextprotocol.registry/official": {
                "status": "active"
            }
        }
    };

    // Populate JSON input
    const jsonInput = document.getElementById('mcp-server-json-input');
    if (jsonInput) {
        jsonInput.value = JSON.stringify(manifest, null, 2);
    }

    // Trigger parse & update UI
    onMCPJsonInputChange();

    // Also sync to Form fields
    if (currentParsedMCPConfig) {
        populateFormFromParsedMCP(currentParsedMCPConfig);
    }

    // Check if this server has variables that need filling (e.g. {tenant_id})
    const hasVariables = currentParsedMCPConfig && Object.keys(currentParsedMCPConfig.variables || {}).length > 0;
    
    // Switch to JSON tab so user can review parameters & fill template variables
    switchMCPModalMode('json');

    // If variables exist, scroll to variables container and focus first input
    if (hasVariables) {
        const varsContainer = document.getElementById('mcp-json-variables-container');
        if (varsContainer) {
            varsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            const firstVarKey = Object.keys(currentParsedMCPConfig.variables)[0];
            const firstInput = document.getElementById(`mcp-var-val-${firstVarKey}`);
            if (firstInput) setTimeout(() => firstInput.focus(), 150);
        }
    }
}

function openSearchMCPServerModal(query = '') {
    openAddMCPServerModal();
    switchMCPModalMode('search');
    const searchInput = document.getElementById('mcp-registry-search-input');
    if (searchInput) {
        if (query) searchInput.value = query;
        searchInput.focus();
    }
    performMCPRegistrySearch(query);
}

function onMCPTransportTypeChange() {
    const transport = document.getElementById('mcp-transport-type-select').value;
    const cmdGroup = document.getElementById('mcp-server-cmd-group');
    const argsGroup = document.getElementById('mcp-server-args-group');
    const urlGroup = document.getElementById('mcp-server-url-group');

    if (transport === 'streamable-http' || transport === 'sse') {
        if (cmdGroup) cmdGroup.style.display = 'none';
        if (argsGroup) argsGroup.style.display = 'none';
        if (urlGroup) urlGroup.style.display = 'block';
    } else {
        if (cmdGroup) cmdGroup.style.display = 'block';
        if (argsGroup) argsGroup.style.display = 'block';
        if (urlGroup) urlGroup.style.display = 'none';
    }
}

function pasteSampleTeamsMCPJson() {
    const jsonInput = document.getElementById('mcp-server-json-input');
    if (!jsonInput) return;
    jsonInput.value = SAMPLE_TEAMS_MCP_JSON;
    onMCPJsonInputChange();
}

function formatMCPJsonInput() {
    const jsonInput = document.getElementById('mcp-server-json-input');
    if (!jsonInput || !jsonInput.value.trim()) return;
    try {
        const obj = JSON.parse(jsonInput.value);
        jsonInput.value = JSON.stringify(obj, null, 2);
        onMCPJsonInputChange();
    } catch (e) {
        alert('Invalid JSON syntax: ' + e.message);
    }
}

function clearMCPJsonInput() {
    const jsonInput = document.getElementById('mcp-server-json-input');
    if (jsonInput) jsonInput.value = '';
    const statusEl = document.getElementById('mcp-json-status');
    if (statusEl) statusEl.style.display = 'none';
    const varsContainer = document.getElementById('mcp-json-variables-container');
    if (varsContainer) varsContainer.style.display = 'none';
    currentParsedMCPConfig = null;
}

function parseMCPConfig(rawText) {
    const raw = JSON.parse(rawText);
    let id = '';
    let name = '';
    let title = '';
    let description = '';
    let command = '';
    let args = [];
    let url = '';
    let transport_type = 'stdio';
    let env = {};
    let variables = {};
    let detectedType = 'Standard MCP Config';

    // Case 1: Official Registry Manifest Schema (server.schema.json)
    if (raw && typeof raw === 'object' && raw.server && typeof raw.server === 'object') {
        detectedType = 'Official Registry Manifest (server.schema.json)';
        const s = raw.server;
        name = s.name || '';
        title = s.title || '';
        description = s.description || title || '';
        id = name ? (name.includes('/') ? name.split('/').pop() : name) : 'mcp-server';

        if (Array.isArray(s.remotes) && s.remotes.length > 0) {
            const rem = s.remotes[0];
            transport_type = rem.type || 'streamable-http';
            url = rem.url || '';
            if (rem.variables && typeof rem.variables === 'object') {
                variables = Object.assign({}, rem.variables);
            }
            if (rem.headers && typeof rem.headers === 'object') {
                env = Object.assign({}, rem.headers);
            }
        } else if (Array.isArray(s.packages) && s.packages.length > 0) {
            const pkg = s.packages[0];
            const regType = (pkg.registryType || '').toLowerCase();
            const ident = pkg.identifier || '';
            if (regType === 'npm') {
                command = 'npx';
                args = ['-y', ident];
            } else if (regType === 'pypi') {
                command = 'uvx';
                args = [ident];
            } else {
                command = ident || 'npx';
            }
            if (Array.isArray(pkg.packageArguments)) {
                args = args.concat(pkg.packageArguments);
            }
            if (pkg.environmentVariables && typeof pkg.environmentVariables === 'object') {
                env = Object.assign({}, pkg.environmentVariables);
            }
            transport_type = 'stdio';
        }
        if (s.command) command = s.command;
        if (Array.isArray(s.args)) args = s.args;
        if (s.env && typeof s.env === 'object') env = Object.assign(env, s.env);
    }
    // Case 2: Claude Desktop / Cursor mcpServers format
    else if (raw && typeof raw === 'object' && raw.mcpServers && typeof raw.mcpServers === 'object') {
        detectedType = 'Claude / Cursor Config (mcpServers)';
        const keys = Object.keys(raw.mcpServers);
        if (keys.length > 0) {
            id = keys[0];
            const s = raw.mcpServers[id];
            command = s.command || '';
            args = Array.isArray(s.args) ? s.args : [];
            url = s.url || '';
            transport_type = s.transport || s.transport_type || (url ? 'streamable-http' : 'stdio');
            description = s.description || '';
            if (s.env && typeof s.env === 'object') env = s.env;
        }
    }
    // Case 3: Direct single server object
    else if (raw && typeof raw === 'object') {
        detectedType = 'Single Server Configuration';
        id = raw.id || raw.name || 'custom-server';
        command = raw.command || '';
        args = Array.isArray(raw.args) ? raw.args : [];
        url = raw.url || '';
        transport_type = raw.transport_type || raw.transport || (url ? 'streamable-http' : 'stdio');
        description = raw.description || raw.title || '';
        if (raw.env && typeof raw.env === 'object') env = raw.env;
        if (raw.variables && typeof raw.variables === 'object') variables = Object.assign({}, raw.variables);
    }

    // Also scan url for any template variables {var_name}
    if (url) {
        const matches = url.match(/\{([a-zA-Z0-9_-]+)\}/g);
        if (matches) {
            matches.forEach(m => {
                const varName = m.slice(1, -1);
                if (!variables[varName]) {
                    variables[varName] = { description: `Value for {${varName}} in remote URL`, isRequired: true };
                }
            });
        }
    }

    return {
        raw,
        id,
        name,
        title,
        description,
        command,
        args,
        url,
        transport_type,
        env,
        variables,
        detectedType
    };
}

function onMCPJsonInputChange() {
    const jsonInput = document.getElementById('mcp-server-json-input');
    const statusEl = document.getElementById('mcp-json-status');
    const statusTextEl = document.getElementById('mcp-json-status-text');
    const statusIconEl = document.getElementById('mcp-json-status-icon');
    const varsContainer = document.getElementById('mcp-json-variables-container');
    const varsFieldsEl = document.getElementById('mcp-json-variables-fields');

    if (!jsonInput) return;
    const text = jsonInput.value.trim();
    if (!text) {
        if (statusEl) statusEl.style.display = 'none';
        if (varsContainer) varsContainer.style.display = 'none';
        currentParsedMCPConfig = null;
        return;
    }

    try {
        const parsed = parseMCPConfig(text);
        currentParsedMCPConfig = parsed;

        if (statusEl) {
            statusEl.style.display = 'flex';
            statusEl.className = 'mcp-json-status valid';
            if (statusIconEl) statusIconEl.textContent = 'check_circle';
            const transportLabel = parsed.url ? `${parsed.transport_type}` : 'CLI stdio';
            statusTextEl.textContent = `Valid ${parsed.detectedType}: "${parsed.id || 'unnamed'}" [${transportLabel}]`;
        }

        // Render URL variables container if variables exist
        const varKeys = Object.keys(parsed.variables || {});
        if (varKeys.length > 0) {
            if (varsContainer) varsContainer.style.display = 'block';
            let varsHtml = '';
            for (const key of varKeys) {
                const meta = parsed.variables[key] || {};
                const desc = meta.description || key;
                const reqBadge = meta.isRequired ? '<span style="color:#f87171; margin-left:3px;">*</span>' : '';
                varsHtml += `
                    <div class="mcp-var-input-row" style="margin-bottom: 8px;">
                        <label style="display:block; font-size:11px; color:#c4b5fd; font-family:monospace; margin-bottom: 2px;">
                            {${escapeHtml(key)}}${reqBadge} <span style="color:var(--text-muted); font-family:var(--font-body); font-size:11px;">(${escapeHtml(desc)})</span>
                        </label>
                        <input type="text" id="mcp-var-val-${escapeHtml(key)}" class="agentic-text-field" placeholder="Enter ${escapeHtml(key)}..." oninput="updateMCPResolvedUrl()" style="font-family:monospace; font-size:12px; padding: 6px 8px;">
                    </div>
                `;
            }
            if (varsFieldsEl) varsFieldsEl.innerHTML = varsHtml;
            updateMCPResolvedUrl();
        } else {
            if (varsContainer) varsContainer.style.display = 'none';
        }
    } catch (e) {
        currentParsedMCPConfig = null;
        if (statusEl) {
            statusEl.style.display = 'flex';
            statusEl.className = 'mcp-json-status error';
            if (statusIconEl) statusIconEl.textContent = 'error';
            statusTextEl.textContent = `JSON Error: ${e.message}`;
        }
        if (varsContainer) varsContainer.style.display = 'none';
    }
}

function updateMCPResolvedUrl() {
    if (!currentParsedMCPConfig || !currentParsedMCPConfig.url) return;
    const previewBox = document.getElementById('mcp-json-resolved-preview-box');
    const previewText = document.getElementById('mcp-json-resolved-url-text');
    if (!previewBox || !previewText) return;

    let resolvedUrl = currentParsedMCPConfig.url;
    const varKeys = Object.keys(currentParsedMCPConfig.variables || {});
    for (const key of varKeys) {
        const input = document.getElementById(`mcp-var-val-${key}`);
        const val = input ? input.value.trim() : '';
        const replacement = val || `{${key}}`;
        resolvedUrl = resolvedUrl.split(`{${key}}`).join(replacement);
    }

    previewText.textContent = resolvedUrl;
    previewBox.style.display = 'block';
}

function populateFormFromParsedMCP(parsed) {
    if (!parsed) return;
    const idInput = document.getElementById('mcp-server-id-input');
    const transportSelect = document.getElementById('mcp-transport-type-select');
    const cmdInput = document.getElementById('mcp-server-cmd-input');
    const argsInput = document.getElementById('mcp-server-args-input');
    const urlInput = document.getElementById('mcp-server-url-input');
    const envInput = document.getElementById('mcp-server-env-input');
    const descInput = document.getElementById('mcp-server-desc-input');

    if (idInput) idInput.value = parsed.id || '';
    if (transportSelect) {
        transportSelect.value = parsed.transport_type || (parsed.url ? 'streamable-http' : 'stdio');
        onMCPTransportTypeChange();
    }
    if (cmdInput) cmdInput.value = parsed.command || '';
    if (argsInput) argsInput.value = Array.isArray(parsed.args) ? parsed.args.join(' ') : '';
    if (urlInput) urlInput.value = parsed.url || '';
    if (envInput) envInput.value = (parsed.env && Object.keys(parsed.env).length > 0) ? JSON.stringify(parsed.env, null, 2) : '';
    if (descInput) descInput.value = parsed.description || parsed.title || '';
}

let editingMCPServerId = null;

function editMCPServer(id) {
    const s = mcpServersList.find(x => x.id === id);
    if (!s) {
        console.warn('[MCP UI] Server not found for editing:', id);
        return;
    }

    editingMCPServerId = s.id;
    const modal = document.getElementById('mcp-server-modal');
    if (!modal) return;

    // Update modal header
    const titleEl = document.getElementById('mcp-modal-title');
    const subEl = modal.querySelector('.modal-sub');
    if (titleEl) titleEl.textContent = `Configure MCP Server: ${s.id}`;
    if (subEl) subEl.textContent = `Edit parameters, command arguments, remote URL, or environment variables`;

    // Populate Form Fields
    const idInput = document.getElementById('mcp-server-id-input');
    const transportSelect = document.getElementById('mcp-transport-type-select');
    const cmdInput = document.getElementById('mcp-server-cmd-input');
    const argsInput = document.getElementById('mcp-server-args-input');
    const urlInput = document.getElementById('mcp-server-url-input');
    const envInput = document.getElementById('mcp-server-env-input');
    const descInput = document.getElementById('mcp-server-desc-input');
    const enabledInput = document.getElementById('mcp-server-enabled-input');

    if (idInput) idInput.value = s.id || '';
    if (transportSelect) {
        transportSelect.value = s.transport_type || (s.url ? 'streamable-http' : 'stdio');
        onMCPTransportTypeChange();
    }
    if (cmdInput) cmdInput.value = s.command || '';
    if (argsInput) {
        if (Array.isArray(s.args)) {
            argsInput.value = s.args.join(' ');
        } else {
            argsInput.value = s.args || '';
        }
    }
    if (urlInput) urlInput.value = s.url || '';
    if (envInput) {
        if (s.env && typeof s.env === 'object' && Object.keys(s.env).length > 0) {
            envInput.value = JSON.stringify(s.env, null, 2);
        } else {
            envInput.value = '';
        }
    }
    if (descInput) descInput.value = s.description || '';
    if (enabledInput) enabledInput.checked = s.enabled ?? true;

    // Match preset if known, otherwise custom
    const presetSelect = document.getElementById('mcp-preset-select');
    if (presetSelect) {
        if (s.id === 'filesystem' || (s.command === 'npx' && String(s.args).includes('server-filesystem'))) {
            presetSelect.value = 'filesystem';
        } else if (s.id === 'sqlite' || String(s.args).includes('mcp-server-sqlite')) {
            presetSelect.value = 'sqlite';
        } else if (s.id === 'fetch' || String(s.args).includes('mcp-server-fetch')) {
            presetSelect.value = 'fetch';
        } else if (s.id === 'git' || String(s.args).includes('mcp-server-git')) {
            presetSelect.value = 'git';
        } else if (s.id === 'memory' || String(s.args).includes('server-memory')) {
            presetSelect.value = 'memory';
        } else {
            presetSelect.value = 'custom';
        }
    }

    // Populate Raw JSON editor
    let sampleObj;
    if (s.transport_type === 'streamable-http' || s.transport_type === 'sse' || s.url) {
        sampleObj = {
            server: {
                name: s.id,
                title: s.description || s.id,
                description: s.description || '',
                remotes: [
                    {
                        type: s.transport_type || 'streamable-http',
                        url: s.url || ''
                    }
                ]
            }
        };
        if (s.env && typeof s.env === 'object' && Object.keys(s.env).length > 0) {
            sampleObj.server.remotes[0].headers = s.env;
        }
    } else {
        let cleanArgs = [];
        if (Array.isArray(s.args)) {
            cleanArgs = s.args;
        } else if (s.args) {
            cleanArgs = String(s.args).split(/\s+/).filter(Boolean);
        }
        sampleObj = {
            mcpServers: {
                [s.id]: {
                    command: s.command || 'npx',
                    args: cleanArgs,
                    env: s.env || {},
                    description: s.description || ''
                }
            }
        };
    }

    const jsonInput = document.getElementById('mcp-server-json-input');
    if (jsonInput) {
        jsonInput.value = JSON.stringify(sampleObj, null, 2);
    }
    onMCPJsonInputChange();

    // Update Save button text
    const saveBtn = document.getElementById('mcp-save-btn');
    if (saveBtn) {
        saveBtn.innerHTML = `
            <span class="material-symbols-outlined" style="font-size: 16px;">save</span>
            <span>Save & Reconnect</span>
        `;
    }

    // Start in Form mode
    switchMCPModalMode('form');
    modal.style.display = 'flex';
}

function openAddMCPServerModal() {
    editingMCPServerId = null;
    const modal = document.getElementById('mcp-server-modal');
    if (!modal) return;

    // Reset title and subtitle
    const titleEl = document.getElementById('mcp-modal-title');
    const subEl = modal.querySelector('.modal-sub');
    if (titleEl) titleEl.textContent = 'Configure MCP Server';
    if (subEl) subEl.textContent = 'Add a Model Context Protocol tool server (Local CLI or Remote Streamable-HTTP)';

    // Reset save button text
    const saveBtn = document.getElementById('mcp-save-btn');
    if (saveBtn) {
        saveBtn.innerHTML = `
            <span class="material-symbols-outlined" style="font-size: 16px;">save</span>
            <span>Save & Connect</span>
        `;
    }

    document.getElementById('mcp-preset-select').value = 'custom';
    document.getElementById('mcp-transport-type-select').value = 'stdio';
    onMCPTransportTypeChange();
    document.getElementById('mcp-server-id-input').value = '';
    document.getElementById('mcp-server-cmd-input').value = '';
    document.getElementById('mcp-server-args-input').value = '';
    document.getElementById('mcp-server-url-input').value = '';
    document.getElementById('mcp-server-env-input').value = '';
    document.getElementById('mcp-server-desc-input').value = '';
    document.getElementById('mcp-server-enabled-input').checked = true;
    clearMCPJsonInput();
    switchMCPModalMode('form');
    modal.style.display = 'flex';
}

function closeAddMCPServerModal() {
    editingMCPServerId = null;
    const modal = document.getElementById('mcp-server-modal');
    if (modal) modal.style.display = 'none';
}

function onMCPPresetSelect() {
    const preset = document.getElementById('mcp-preset-select').value;
    if (preset === 'search-registry') {
        switchMCPModalMode('search');
        return;
    }
    if (preset === 'json-custom') {
        switchMCPModalMode('json');
        const jsonInput = document.getElementById('mcp-server-json-input');
        if (jsonInput && !jsonInput.value.trim()) {
            pasteSampleTeamsMCPJson();
        }
        return;
    }

    switchMCPModalMode('form');
    const idInput = document.getElementById('mcp-server-id-input');
    const transportSelect = document.getElementById('mcp-transport-type-select');
    const cmdInput = document.getElementById('mcp-server-cmd-input');
    const argsInput = document.getElementById('mcp-server-args-input');
    const urlInput = document.getElementById('mcp-server-url-input');
    const envInput = document.getElementById('mcp-server-env-input');
    const descInput = document.getElementById('mcp-server-desc-input');

    if (transportSelect) {
        transportSelect.value = 'stdio';
        onMCPTransportTypeChange();
    }
    if (urlInput) urlInput.value = '';

    if (preset === 'filesystem') {
        idInput.value = 'filesystem';
        cmdInput.value = 'npx';
        argsInput.value = '-y @modelcontextprotocol/server-filesystem .';
        descInput.value = 'Workspace filesystem operations via official MCP server';
    } else if (preset === 'sqlite') {
        idInput.value = 'sqlite';
        cmdInput.value = 'uvx';
        argsInput.value = 'mcp-server-sqlite --db-path ./database.db';
        descInput.value = 'SQLite database querying and schema inspection';
    } else if (preset === 'fetch') {
        idInput.value = 'fetch';
        cmdInput.value = 'uvx';
        argsInput.value = 'mcp-server-fetch';
        descInput.value = 'Web page fetching and Markdown conversion';
    } else if (preset === 'git') {
        idInput.value = 'git';
        cmdInput.value = 'uvx';
        argsInput.value = 'mcp-server-git --repository .';
        descInput.value = 'Git repository version control and commits';
    } else if (preset === 'memory') {
        idInput.value = 'memory';
        cmdInput.value = 'npx';
        argsInput.value = '-y @modelcontextprotocol/server-memory';
        descInput.value = 'Knowledge graph memory server';
    }
}

async function saveMCPServerForm() {
    let payload = null;

    if (currentMCPModalMode === 'json') {
        const jsonInput = document.getElementById('mcp-server-json-input');
        const rawText = jsonInput ? jsonInput.value.trim() : '';
        if (!rawText) {
            alert('Please paste or enter an MCP configuration JSON.');
            return;
        }

        let parsed;
        try {
            parsed = parseMCPConfig(rawText);
        } catch (e) {
            alert('Invalid JSON syntax: ' + e.message);
            return;
        }

        let finalUrl = parsed.url;
        // Resolve variables
        const varKeys = Object.keys(parsed.variables || {});
        for (const key of varKeys) {
            const meta = parsed.variables[key] || {};
            const input = document.getElementById(`mcp-var-val-${key}`);
            const val = input ? input.value.trim() : '';
            if (meta.isRequired && !val) {
                alert(`Please provide a value for required variable: {${key}} (${meta.description || key})`);
                if (input) input.focus();
                return;
            }
            if (val) {
                finalUrl = finalUrl.split(`{${key}}`).join(val);
            }
        }

        const id = parsed.id || 'mcp-server';
        payload = {
            id,
            command: parsed.command || '',
            args: parsed.args || [],
            url: finalUrl || '',
            transport_type: parsed.transport_type || (finalUrl ? 'streamable-http' : 'stdio'),
            env: parsed.env || {},
            description: parsed.description || parsed.title || '',
            enabled: document.getElementById('mcp-server-enabled-input')?.checked ?? true
        };
    } else {
        const id = document.getElementById('mcp-server-id-input').value.trim();
        const transport = document.getElementById('mcp-transport-type-select').value;
        const cmd = document.getElementById('mcp-server-cmd-input').value.trim();
        const rawArgs = document.getElementById('mcp-server-args-input').value.trim();
        const url = document.getElementById('mcp-server-url-input').value.trim();
        const rawEnv = document.getElementById('mcp-server-env-input').value.trim();
        const desc = document.getElementById('mcp-server-desc-input').value.trim();
        const enabled = document.getElementById('mcp-server-enabled-input')?.checked ?? true;

        if (!id) {
            alert('Server ID / Name is required.');
            return;
        }

        if (transport === 'streamable-http' || transport === 'sse') {
            if (!url) {
                alert('Remote Server URL is required for remote transport.');
                return;
            }
        } else {
            if (!cmd) {
                alert('Command is required for local stdio transport.');
                return;
            }
        }

        let args = [];
        if (rawArgs.startsWith('[') && rawArgs.endsWith(']')) {
            try { args = JSON.parse(rawArgs); } catch (e) { args = rawArgs.split(/\s+/).filter(Boolean); }
        } else if (rawArgs) {
            args = rawArgs.split(/\s+/).filter(Boolean);
        }

        let env = {};
        if (rawEnv) {
            try {
                env = JSON.parse(rawEnv);
            } catch (e) {
                console.warn('[MCP UI] Invalid JSON for env, ignoring');
            }
        }

        payload = {
            id,
            command: cmd,
            args,
            url: (transport === 'streamable-http' || transport === 'sse') ? url : '',
            transport_type: transport,
            env,
            description: desc,
            enabled
        };
    }

    if (editingMCPServerId) {
        payload.old_id = editingMCPServerId;
    }

    try {
        const res = await fetch(`${getApiBase()}/v1/mcp/servers`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        closeAddMCPServerModal();
        if (data && data.servers) {
            mcpServersList = data.servers;
            renderMCPServers(mcpServersList);
        } else {
            await fetchMCPServers();
        }
    } catch (e) {
        alert(`Failed to save MCP server: ${e.message}`);
    }
}

function initMCPUI() {
    fetchMCPServers();
}

// ════════════════════════════════════════════════════════════════════════════════
//  3D Detection & Content Classifier
// ════════════════════════════════════════════════════════════════════════════════

function is3DContent(code, lang) {
    if (!code || typeof code !== 'string') return false;

    const trimmed = code.trim();
    if (trimmed.length < 20) return false;

    // Check for an explicit 3D scene/model creation function (createScene, createModel, initModel)
    const funcPattern = /(?:(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s+(?:createModel|createScene|initModel)\s*\(|(?:const|let|var)\s+(?:createModel|createScene|initModel)\s*=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*=>|[a-zA-Z0-9_$]+\s*=>))/i;
    if (funcPattern.test(trimmed)) {
        // Ensure the function body actually references 3D constructs (THREE, scene, helpers, geometry, mesh)
        if (/\b(?:THREE|scene|helpers|geometry|material|mesh)\b/i.test(trimmed)) {
            return true;
        }
    }

    // Direct scene construction fallback: explicit scene.add(...) with new THREE.* objects
    if (/\bscene\.add\s*\(/.test(trimmed) && (/\bnew\s+THREE\./.test(trimmed) || /\bTHREE\.(?:Mesh|Group|Points|Line|LineSegments)\b/.test(trimmed))) {
        return true;
    }

    return false;
}

// ════════════════════════════════════════════════════════════════════════════════
//  Vision Capabilities & Image Attachment Handling
// ════════════════════════════════════════════════════════════════════════════════

let activeImageAttachment = null;
let currentModelHasVision = false;

function updateVisionUploadVisibility() {
    const uploadBtn = document.getElementById('vision-upload-btn');
    if (!uploadBtn) return;
    if (currentModelHasVision) {
        uploadBtn.classList.remove('hidden');
    } else {
        uploadBtn.classList.add('hidden');
        removeImageAttachment();
    }
}

function triggerVisionUpload() {
    const fileInput = document.getElementById('vision-file-input');
    if (fileInput) fileInput.click();
}

function handleVisionFileSelect(event) {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    processImageFile(file);
    event.target.value = '';
}

function processImageFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file (PNG, JPG, WebP, BMP).');
        return;
    }
    const reader = new FileReader();
    reader.onload = function(e) {
        const dataUrl = e.target.result;
        const img = new Image();
        img.onload = function() {
            activeImageAttachment = {
                filename: file.name,
                filesize: `${img.width}x${img.height}`,
                dataUrl: dataUrl
            };
            window.__lastUploadedImage = dataUrl;
            window.__lastUploadedImageElement = img;
            renderImageAttachmentChip();
            if (typeof ThreeStudio !== 'undefined' && ThreeStudio.updatePhotoTextureButtons) {
                ThreeStudio.updatePhotoTextureButtons();
            }
        };
        img.src = dataUrl;
    };
    reader.readAsDataURL(file);
}

function renderImageAttachmentChip() {
    const chip = document.getElementById('image-attachment-preview');
    const thumb = document.getElementById('attachment-thumb-img');
    const nameEl = document.getElementById('attachment-filename');
    const sizeEl = document.getElementById('attachment-filesize');
    if (!chip || !thumb) return;

    if (activeImageAttachment) {
        thumb.src = activeImageAttachment.dataUrl;
        if (nameEl) nameEl.textContent = activeImageAttachment.filename;
        if (sizeEl) sizeEl.textContent = activeImageAttachment.filesize;
        chip.classList.remove('hidden');
    } else {
        chip.classList.add('hidden');
    }
}

function removeImageAttachment() {
    activeImageAttachment = null;
    const chip = document.getElementById('image-attachment-preview');
    if (chip) chip.classList.add('hidden');
}

function getLastUploadedPhoto() {
    if (window.__lastUploadedImage) return window.__lastUploadedImage;
    if (typeof chatHistory !== 'undefined' && Array.isArray(chatHistory)) {
        for (let i = chatHistory.length - 1; i >= 0; i--) {
            const m = chatHistory[i];
            if (m.role === 'user' && Array.isArray(m.content)) {
                for (const part of m.content) {
                    if (part.type === 'image_url' && part.image_url && part.image_url.url) {
                        window.__lastUploadedImage = part.image_url.url;
                        return part.image_url.url;
                    }
                }
            }
        }
    }
    const imgs = document.querySelectorAll('.msg-user img');
    if (imgs.length > 0) {
        const lastImg = imgs[imgs.length - 1];
        if (lastImg.src && lastImg.src.startsWith('data:image/')) {
            window.__lastUploadedImage = lastImg.src;
            return lastImg.src;
        }
    }
    return '';
}

function quickPromptModel3D() {
    window.lastUserPromptWas3D = true;
    if (!isPreviewOpen) openPreviewPanel();
    switchPreviewTab('tab-3d');
    if (chatInput) {
        chatInput.value = 'Reconstruct an accurate, watertight 3D model matching the exact shape, silhouette, and contours of the object in this image. Apply the photo texture using `new THREE.TextureLoader().load(inputImage)` with RepeatWrapping and DoubleSide on the primary mesh.\n\nGEOMETRIC QUALITY RULES (AVOID HOLES):\n1. Watertight Solid: Zero holes. Lathe profiles must start and end at x=0 (or use helpers.createWatertightLathe). Open vessels (cups, bowls, vases) must have solid wall thickness (use helpers.createHollowVessel).\n2. Capped Ends: Tubes, pipes, handles, and spouts must have closed ends (use helpers.createCappedTube).\n3. Deep Embedding: Attachments (handles, spouts, limbs) must penetrate 5-10% deep into parent meshes so there are no floating gaps or seam cracks.\n4. Smooth Shading: Use 32-64 radial segments for curves and call geometry.computeVertexNormals().\n\nReturn Three.js function createModel(scene, THREE, inputImage, helpers) in a ```javascript block.';
        chatInput.focus();
        sendBtn.disabled = false;
    }
}

function setupVisionDragAndDrop() {
    window.addEventListener('dragover', (e) => {
        const types = e.dataTransfer?.types;
        if (types && (types.includes('Files') || types.includes('public.file-url'))) {
            e.preventDefault();
        }
    });
    window.addEventListener('drop', (e) => {
        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];
            if (file.type.startsWith('image/')) {
                e.preventDefault();
                currentModelHasVision = true;
                updateVisionUploadVisibility();
                processImageFile(file);
            }
        }
    });

    const handlePasteEvent = (e) => {
        const items = (e.clipboardData || e.originalEvent?.clipboardData)?.items;
        if (!items) return;
        for (let i = 0; i < items.length; i++) {
            if (items[i].type && items[i].type.indexOf('image') !== -1) {
                const file = items[i].getAsFile();
                if (file) {
                    currentModelHasVision = true;
                    updateVisionUploadVisibility();
                    processImageFile(file);
                    e.preventDefault();
                    if (chatInput) chatInput.focus();
                    break;
                }
            }
        }
    };

    if (chatInput) {
        chatInput.addEventListener('paste', handlePasteEvent);
    }
    window.addEventListener('paste', (e) => {
        // Fallback: If not already focused in another text input, catch paste on window
        const activeTag = document.activeElement ? document.activeElement.tagName.toLowerCase() : '';
        if (activeTag !== 'textarea' && activeTag !== 'input') {
            handlePasteEvent(e);
        }
    });
}

// ════════════════════════════════════════════════════════════════════════════════
//  Interactive 3D Studio (Three.js Visualizer & Editor)
// ════════════════════════════════════════════════════════════════════════════════

const ThreeStudio = {
    isInitialized: false,
    scene: null,
    camera: null,
    renderer: null,
    controls: null,
    transformControls: null,
    modelGroup: null,
    selectedObject: null,
    vertexMode: false,
    vertexHandlesGroup: null,
    wireframeMode: false,
    gridHelper: null,
    raycaster: null,
    mouse: null,
    canvasEl: null,
    currentCode: '',

    helpers: {
        /**
         * Ensures a LatheGeometry profile starts strictly at x=0 and ends at x=0,
         * completely sealing top and bottom holes.
         */
        createWatertightLathe(points, segments = 32, phiStart = 0, phiLength = Math.PI * 2) {
            if (!Array.isArray(points) || points.length < 2) {
                return new THREE.BufferGeometry();
            }
            const pts = points.map(p => (p instanceof THREE.Vector2 ? p.clone() : new THREE.Vector2(p.x, p.y)));

            // Ensure bottom point starts at x=0
            if (pts[0].x > 0.001) {
                pts.unshift(new THREE.Vector2(0, pts[0].y));
            }
            // Ensure top point ends at x=0
            if (pts[pts.length - 1].x > 0.001) {
                pts.push(new THREE.Vector2(0, pts[pts.length - 1].y));
            }

            const LatheCtor = THREE.LatheBufferGeometry || THREE.LatheGeometry;
            let geom = new LatheCtor(pts, Math.max(segments, 16), phiStart, phiLength);
            if (geom.vertices && !geom.attributes && THREE.BufferGeometry) {
                geom = new THREE.BufferGeometry().fromGeometry(geom);
            }
            geom.computeVertexNormals();
            return geom;
        },

        /**
         * Creates a vessel (cup, mug, bowl, vase, bottle) with TRUE solid wall thickness
         * (outer shell + rounded rim + inner shell + solid bottom), eliminating paper-thin hollow geometry!
         */
        createHollowVessel(points, wallThickness = 0.08, segments = 32) {
            if (!Array.isArray(points) || points.length < 2) {
                return new THREE.BufferGeometry();
            }
            const outer = points.map(p => (p instanceof THREE.Vector2 ? p.clone() : new THREE.Vector2(p.x, p.y)));

            // Ensure outer starts at x=0
            if (outer[0].x > 0.001) {
                outer.unshift(new THREE.Vector2(0, outer[0].y));
            }

            const t = Math.max(0.01, wallThickness);
            const closedProfile = [];

            // 1. Add outer profile points from bottom to top rim
            for (let i = 0; i < outer.length; i++) {
                closedProfile.push(outer[i]);
            }

            // 2. Add rim connector (rounded or flat inward offset)
            const rimTop = outer[outer.length - 1];
            const innerRimX = Math.max(0.02, rimTop.x - t);
            closedProfile.push(new THREE.Vector2(innerRimX, rimTop.y));

            // 3. Generate inner profile points going downward
            for (let i = outer.length - 2; i >= 1; i--) {
                const p = outer[i];
                const innerX = Math.max(0.01, p.x - t);
                const innerY = p.y;
                closedProfile.push(new THREE.Vector2(innerX, innerY));
            }

            // 4. Inner bottom closure at x=0
            const innerBottomY = outer[0].y + t;
            closedProfile.push(new THREE.Vector2(0, innerBottomY));

            const geom = new THREE.LatheGeometry(closedProfile, Math.max(segments, 24));
            geom.computeVertexNormals();
            return geom;
        },

        /**
         * Creates a tube along a 3D curve with sealed hemispherical/disc end caps.
         * Solves the open-ended tube problem for handles, spouts, wires, and pipes.
         */
        createCappedTube(curveOrRadius, tubularSegmentsOrHeight = 64, radiusOrMat = 0.1, radialSegments = 16, closed = false, targetScene = null) {
            // Handle numeric cylinder / capped tube invocation: helpers.createCappedTube(radius, height, material, scene)
            if (typeof curveOrRadius === 'number') {
                const radius = Math.max(0.01, curveOrRadius);
                const height = typeof tubularSegmentsOrHeight === 'number' ? Math.max(0.01, tubularSegmentsOrHeight) : 1.0;
                const mat = (radiusOrMat && (radiusOrMat.isMaterial || (typeof radiusOrMat === 'object' && !radiusOrMat.isBufferGeometry && !radiusOrMat.isGeometry))) ? radiusOrMat : null;
                const radSegs = typeof radialSegments === 'number' ? radialSegments : 24;
                const cylGeom = new THREE.CylinderGeometry(radius, radius, height, radSegs, 1, false);
                cylGeom.computeVertexNormals();
                if (mat) {
                    const mesh = new THREE.Mesh(cylGeom, mat);
                    const sceneToAdd = (targetScene && typeof targetScene.add === 'function') ? targetScene : ((closed && typeof closed.add === 'function') ? closed : null);
                    if (sceneToAdd) sceneToAdd.add(mesh);
                    return mesh;
                }
                return cylGeom;
            }

            const curve = curveOrRadius;
            const tubularSegments = typeof tubularSegmentsOrHeight === 'number' ? tubularSegmentsOrHeight : 64;
            const radius = typeof radiusOrMat === 'number' ? radiusOrMat : 0.1;
            const tubeGeom = new THREE.TubeGeometry(curve, tubularSegments, radius, radialSegments, closed);
            if (closed) {
                tubeGeom.computeVertexNormals();
                return tubeGeom;
            }

            // Generate start and end spherical caps
            try {
                const p0 = curve.getPointAt(0);
                const p1 = curve.getPointAt(1);

                const cap0 = new THREE.SphereGeometry(radius, radialSegments, Math.max(4, Math.floor(radialSegments / 2)));
                cap0.translate(p0.x, p0.y, p0.z);

                const cap1 = new THREE.SphereGeometry(radius, radialSegments, Math.max(4, Math.floor(radialSegments / 2)));
                cap1.translate(p1.x, p1.y, p1.z);

                if (THREE.BufferGeometryUtils && typeof THREE.BufferGeometryUtils.mergeBufferGeometries === 'function') {
                    const merged = THREE.BufferGeometryUtils.mergeBufferGeometries([tubeGeom, cap0, cap1]);
                    if (merged) {
                        merged.computeVertexNormals();
                        return merged;
                    }
                }
            } catch(e) {
                console.warn('[3D Studio helpers.createCappedTube] merge fallback:', e);
            }
            tubeGeom.computeVertexNormals();
            return tubeGeom;
        },

        /**
         * Creates a smooth beveled / rounded box without razor-sharp polygon edges.
         */
        createRoundedBox(width = 1, height = 1, depth = 1, radius = 0.08, smoothness = 4) {
            const r = Math.min(radius, width / 2 - 0.001, height / 2 - 0.001, depth / 2 - 0.001);
            if (r <= 0) {
                return new THREE.BoxGeometry(width, height, depth);
            }
            const shape = new THREE.Shape();
            const w = width - 2 * r;
            const h = height - 2 * r;
            shape.moveTo(-w / 2, -h / 2 - r);
            shape.lineTo(w / 2, -h / 2 - r);
            shape.absarc(w / 2, -h / 2, r, -Math.PI / 2, 0, false);
            shape.lineTo(w / 2 + r, h / 2);
            shape.absarc(w / 2, h / 2, r, 0, Math.PI / 2, false);
            shape.lineTo(-w / 2, h / 2 + r);
            shape.absarc(-w / 2, h / 2, r, Math.PI / 2, Math.PI, false);
            shape.lineTo(-w / 2 - r, -h / 2);
            shape.absarc(-w / 2, -h / 2, r, Math.PI, Math.PI * 1.5, false);
            shape.closePath();

            const extrudeSettings = {
                depth: Math.max(0.01, depth - 2 * r),
                bevelEnabled: true,
                bevelSegments: Math.max(2, smoothness),
                steps: 1,
                bevelSize: r,
                bevelThickness: r
            };
            const geom = new THREE.ExtrudeGeometry(shape, extrudeSettings);
            geom.center();
            geom.computeVertexNormals();
            return geom;
        },

        /**
         * Creates a watertight capsule (cylinder with hemispherical caps).
         */
        createCapsule(radius = 0.2, length = 0.8, capSegments = 8, radialSegments = 24) {
            const cylH = Math.max(0.01, length);
            const cyl = new THREE.CylinderGeometry(radius, radius, cylH, radialSegments, 1, true);
            const topCap = new THREE.SphereGeometry(radius, radialSegments, capSegments, 0, Math.PI * 2, 0, Math.PI / 2);
            topCap.translate(0, cylH / 2, 0);
            const botCap = new THREE.SphereGeometry(radius, radialSegments, capSegments, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2);
            botCap.translate(0, -cylH / 2, 0);

            if (THREE.BufferGeometryUtils && typeof THREE.BufferGeometryUtils.mergeBufferGeometries === 'function') {
                const merged = THREE.BufferGeometryUtils.mergeBufferGeometries([cyl, topCap, botCap]);
                if (merged) {
                    const welded = THREE.BufferGeometryUtils.mergeVertices(merged, 1e-4);
                    welded.computeVertexNormals();
                    return welded;
                }
            }
            const fallback = new THREE.CylinderGeometry(radius, radius, cylH, radialSegments, 1, false);
            fallback.computeVertexNormals();
            return fallback;
        },

        /**
         * Welds duplicate vertices to eliminate seam cracks and recalculates smooth normals.
         */
        weldAndSmooth(geometry, tolerance = 1e-4) {
            if (!geometry) return geometry;
            let res = geometry;
            if (THREE.BufferGeometryUtils && typeof THREE.BufferGeometryUtils.mergeVertices === 'function') {
                try {
                    res = THREE.BufferGeometryUtils.mergeVertices(geometry, tolerance);
                } catch(e) {}
            }
            try {
                res.computeVertexNormals();
            } catch(e) {}
            return res;
        },

        /**
         * Helper to create a production-quality PBR material with DoubleSide enabled.
         */
        createPBRMaterial(options = {}) {
            return new THREE.MeshStandardMaterial({
                color: options.color !== undefined ? options.color : 0x3b82f6,
                roughness: options.roughness !== undefined ? options.roughness : 0.35,
                metalness: options.metalness !== undefined ? options.metalness : 0.1,
                side: THREE.DoubleSide,
                shadowSide: THREE.DoubleSide,
                ...options
            });
        },

        /**
         * Generates cylindrical UV coordinates around the Y-axis.
         */
        applyCylindricalUV(geometry) {
            if (!geometry) return geometry;
            try {
                if (geometry.vertices && !geometry.attributes && typeof THREE.BufferGeometry === 'function') {
                    geometry = new THREE.BufferGeometry().fromGeometry(geometry);
                }
                geometry.computeBoundingBox();
                const bbox = geometry.boundingBox || new THREE.Box3();
                const pos = geometry.attributes ? geometry.attributes.position : null;
                if (!pos) return geometry;
                const uvs = [];
                const minY = bbox.min.y;
                const rangeY = (bbox.max.y - bbox.min.y) || 1.0;
                for (let i = 0; i < pos.count; i++) {
                    const x = pos.getX(i);
                    const y = pos.getY(i);
                    const z = pos.getZ(i);
                    let u = (Math.atan2(x, z) / (2 * Math.PI)) + 0.5;
                    let v = (y - minY) / rangeY;
                    uvs.push(u, v);
                }
                const attr = new THREE.Float32BufferAttribute(uvs, 2);
                if (typeof geometry.setAttribute === 'function') {
                    geometry.setAttribute('uv', attr);
                } else if (typeof geometry.addAttribute === 'function') {
                    geometry.addAttribute('uv', attr);
                } else if (geometry.attributes) {
                    geometry.attributes.uv = attr;
                }
                geometry.uvsNeedUpdate = true;
            } catch(e) {
                console.warn('[3D Studio applyCylindricalUV]', e);
            }
            return geometry;
        },

        /**
         * Generates 2D planar projection UV coordinates along the specified axis ('x', 'y', or 'z').
         */
        applyPlanarUV(geometry, axis = 'z') {
            if (!geometry) return geometry;
            try {
                if (geometry.vertices && !geometry.attributes && typeof THREE.BufferGeometry === 'function') {
                    geometry = new THREE.BufferGeometry().fromGeometry(geometry);
                }
                geometry.computeBoundingBox();
                const bbox = geometry.boundingBox || new THREE.Box3();
                const pos = geometry.attributes ? geometry.attributes.position : null;
                if (!pos) return geometry;
                const uvs = [];
                const minX = bbox.min.x, rangeX = (bbox.max.x - bbox.min.x) || 1.0;
                const minY = bbox.min.y, rangeY = (bbox.max.y - bbox.min.y) || 1.0;
                const minZ = bbox.min.z, rangeZ = (bbox.max.z - bbox.min.z) || 1.0;
                const ax = (axis || 'z').toLowerCase();
                for (let i = 0; i < pos.count; i++) {
                    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
                    let u, v;
                    if (ax === 'x') {
                        u = (z - minZ) / rangeZ;
                        v = (y - minY) / rangeY;
                    } else if (ax === 'y') {
                        u = (x - minX) / rangeX;
                        v = (z - minZ) / rangeZ;
                    } else {
                        u = (x - minX) / rangeX;
                        v = (y - minY) / rangeY;
                    }
                    uvs.push(Math.max(0, Math.min(1, u)), Math.max(0, Math.min(1, v)));
                }
                const attr = new THREE.Float32BufferAttribute(uvs, 2);
                if (typeof geometry.setAttribute === 'function') {
                    geometry.setAttribute('uv', attr);
                } else if (typeof geometry.addAttribute === 'function') {
                    geometry.addAttribute('uv', attr);
                } else if (geometry.attributes) {
                    geometry.attributes.uv = attr;
                }
                geometry.uvsNeedUpdate = true;
            } catch(e) {
                console.warn('[3D Studio applyPlanarUV]', e);
            }
            return geometry;
        },

        /**
         * Generates triplanar box projection UV coordinates.
         */
        applyBoxUV(geometry) {
            if (!geometry) return geometry;
            try {
                geometry.computeBoundingBox();
                geometry.computeVertexNormals();
                const bbox = geometry.boundingBox || new THREE.Box3();
                const pos = geometry.attributes.position;
                const norm = geometry.attributes.normal;
                if (!pos) return geometry;
                const uvs = [];
                const minX = bbox.min.x, rangeX = (bbox.max.x - bbox.min.x) || 1.0;
                const minY = bbox.min.y, rangeY = (bbox.max.y - bbox.min.y) || 1.0;
                const minZ = bbox.min.z, rangeZ = (bbox.max.z - bbox.min.z) || 1.0;
                for (let i = 0; i < pos.count; i++) {
                    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
                    let nx = 0, ny = 0, nz = 1;
                    if (norm) {
                        nx = Math.abs(norm.getX(i));
                        ny = Math.abs(norm.getY(i));
                        nz = Math.abs(norm.getZ(i));
                    }
                    let u, v;
                    if (nx >= ny && nx >= nz) {
                        u = (z - minZ) / rangeZ;
                        v = (y - minY) / rangeY;
                    } else if (ny >= nx && ny >= nz) {
                        u = (x - minX) / rangeX;
                        v = (z - minZ) / rangeZ;
                    } else {
                        u = (x - minX) / rangeX;
                        v = (y - minY) / rangeY;
                    }
                    uvs.push(u, v);
                }
                geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
                geometry.uvsNeedUpdate = true;
            } catch(e) {
                console.warn('[3D Studio applyBoxUV]', e);
            }
            return geometry;
        },

        /**
         * Samples pixel RGB color from the input photo or image.
         */
        sampleColor(inputImage, u = 0.5, v = 0.5) {
            try {
                const src = inputImage || (typeof getLastUploadedPhoto === 'function' ? getLastUploadedPhoto() : window.__lastUploadedImage);
                if (!src || typeof document === 'undefined') return new THREE.Color(0xC41E3E);

                if (!ThreeStudio._sampleCanvas) {
                    ThreeStudio._sampleCanvas = document.createElement('canvas');
                    ThreeStudio._sampleCtx = ThreeStudio._sampleCanvas.getContext('2d', { willReadFrequently: true });
                }
                if (ThreeStudio._cachedSampleImg && ThreeStudio._cachedSampleImg.src === src && ThreeStudio._cachedSampleImg.complete) {
                    const w = ThreeStudio._sampleCanvas.width;
                    const h = ThreeStudio._sampleCanvas.height;
                    const px = Math.min(w - 1, Math.max(0, Math.floor(u * w)));
                    const py = Math.min(h - 1, Math.max(0, Math.floor(v * h)));
                    const pixel = ThreeStudio._sampleCtx.getImageData(px, py, 1, 1).data;
                    return new THREE.Color(pixel[0] / 255, pixel[1] / 255, pixel[2] / 255);
                } else if (typeof src === 'string' && src.startsWith('data:image/')) {
                    const img = new Image();
                    img.crossOrigin = 'anonymous';
                    img.onload = () => {
                        ThreeStudio._sampleCanvas.width = img.naturalWidth || 256;
                        ThreeStudio._sampleCanvas.height = img.naturalHeight || 256;
                        ThreeStudio._sampleCtx.drawImage(img, 0, 0);
                        ThreeStudio._cachedSampleImg = img;
                    };
                    img.src = src;
                }
            } catch(e) {
                console.warn('[3D Studio sampleColor]', e);
            }
            return new THREE.Color(0xC41E3E);
        },

        /**
         * Crops and returns a THREE.CanvasTexture from the input photo or image.
         */
        cropTexture(inputImage, uMin = 0, vMin = 0, uMax = 1, vMax = 1, options = {}) {
            const canvas = document.createElement('canvas');
            canvas.width = 512;
            canvas.height = 512;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#C41E3E';
            ctx.fillRect(0, 0, 512, 512);

            const texture = new THREE.CanvasTexture(canvas);
            texture.wrapS = options.wrapS !== undefined ? options.wrapS : THREE.ClampToEdgeWrapping;
            texture.wrapT = options.wrapT !== undefined ? options.wrapT : THREE.ClampToEdgeWrapping;

            const src = inputImage || (typeof getLastUploadedPhoto === 'function' ? getLastUploadedPhoto() : window.__lastUploadedImage);
            if (src && typeof src === 'string' && src.startsWith('data:image/')) {
                const img = new Image();
                img.crossOrigin = 'anonymous';
                img.onload = () => {
                    const iw = img.naturalWidth || 512;
                    const ih = img.naturalHeight || 512;
                    const sx = Math.max(0, uMin * iw);
                    const sy = Math.max(0, vMin * ih);
                    const sw = Math.max(1, (uMax - uMin) * iw);
                    const sh = Math.max(1, (vMax - vMin) * ih);
                    canvas.width = Math.min(1024, Math.max(64, sw));
                    canvas.height = Math.min(1024, Math.max(64, sh));
                    ctx.drawImage(img, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
                    texture.needsUpdate = true;
                };
                img.src = src;
            }
            return texture;
        },

        /**
         * Creates and returns a named THREE.Mesh.
         */
        createPart(name, geometry, material, targetScene) {
            const mesh = new THREE.Mesh(geometry, material || this.createPBRMaterial());
            if (name) mesh.name = name;
            if (targetScene && typeof targetScene.add === 'function') {
                targetScene.add(mesh);
            }
            return mesh;
        }
    },

    getSafeHelpers() {
        const target = this.helpers;
        return new Proxy(target, {
            get(t, prop, receiver) {
                if (prop in t) {
                    return t[prop];
                }
                return function(...args) {
                    console.warn(`[3D Studio Helpers] Undefined helper called: helpers.${String(prop)}`, args);
                    if (args[0] && (args[0].isBufferGeometry || args[0].isGeometry || args[0].isObject3D)) {
                        return args[0];
                    }
                    return null;
                };
            }
        });
    },

    showErrorNotification(msg) {
        const banner = document.getElementById('three-error-banner');
        const text = document.getElementById('three-error-message');
        if (banner && text) {
            text.textContent = msg || '3D Execution Error';
            banner.style.display = 'flex';
        }
    },

    clearErrorNotification() {
        const banner = document.getElementById('three-error-banner');
        if (banner) banner.style.display = 'none';
    },

    postProcessModelGeometries(rootGroup) {
        if (!rootGroup) return;
        rootGroup.traverse(child => {
            if (child.isMesh && child.geometry) {
                let geom = child.geometry;

                // 1. Auto-weld duplicate/split vertices to seal cracks and holes
                if (typeof THREE.BufferGeometryUtils !== 'undefined' && typeof THREE.BufferGeometryUtils.mergeVertices === 'function') {
                    try {
                        const welded = THREE.BufferGeometryUtils.mergeVertices(geom, 1e-4);
                        if (welded) {
                            child.geometry = welded;
                            geom = welded;
                        }
                    } catch(e) {
                        console.warn('[3D Studio Healer] Vertex weld note:', e);
                    }
                }

                // 2. Compute smooth vertex normals
                try {
                    geom.computeVertexNormals();
                } catch(e) {}

                // 3. Ensure double-sided material rendering
                if (child.material) {
                    const mats = Array.isArray(child.material) ? child.material : [child.material];
                    mats.forEach(m => {
                        m.side = THREE.DoubleSide;
                        m.shadowSide = THREE.DoubleSide;
                    });
                }

                child.castShadow = true;
                child.receiveShadow = true;
            }
        });
    },

    init() {
        if (this.isInitialized) return;
        this.canvasEl = document.getElementById('three-viewport-canvas');
        if (!this.canvasEl) return;
        if (typeof THREE === 'undefined') {
            console.warn('[3D Studio] THREE is not loaded.');
            return;
        }

        // Global Three.js compatibility aliases for LLM generated code
        if (!THREE.CatmullRomCurve && THREE.CatmullRomCurve3) THREE.CatmullRomCurve = THREE.CatmullRomCurve3;
        if (!THREE.SplineCurve3 && THREE.CatmullRomCurve3) THREE.SplineCurve3 = THREE.CatmullRomCurve3;
        if (!THREE.CubicBezierCurve && THREE.CubicBezierCurve3) THREE.CubicBezierCurve = THREE.CubicBezierCurve3;
        if (!THREE.QuadraticBezierCurve && THREE.QuadraticBezierCurve3) THREE.QuadraticBezierCurve = THREE.QuadraticBezierCurve3;
        if (!THREE.LineCurve && THREE.LineCurve3) THREE.LineCurve = THREE.LineCurve3;
        if (!THREE.Geometry && THREE.BufferGeometry) THREE.Geometry = THREE.BufferGeometry;
        if (THREE.BufferGeometry && !THREE.BufferGeometry.prototype.setAttribute && THREE.BufferGeometry.prototype.addAttribute) {
            THREE.BufferGeometry.prototype.setAttribute = THREE.BufferGeometry.prototype.addAttribute;
        }

        const wrapper = document.getElementById('three-canvas-wrapper');
        const width = wrapper ? wrapper.clientWidth : 600;
        const height = wrapper ? wrapper.clientHeight : 400;

        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0f1115);

        // Camera
        this.camera = new THREE.PerspectiveCamera(45, width / (height || 1), 0.1, 1000);
        this.camera.position.set(4, 4, 6);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ canvas: this.canvasEl, antialias: true, alpha: true });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

        // Lighting
        const hemiLight = new THREE.HemisphereLight(0xffffff, 0x334155, 0.7);
        this.scene.add(hemiLight);

        const dirLight = new THREE.DirectionalLight(0xffffff, 0.85);
        dirLight.position.set(6, 12, 8);
        dirLight.castShadow = true;
        dirLight.shadow.mapSize.width = 1024;
        dirLight.shadow.mapSize.height = 1024;
        this.scene.add(dirLight);

        const fillLight = new THREE.DirectionalLight(0x60a5fa, 0.35);
        fillLight.position.set(-6, -4, -6);
        this.scene.add(fillLight);

        // Ground Grid
        this.gridHelper = new THREE.GridHelper(20, 20, 0x3b82f6, 0x334155);
        this.gridHelper.position.y = -0.01;
        this.scene.add(this.gridHelper);

        // Model Group Container
        this.modelGroup = new THREE.Group();
        this.modelGroup.name = 'UserScene';
        this.scene.add(this.modelGroup);

        // Vertex Handles Group
        this.vertexHandlesGroup = new THREE.Group();
        this.vertexHandlesGroup.name = 'VertexHandles';
        this.scene.add(this.vertexHandlesGroup);

        // OrbitControls
        if (typeof THREE.OrbitControls !== 'undefined') {
            this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.05;
            this.controls.target.set(0, 0, 0);
        }

        // TransformControls
        if (typeof THREE.TransformControls !== 'undefined') {
            this.transformControls = new THREE.TransformControls(this.camera, this.renderer.domElement);
            this.transformControls.size = 0.75;
            this.transformControls.addEventListener('dragging-changed', (event) => {
                if (this.controls) this.controls.enabled = !event.value;
            });
            this.transformControls.addEventListener('change', () => {
                this.syncInspectorFromTransform();
            });
            this.scene.add(this.transformControls);
        }

        // Raycasting for object selection
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();

        this.canvasEl.addEventListener('pointerdown', (e) => {
            this.onPointerDown(e);
        });

        // Window resize
        window.addEventListener('resize', () => {
            this.onResize();
        });

        this.isInitialized = true;
        this.animate();
    },

    animate() {
        requestAnimationFrame(() => this.animate());
        if (this.controls) this.controls.update();
        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
    },

    onTabActivated() {
        if (!this.isInitialized) {
            this.init();
        }
        setTimeout(() => this.onResize(), 60);
    },

    onResize() {
        const wrapper = document.getElementById('three-canvas-wrapper');
        if (!wrapper || !this.renderer || !this.camera) return;
        const width = wrapper.clientWidth;
        const height = wrapper.clientHeight;
        if (width <= 0 || height <= 0) return;
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    },

    onPointerDown(event) {
        if (this.transformControls && this.transformControls.dragging) return;
        const rect = this.canvasEl.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        this.raycaster.setFromCamera(this.mouse, this.camera);

        // Check vertex handles first if vertex mode is active
        if (this.vertexMode && this.vertexHandlesGroup.children.length > 0) {
            const vertexHits = this.raycaster.intersectObjects(this.vertexHandlesGroup.children);
            if (vertexHits.length > 0) {
                const handle = vertexHits[0].object;
                if (this.transformControls) {
                    this.transformControls.attach(handle);
                    this.transformControls.setMode('translate');
                }
                return;
            }
        }

        // Check meshes in modelGroup
        const intersects = this.raycaster.intersectObjects(this.modelGroup.children, true);
        if (intersects.length > 0) {
            let hit = intersects[0].object;
            while (hit.parent && hit.parent !== this.modelGroup && hit.parent.type === 'Group') {
                hit = hit.parent;
            }
            this.selectObject(hit);
        } else {
            // Clicked empty space
            if (this.transformControls && !this.vertexMode) {
                this.transformControls.detach();
            }
            this.selectedObject = null;
            this.updateInspectorUI(null);
            this.highlightOutliner(null);
        }
    },

    selectObject(obj) {
        this.selectedObject = obj;
        if (this.transformControls) {
            this.transformControls.attach(obj);
            const activeTool = document.querySelector('.three-tool-btn.active');
            const toolId = activeTool ? activeTool.id : '';
            if (toolId === 'btn-tool-rotate') this.transformControls.setMode('rotate');
            else if (toolId === 'btn-tool-scale') this.transformControls.setMode('scale');
            else this.transformControls.setMode('translate');
        }
        this.updateInspectorUI(obj);
        this.highlightOutliner(obj);
        if (this.vertexMode) {
            this.rebuildVertexHandles(obj);
        }
    },

    setTransformMode(mode) {
        document.querySelectorAll('.three-toolbar #btn-tool-select, #btn-tool-translate, #btn-tool-rotate, #btn-tool-scale').forEach(b => b.classList.remove('active'));
        const btn = document.getElementById(`btn-tool-${mode}`);
        if (btn) btn.classList.add('active');

        if (!this.transformControls) return;
        if (mode === 'select') {
            this.transformControls.detach();
        } else if (this.selectedObject) {
            this.transformControls.attach(this.selectedObject);
            this.transformControls.setMode(mode);
        }
    },

    toggleVertexMode() {
        this.vertexMode = !this.vertexMode;
        const btn = document.getElementById('btn-tool-vertex');
        if (btn) btn.classList.toggle('active', this.vertexMode);

        if (this.vertexMode) {
            if (this.selectedObject) {
                this.rebuildVertexHandles(this.selectedObject);
            }
        } else {
            this.clearVertexHandles();
            if (this.selectedObject && this.transformControls) {
                this.transformControls.attach(this.selectedObject);
            }
        }
    },

    rebuildVertexHandles(mesh) {
        this.clearVertexHandles();
        if (!mesh || !mesh.geometry) return;

        const geom = mesh.geometry;
        const posAttr = geom.attributes.position;
        if (!posAttr) return;

        const handleGeom = new THREE.SphereGeometry(0.06, 8, 8);
        const handleMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });

        const count = Math.min(posAttr.count, 256); // clamp for performance
        for (let i = 0; i < count; i++) {
            const v = new THREE.Vector3().fromBufferAttribute(posAttr, i);
            mesh.localToWorld(v);
            const handle = new THREE.Mesh(handleGeom, handleMat);
            handle.position.copy(v);
            handle.userData = { vertexIndex: i, targetMesh: mesh };
            this.vertexHandlesGroup.add(handle);
        }
    },

    clearVertexHandles() {
        if (!this.vertexHandlesGroup) return;
        while (this.vertexHandlesGroup.children.length > 0) {
            const h = this.vertexHandlesGroup.children.pop();
            if (h.geometry) h.geometry.dispose();
        }
    },

    syncInspectorFromTransform() {
        if (!this.selectedObject) return;

        // If dragging vertex handle
        if (this.vertexMode && this.transformControls && this.transformControls.object && this.transformControls.object.userData.targetMesh) {
            const handle = this.transformControls.object;
            const mesh = handle.userData.targetMesh;
            const idx = handle.userData.vertexIndex;
            const localPos = handle.position.clone();
            mesh.worldToLocal(localPos);
            mesh.geometry.attributes.position.setXYZ(idx, localPos.x, localPos.y, localPos.z);
            mesh.geometry.attributes.position.needsUpdate = true;
            mesh.geometry.computeVertexNormals();
            return;
        }

        const posX = document.getElementById('mesh-pos-x');
        const posY = document.getElementById('mesh-pos-y');
        const posZ = document.getElementById('mesh-pos-z');
        if (posX && posY && posZ) {
            posX.value = this.selectedObject.position.x.toFixed(2);
            posY.value = this.selectedObject.position.y.toFixed(2);
            posZ.value = this.selectedObject.position.z.toFixed(2);
        }
    },

    toggleWireframe() {
        this.wireframeMode = !this.wireframeMode;
        const btn = document.getElementById('btn-tool-wireframe');
        if (btn) btn.classList.toggle('active', this.wireframeMode);

        if (this.modelGroup) {
            this.modelGroup.traverse((child) => {
                if (child.isMesh && child.material) {
                    if (Array.isArray(child.material)) {
                        child.material.forEach(m => m.wireframe = this.wireframeMode);
                    } else {
                        child.material.wireframe = this.wireframeMode;
                    }
                }
            });
        }
    },

    toggleAutoRotate() {
        if (!this.controls) return;
        this.controls.autoRotate = !this.controls.autoRotate;
        const btn = document.getElementById('btn-tool-autorotate');
        if (btn) btn.classList.toggle('active', this.controls.autoRotate);
    },

    toggleGrid() {
        if (!this.gridHelper) return;
        this.gridHelper.visible = !this.gridHelper.visible;
        const btn = document.getElementById('btn-tool-grid');
        if (btn) btn.classList.toggle('active', this.gridHelper.visible);
    },

    resetCamera() {
        if (!this.camera || !this.controls) return;
        this.camera.position.set(4, 4, 6);
        this.camera.lookAt(0, 0, 0);
        this.controls.target.set(0, 0, 0);
        this.controls.update();
    },

    updateSelectedPosition() {
        if (!this.selectedObject) return;
        const x = parseFloat(document.getElementById('mesh-pos-x')?.value || 0);
        const y = parseFloat(document.getElementById('mesh-pos-y')?.value || 0);
        const z = parseFloat(document.getElementById('mesh-pos-z')?.value || 0);
        this.selectedObject.position.set(x, y, z);
    },

    updateSelectedMaterial() {
        if (!this.selectedObject || !this.selectedObject.material) return;
        const mat = Array.isArray(this.selectedObject.material) ? this.selectedObject.material[0] : this.selectedObject.material;
        
        const colorInput = document.getElementById('mesh-color-picker');
        const colorHex = document.getElementById('mesh-color-hex');
        if (colorInput && mat.color) {
            mat.color.set(colorInput.value);
            if (colorHex) colorHex.textContent = colorInput.value;
        }

        const emissiveInput = document.getElementById('mesh-emissive-picker');
        const emissiveHex = document.getElementById('mesh-emissive-hex');
        if (emissiveInput && mat.emissive) {
            mat.emissive.set(emissiveInput.value);
            if (emissiveHex) emissiveHex.textContent = emissiveInput.value;
        }

        const roughSlider = document.getElementById('mesh-roughness-slider');
        const roughVal = document.getElementById('mesh-roughness-val');
        if (roughSlider && 'roughness' in mat) {
            mat.roughness = parseFloat(roughSlider.value);
            if (roughVal) roughVal.textContent = parseFloat(roughSlider.value).toFixed(2);
        }

        const metalSlider = document.getElementById('mesh-metalness-slider');
        const metalVal = document.getElementById('mesh-metalness-val');
        if (metalSlider && 'metalness' in mat) {
            mat.metalness = parseFloat(metalSlider.value);
            if (metalVal) metalVal.textContent = parseFloat(metalSlider.value).toFixed(2);
        }

        const opacSlider = document.getElementById('mesh-opacity-slider');
        const opacVal = document.getElementById('mesh-opacity-val');
        if (opacSlider) {
            const opVal = parseFloat(opacSlider.value);
            mat.opacity = opVal;
            mat.transparent = opVal < 1.0;
            if (opacVal) opacVal.textContent = opVal.toFixed(2);
        }

        const wireToggle = document.getElementById('mesh-wireframe-toggle');
        if (wireToggle) {
            mat.wireframe = wireToggle.checked;
        }
        mat.needsUpdate = true;
    },

    changeSelectedMaterialType(newType) {
        if (!this.selectedObject || !this.selectedObject.material || !THREE[newType]) return;
        const oldMat = Array.isArray(this.selectedObject.material) ? this.selectedObject.material[0] : this.selectedObject.material;
        const params = {
            color: oldMat.color ? oldMat.color.clone() : new THREE.Color(0x3b82f6),
            map: oldMat.map || null,
            wireframe: !!oldMat.wireframe,
            opacity: oldMat.opacity !== undefined ? oldMat.opacity : 1.0,
            transparent: oldMat.transparent !== undefined ? oldMat.transparent : false
        };
        if (newType === 'MeshStandardMaterial' || newType === 'MeshPhysicalMaterial') {
            params.roughness = oldMat.roughness !== undefined ? oldMat.roughness : 0.5;
            params.metalness = oldMat.metalness !== undefined ? oldMat.metalness : 0.2;
        }
        if (oldMat.emissive) {
            params.emissive = oldMat.emissive.clone();
        }

        const newMat = new THREE[newType](params);
        this.selectedObject.material = newMat;
        this.updateInspectorUI(this.selectedObject);
    },

    updateInspectorUI(mesh) {
        const nameEl = document.getElementById('selected-mesh-name');
        const posX = document.getElementById('mesh-pos-x');
        const posY = document.getElementById('mesh-pos-y');
        const posZ = document.getElementById('mesh-pos-z');
        const matTypeSelect = document.getElementById('mesh-material-type');
        const colorInput = document.getElementById('mesh-color-picker');
        const colorHex = document.getElementById('mesh-color-hex');
        const emissiveInput = document.getElementById('mesh-emissive-picker');
        const emissiveHex = document.getElementById('mesh-emissive-hex');
        const roughSlider = document.getElementById('mesh-roughness-slider');
        const roughVal = document.getElementById('mesh-roughness-val');
        const metalSlider = document.getElementById('mesh-metalness-slider');
        const metalVal = document.getElementById('mesh-metalness-val');
        const opacSlider = document.getElementById('mesh-opacity-slider');
        const opacVal = document.getElementById('mesh-opacity-val');
        const wireToggle = document.getElementById('mesh-wireframe-toggle');

        const btnCrop = document.getElementById('btn-crop-chat-photo');
        const btnApplyPhoto = document.getElementById('btn-apply-full-photo');
        const btnCustomTex = document.getElementById('btn-upload-custom-tex');
        const btnRemoveTex = document.getElementById('btn-remove-tex');

        const previewImg = document.getElementById('texture-preview-img');
        const placeholder = document.getElementById('no-texture-placeholder');
        const metaDim = document.getElementById('texture-meta-dim');

        const uvInputs = [
            document.getElementById('uv-repeat-u'),
            document.getElementById('uv-repeat-v'),
            document.getElementById('uv-offset-u'),
            document.getElementById('uv-offset-v'),
            document.getElementById('uv-rotation-slider'),
            document.getElementById('uv-wrap-mode'),
            document.getElementById('btn-reset-uv')
        ];

        if (!mesh) {
            if (nameEl) nameEl.textContent = 'None';
            [posX, posY, posZ, matTypeSelect, colorInput, emissiveInput, roughSlider, metalSlider, opacSlider, wireToggle, btnCrop, btnApplyPhoto, btnCustomTex, btnRemoveTex, ...uvInputs].forEach(el => {
                if (el) el.disabled = true;
            });
            if (previewImg) previewImg.style.display = 'none';
            if (placeholder) placeholder.style.display = 'flex';
            if (metaDim) metaDim.textContent = 'Solid Material';
            return;
        }

        [posX, posY, posZ, matTypeSelect, colorInput, emissiveInput, roughSlider, metalSlider, opacSlider, wireToggle, btnCustomTex].forEach(el => {
            if (el) el.disabled = false;
        });

        const hasPhoto = !!window.__lastUploadedImage;
        if (btnCrop) btnCrop.disabled = !hasPhoto;
        if (btnApplyPhoto) btnApplyPhoto.disabled = !hasPhoto;

        if (nameEl) nameEl.textContent = mesh.name || mesh.type || 'Object';
        if (posX) posX.value = mesh.position.x.toFixed(2);
        if (posY) posY.value = mesh.position.y.toFixed(2);
        if (posZ) posZ.value = mesh.position.z.toFixed(2);

        const mat = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
        if (mat) {
            if (matTypeSelect) matTypeSelect.value = mat.type || 'MeshStandardMaterial';
            if (mat.color && colorInput) {
                const hex = '#' + mat.color.getHexString();
                colorInput.value = hex;
                if (colorHex) colorHex.textContent = hex;
            }
            if (mat.emissive && emissiveInput) {
                const hex = '#' + mat.emissive.getHexString();
                emissiveInput.value = hex;
                if (emissiveHex) emissiveHex.textContent = hex;
            }
            if ('roughness' in mat && roughSlider) {
                roughSlider.value = mat.roughness;
                if (roughVal) roughVal.textContent = mat.roughness.toFixed(2);
                roughSlider.disabled = false;
            } else if (roughSlider) {
                roughSlider.disabled = true;
            }
            if ('metalness' in mat && metalSlider) {
                metalSlider.value = mat.metalness;
                if (metalVal) metalVal.textContent = mat.metalness.toFixed(2);
                metalSlider.disabled = false;
            } else if (metalSlider) {
                metalSlider.disabled = true;
            }
            if (opacSlider) {
                opacSlider.value = mat.opacity !== undefined ? mat.opacity : 1.0;
                if (opacVal) opacVal.textContent = (mat.opacity !== undefined ? mat.opacity : 1.0).toFixed(2);
            }
            if (wireToggle) {
                wireToggle.checked = !!mat.wireframe;
            }

            // Texture Map Inspection
            if (mat.map) {
                const tex = mat.map;
                if (btnRemoveTex) btnRemoveTex.disabled = false;
                uvInputs.forEach(el => { if (el) el.disabled = false; });

                let imgSrc = '';
                if (tex.image) {
                    if (tex.image.src) imgSrc = tex.image.src;
                    else if (tex.image.toDataURL) imgSrc = tex.image.toDataURL();
                }
                if (previewImg && imgSrc) {
                    previewImg.src = imgSrc;
                    previewImg.style.display = 'block';
                    if (placeholder) placeholder.style.display = 'none';
                }
                if (metaDim) {
                    const w = tex.image ? (tex.image.naturalWidth || tex.image.width || 512) : 512;
                    const h = tex.image ? (tex.image.naturalHeight || tex.image.height || 512) : 512;
                    metaDim.textContent = `${w}x${h} Map`;
                }

                const repU = document.getElementById('uv-repeat-u');
                const repV = document.getElementById('uv-repeat-v');
                if (repU && tex.repeat) repU.value = tex.repeat.x.toFixed(1);
                if (repV && tex.repeat) repV.value = tex.repeat.y.toFixed(1);

                const offU = document.getElementById('uv-offset-u');
                const offV = document.getElementById('uv-offset-v');
                if (offU && tex.offset) offU.value = tex.offset.x.toFixed(2);
                if (offV && tex.offset) offV.value = tex.offset.y.toFixed(2);

                const rotSlider = document.getElementById('uv-rotation-slider');
                const rotVal = document.getElementById('uv-rotation-val');
                const deg = Math.round(((tex.rotation || 0) * (180 / Math.PI)) % 360);
                const posDeg = deg >= 0 ? deg : deg + 360;
                if (rotSlider) rotSlider.value = posDeg;
                if (rotVal) rotVal.textContent = `${posDeg}°`;

                const wrapSelect = document.getElementById('uv-wrap-mode');
                if (wrapSelect) {
                    if (tex.wrapS === THREE.ClampToEdgeWrapping) wrapSelect.value = 'ClampToEdgeWrapping';
                    else if (tex.wrapS === THREE.MirroredRepeatWrapping) wrapSelect.value = 'MirroredRepeatWrapping';
                    else wrapSelect.value = 'RepeatWrapping';
                }
            } else {
                if (btnRemoveTex) btnRemoveTex.disabled = true;
                uvInputs.forEach(el => { if (el) el.disabled = true; });
                if (previewImg) previewImg.style.display = 'none';
                if (placeholder) placeholder.style.display = 'flex';
                if (metaDim) metaDim.textContent = 'Solid Material';
            }
        }
    },

    updateUVMapping() {
        if (!this.selectedObject || !this.selectedObject.material) return;
        const mat = Array.isArray(this.selectedObject.material) ? this.selectedObject.material[0] : this.selectedObject.material;
        if (!mat.map) return;
        const tex = mat.map;

        const repU = parseFloat(document.getElementById('uv-repeat-u')?.value || 1.0);
        const repV = parseFloat(document.getElementById('uv-repeat-v')?.value || 1.0);
        tex.repeat.set(repU, repV);

        const offU = parseFloat(document.getElementById('uv-offset-u')?.value || 0.0);
        const offV = parseFloat(document.getElementById('uv-offset-v')?.value || 0.0);
        tex.offset.set(offU, offV);

        const rotDeg = parseFloat(document.getElementById('uv-rotation-slider')?.value || 0);
        const rotVal = document.getElementById('uv-rotation-val');
        if (rotVal) rotVal.textContent = `${rotDeg}°`;
        tex.center.set(0.5, 0.5);
        tex.rotation = rotDeg * (Math.PI / 180);

        const wrapMode = document.getElementById('uv-wrap-mode')?.value || 'RepeatWrapping';
        tex.wrapS = THREE[wrapMode] || THREE.RepeatWrapping;
        tex.wrapT = THREE[wrapMode] || THREE.RepeatWrapping;
        tex.needsUpdate = true;
    },

    setUVRepeatPreset(u, v) {
        const inputU = document.getElementById('uv-repeat-u');
        const inputV = document.getElementById('uv-repeat-v');
        if (inputU) inputU.value = u;
        if (inputV) inputV.value = v;
        this.updateUVMapping();
    },

    resetUVMapping() {
        const inputU = document.getElementById('uv-repeat-u');
        const inputV = document.getElementById('uv-repeat-v');
        const offU = document.getElementById('uv-offset-u');
        const offV = document.getElementById('uv-offset-v');
        const rotSlider = document.getElementById('uv-rotation-slider');
        const wrapSelect = document.getElementById('uv-wrap-mode');
        if (inputU) inputU.value = 1.0;
        if (inputV) inputV.value = 1.0;
        if (offU) offU.value = 0.0;
        if (offV) offV.value = 0.0;
        if (rotSlider) rotSlider.value = 0;
        if (wrapSelect) wrapSelect.value = 'RepeatWrapping';
        this.updateUVMapping();
    },

    applyTextureToSelected(url) {
        if (!this.selectedObject || !this.selectedObject.material) return;
        const mat = Array.isArray(this.selectedObject.material) ? this.selectedObject.material[0] : this.selectedObject.material;
        const loader = new THREE.TextureLoader();
        loader.load(url, (texture) => {
            texture.wrapS = THREE.RepeatWrapping;
            texture.wrapT = THREE.RepeatWrapping;
            texture.repeat.set(1, 1);
            texture.center.set(0.5, 0.5);
            if (mat.map) mat.map.dispose();
            mat.map = texture;
            mat.needsUpdate = true;
            this.updateInspectorUI(this.selectedObject);
        });
    },

    applyChatPhotoToSelected() {
        const photo = getLastUploadedPhoto();
        if (!this.selectedObject || !photo) return;
        this.applyTextureToSelected(photo);
    },

    handleCustomTextureUpload(input) {
        if (!input.files || !input.files[0] || !this.selectedObject) return;
        const file = input.files[0];
        const reader = new FileReader();
        reader.onload = (e) => {
            this.applyTextureToSelected(e.target.result);
        };
        reader.readAsDataURL(file);
        input.value = '';
    },

    removeTextureFromSelected() {
        if (!this.selectedObject || !this.selectedObject.material) return;
        const mat = Array.isArray(this.selectedObject.material) ? this.selectedObject.material[0] : this.selectedObject.material;
        if (mat.map) {
            mat.map.dispose();
            mat.map = null;
            mat.needsUpdate = true;
        }
        this.updateInspectorUI(this.selectedObject);
    },

    updatePhotoTextureButtons() {
        const hasPhoto = !!getLastUploadedPhoto();
        const hasMesh = !!this.selectedObject;
        const btnCrop = document.getElementById('btn-crop-chat-photo');
        const btnApply = document.getElementById('btn-apply-full-photo');
        if (btnCrop) btnCrop.disabled = !(hasPhoto && hasMesh);
        if (btnApply) btnApply.disabled = !(hasPhoto && hasMesh);
    },

    /* Photo Texture Cropper State & Methods */
    cropper: {
        img: null,
        cropX: 0,
        cropY: 0,
        cropW: 100,
        cropH: 100,
        ratio: 'free',
        scale: 1,
        isDragging: false,
        dragMode: null,
        startX: 0,
        startY: 0,
        origX: 0,
        origY: 0,
        origW: 0,
        origH: 0
    },

    openPhotoCropper() {
        const photo = getLastUploadedPhoto();
        if (!photo) {
            alert('No photo was attached or uploaded in chat yet.');
            return;
        }
        if (!this.selectedObject) {
            alert('Please select a 3D mesh first to apply the cropped texture.');
            return;
        }
        const modal = document.getElementById('photo-texture-crop-modal');
        if (modal) modal.style.display = 'flex';

        const canvas = document.getElementById('photo-crop-canvas');
        const img = new Image();
        img.onload = () => {
            this.cropper.img = img;
            const maxW = Math.min(620, window.innerWidth * 0.85);
            const maxH = Math.min(380, window.innerHeight * 0.5);
            const scaleW = maxW / img.naturalWidth;
            const scaleH = maxH / img.naturalHeight;
            const scale = Math.min(scaleW, scaleH, 1.0);
            this.cropper.scale = scale;

            canvas.width = Math.round(img.naturalWidth * scale);
            canvas.height = Math.round(img.naturalHeight * scale);

            const initW = Math.round(canvas.width * 0.6);
            const initH = this.cropper.ratio === '1:1' ? initW : Math.round(canvas.height * 0.6);
            this.cropper.cropW = Math.min(initW, canvas.width);
            this.cropper.cropH = Math.min(initH, canvas.height);
            this.cropper.cropX = Math.round((canvas.width - this.cropper.cropW) / 2);
            this.cropper.cropY = Math.round((canvas.height - this.cropper.cropH) / 2);

            this.setupCropCanvasListeners(canvas);
            this.renderCropCanvas();
        };
        img.src = window.__lastUploadedImage;
    },

    setupCropCanvasListeners(canvas) {
        if (canvas._hasCropListeners) return;
        canvas._hasCropListeners = true;

        const getPos = (e) => {
            const rect = canvas.getBoundingClientRect();
            return {
                x: (e.clientX - rect.left) * (canvas.width / rect.width),
                y: (e.clientY - rect.top) * (canvas.height / rect.height)
            };
        };

        const getHandle = (x, y) => {
            const { cropX, cropY, cropW, cropH } = this.cropper;
            const pad = 16;
            if (Math.hypot(x - cropX, y - cropY) < pad) return 'nw';
            if (Math.hypot(x - (cropX + cropW), y - cropY) < pad) return 'ne';
            if (Math.hypot(x - (cropX + cropW), y - (cropY + cropH)) < pad) return 'se';
            if (Math.hypot(x - cropX, y - (cropY + cropH)) < pad) return 'sw';
            if (x >= cropX && x <= cropX + cropW && y >= cropY && y <= cropY + cropH) return 'move';
            return null;
        };

        canvas.addEventListener('pointerdown', (e) => {
            const pos = getPos(e);
            const handle = getHandle(pos.x, pos.y);
            if (!handle) return;
            canvas.setPointerCapture(e.pointerId);
            this.cropper.isDragging = true;
            this.cropper.dragMode = handle;
            this.cropper.startX = pos.x;
            this.cropper.startY = pos.y;
            this.cropper.origX = this.cropper.cropX;
            this.cropper.origY = this.cropper.cropY;
            this.cropper.origW = this.cropper.cropW;
            this.cropper.origH = this.cropper.cropH;
        });

        canvas.addEventListener('pointermove', (e) => {
            const pos = getPos(e);
            if (!this.cropper.isDragging) {
                const handle = getHandle(pos.x, pos.y);
                if (handle === 'nw' || handle === 'se') canvas.style.cursor = 'nwse-resize';
                else if (handle === 'ne' || handle === 'sw') canvas.style.cursor = 'nesw-resize';
                else if (handle === 'move') canvas.style.cursor = 'move';
                else canvas.style.cursor = 'crosshair';
                return;
            }

            const dx = pos.x - this.cropper.startX;
            const dy = pos.y - this.cropper.startY;
            let { origX, origY, origW, origH, dragMode, ratio } = this.cropper;

            if (dragMode === 'move') {
                this.cropper.cropX = Math.max(0, Math.min(canvas.width - origW, origX + dx));
                this.cropper.cropY = Math.max(0, Math.min(canvas.height - origH, origY + dy));
            } else if (dragMode === 'se') {
                let newW = Math.max(24, Math.min(canvas.width - origX, origW + dx));
                let newH = Math.max(24, Math.min(canvas.height - origY, origH + dy));
                if (ratio === '1:1') newH = newW = Math.min(newW, newH);
                else if (ratio === '2:1') newH = Math.round(newW / 2);
                else if (ratio === '1:2') newW = Math.round(newH / 2);
                this.cropper.cropW = newW;
                this.cropper.cropH = newH;
            } else if (dragMode === 'nw') {
                let newX = Math.max(0, Math.min(origX + origW - 24, origX + dx));
                let newY = Math.max(0, Math.min(origY + origH - 24, origY + dy));
                this.cropper.cropW = origX + origW - newX;
                this.cropper.cropH = origY + origH - newY;
                if (ratio === '1:1') this.cropper.cropH = this.cropper.cropW;
                this.cropper.cropX = newX;
                this.cropper.cropY = newY;
            }
            this.renderCropCanvas();
        });

        const stopDrag = (e) => {
            if (this.cropper.isDragging) {
                this.cropper.isDragging = false;
                this.cropper.dragMode = null;
                try { canvas.releasePointerCapture(e.pointerId); } catch(ex) {}
                this.renderCropCanvas();
            }
        };
        canvas.addEventListener('pointerup', stopDrag);
        canvas.addEventListener('pointercancel', stopDrag);
    },

    renderCropCanvas() {
        const canvas = document.getElementById('photo-crop-canvas');
        if (!canvas || !this.cropper.img) return;
        const ctx = canvas.getContext('2d');
        const { cropX, cropY, cropW, cropH } = this.cropper;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(this.cropper.img, 0, 0, canvas.width, canvas.height);

        // Dark dim overlay outside crop area
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.fillRect(0, 0, canvas.width, cropY);
        ctx.fillRect(0, cropY, cropX, cropH);
        ctx.fillRect(cropX + cropW, cropY, canvas.width - (cropX + cropW), cropH);
        ctx.fillRect(0, cropY + cropH, canvas.width, canvas.height - (cropY + cropH));

        // Crop rect outline
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.strokeRect(cropX, cropY, cropW, cropH);
        ctx.setLineDash([]);

        // Corner handles
        const handleSize = 8;
        ctx.fillStyle = '#ffffff';
        ctx.strokeStyle = '#0284c7';
        ctx.lineWidth = 2;
        const corners = [
            [cropX, cropY],
            [cropX + cropW, cropY],
            [cropX + cropW, cropY + cropH],
            [cropX, cropY + cropH]
        ];
        corners.forEach(([cx, cy]) => {
            ctx.fillRect(cx - handleSize / 2, cy - handleSize / 2, handleSize, handleSize);
            ctx.strokeRect(cx - handleSize / 2, cy - handleSize / 2, handleSize, handleSize);
        });
    },

    setCropRatio(ratio, btn) {
        this.cropper.ratio = ratio;
        document.querySelectorAll('.crop-ratio-btn').forEach(b => b.classList.remove('active'));
        if (btn) btn.classList.add('active');

        const canvas = document.getElementById('photo-crop-canvas');
        if (!canvas) return;
        let { cropX, cropY, cropW } = this.cropper;
        if (ratio === '1:1') {
            this.cropper.cropH = Math.min(cropW, canvas.height - cropY);
            this.cropper.cropW = this.cropper.cropH;
        } else if (ratio === '2:1') {
            this.cropper.cropH = Math.min(Math.round(cropW / 2), canvas.height - cropY);
        } else if (ratio === '1:2') {
            this.cropper.cropW = Math.min(Math.round(this.cropper.cropH / 2), canvas.width - cropX);
        }
        this.renderCropCanvas();
    },

    resetCropToFull() {
        const canvas = document.getElementById('photo-crop-canvas');
        if (!canvas) return;
        this.cropper.cropX = 0;
        this.cropper.cropY = 0;
        this.cropper.cropW = canvas.width;
        this.cropper.cropH = canvas.height;
        this.renderCropCanvas();
    },

    closePhotoCropper() {
        const modal = document.getElementById('photo-texture-crop-modal');
        if (modal) modal.style.display = 'none';
    },

    applyCroppedTexture() {
        if (!this.selectedObject || !this.cropper.img) return;
        const scale = this.cropper.scale || 1.0;
        const sx = Math.max(0, Math.round(this.cropper.cropX / scale));
        const sy = Math.max(0, Math.round(this.cropper.cropY / scale));
        const sw = Math.min(this.cropper.img.naturalWidth - sx, Math.round(this.cropper.cropW / scale));
        const sh = Math.min(this.cropper.img.naturalHeight - sy, Math.round(this.cropper.cropH / scale));

        if (sw <= 0 || sh <= 0) return;

        const offCanvas = document.createElement('canvas');
        offCanvas.width = sw;
        offCanvas.height = sh;
        const offCtx = offCanvas.getContext('2d');
        offCtx.drawImage(this.cropper.img, sx, sy, sw, sh, 0, 0, sw, sh);

        const tex = new THREE.CanvasTexture(offCanvas);
        tex.wrapS = THREE.RepeatWrapping;
        tex.wrapT = THREE.RepeatWrapping;
        tex.repeat.set(1, 1);
        tex.center.set(0.5, 0.5);

        const mat = Array.isArray(this.selectedObject.material) ? this.selectedObject.material[0] : this.selectedObject.material;
        if (mat.map) mat.map.dispose();
        mat.map = tex;
        mat.needsUpdate = true;

        this.closePhotoCropper();
        this.updateInspectorUI(this.selectedObject);
    },

    updateOutliner() {
        const outliner = document.getElementById('outliner-tree');
        if (!outliner || !this.modelGroup) return;

        outliner.innerHTML = '';
        const meshes = [];
        this.modelGroup.traverse((child) => {
            if (child.isMesh) meshes.push(child);
        });

        if (meshes.length === 0) {
            outliner.innerHTML = '<div class="outliner-empty">No objects in scene</div>';
            return;
        }

        meshes.forEach((m, idx) => {
            if (!m.name) m.name = `Mesh_${idx + 1}`;
            const item = document.createElement('div');
            item.className = 'outliner-item' + (this.selectedObject === m ? ' selected' : '');
            item.dataset.uuid = m.uuid;
            item.innerHTML = `
                <span class="material-symbols-outlined">deployed_code</span>
                <span>${escapeHtml(m.name)}</span>
            `;
            item.addEventListener('click', () => {
                this.selectObject(m);
            });
            outliner.appendChild(item);
        });
    },

    highlightOutliner(mesh) {
        document.querySelectorAll('.outliner-item').forEach(el => {
            el.classList.toggle('selected', mesh && el.dataset.uuid === mesh.uuid);
        });
    },

    toggleInspector() {
        const drawer = document.getElementById('three-inspector-drawer');
        if (drawer) drawer.classList.toggle('collapsed');
    },

    isModelGroupEmpty() {
        return !this.modelGroup || this.modelGroup.children.length === 0;
    },

    openCodeEditor() {
        if (this.currentCode && previewCodeEditor) {
            previewCodeEditor.value = this.currentCode;
            updateEditorLineNumbers();
        }
        switchPreviewTab('tab-code');
    },

    sanitizeModuleCode(code) {
        if (!code) return '';
        let result = code;

        // 1. Remove side-effect imports: import "something"; or import 'something';
        result = result.replace(/^[\t ]*import\s+['"][^'"]+['"];?/gm, '// [studio side-effect import removed]');

        // 2. Transform mixed default and named imports: import THREE, { OrbitControls } from "..."
        result = result.replace(/^[\t ]*import\s+([a-zA-Z0-9_$]+)\s*,\s*\{([\s\S]*?)\}\s*from\s*['"][^'"]+['"];?/gm, (match, defaultName, namedList) => {
            let defCode = '';
            if (defaultName === 'THREE') {
                defCode = '// [studio THREE mixed import]\n';
            } else {
                defCode = `const ${defaultName} = (typeof ${defaultName} !== "undefined" ? ${defaultName} : ((typeof THREE !== "undefined" && THREE.${defaultName}) || (typeof window !== "undefined" && window.${defaultName}) || __studioThree || {}));\n`;
            }
            const parts = namedList.split(',').map(s => s.trim()).filter(Boolean);
            const mapped = [];
            for (const part of parts) {
                const m = part.match(/^([a-zA-Z0-9_$]+)(?:\s+as\s+([a-zA-Z0-9_$]+))?$/);
                if (m) {
                    const orig = m[1];
                    const alias = m[2] || orig;
                    mapped.push(`${orig}: ${alias}`);
                }
            }
            const namedCode = mapped.length > 0
                ? `const { ${mapped.join(', ')} } = (typeof THREE !== "undefined" ? THREE : __studioThree);\n`
                : '';
            return defCode + namedCode;
        });

        // 3. Transform namespace imports: import * as Foo from "..."
        result = result.replace(/^[\t ]*import\s+\*\s+as\s+([a-zA-Z0-9_$]+)\s+from\s+['"][^'"]+['"];?/gm, (match, p1) => {
            if (p1 === 'THREE') return '// [studio THREE namespace import]';
            return `const ${p1} = (typeof ${p1} !== "undefined" ? ${p1} : ((typeof THREE !== "undefined" && THREE.${p1}) || (typeof window !== "undefined" && window.${p1}) || __studioThree || {}));`;
        });

        // 4. Transform default imports: import Foo from "..."
        result = result.replace(/^[\t ]*import\s+([a-zA-Z0-9_$]+)\s+from\s+['"][^'"]+['"];?/gm, (match, p1) => {
            if (p1 === 'THREE') return '// [studio THREE default import]';
            return `const ${p1} = (typeof ${p1} !== "undefined" ? ${p1} : ((typeof THREE !== "undefined" && THREE.${p1}) || (typeof window !== "undefined" && window.${p1}) || __studioThree || {}));`;
        });

        // 5. Transform named imports: import { A, B as C } from "..." (single or multi-line)
        result = result.replace(/^[\t ]*import\s*\{([\s\S]*?)\}\s*from\s*['"][^'"]+['"];?/gm, (match, names) => {
            const parts = names.split(',').map(s => s.trim()).filter(Boolean);
            const mapped = [];
            for (const part of parts) {
                const m = part.match(/^([a-zA-Z0-9_$]+)(?:\s+as\s+([a-zA-Z0-9_$]+))?$/);
                if (m) {
                    const orig = m[1];
                    const alias = m[2] || orig;
                    mapped.push(`${orig}: ${alias}`);
                }
            }
            if (mapped.length === 0) return '// [studio empty import]';
            return `const { ${mapped.join(', ')} } = (typeof THREE !== "undefined" ? THREE : __studioThree);`;
        });

        // 6. Transform export default
        result = result.replace(/^[\t ]*export\s+default\s+function\s*\(/gm, 'function createModel(');
        result = result.replace(/^[\t ]*export\s+default\s+function\s+([a-zA-Z0-9_$]+)/gm, (match, fnName) => {
            return `function ${fnName}`;
        });
        result = result.replace(/^[\t ]*export\s+default\s+class\s+([a-zA-Z0-9_$]+)?/gm, (match, clsName) => {
            return clsName ? `class ${clsName}` : 'const __studioExportedClass = class';
        });
        result = result.replace(/^[\t ]*export\s+default\s+([a-zA-Z0-9_$]+);?/gm, (match, name) => {
            return `if (typeof createModel === "undefined" && typeof ${name} === "function") { var createModel = ${name}; }`;
        });
        result = result.replace(/^[\t ]*export\s+default\s+/gm, '// [studio export default removed] ');

        // 7. Transform named exports
        result = result.replace(/^[\t ]*export\s+(?:async\s+)?function\s+/gm, 'function ');
        result = result.replace(/^[\t ]*export\s+class\s+/gm, 'class ');
        result = result.replace(/^[\t ]*export\s+(?:const|let|var)\s+/gm, (match) => match.replace('export', '').trim() + ' ');
        result = result.replace(/^[\t ]*export\s*\{[\s\S]*?\};?/gm, '// [studio removed export]');

        return result;
    },

    extractModelCode(code) {
        if (!code) return '';
        let cleaned = code.trim();

        // If markdown code block exists, strip fences
        cleaned = cleaned.replace(/^```[a-zA-Z0-9_-]*\n?/, '').replace(/\n?```$/, '').trim();

        // If it's a full HTML document or contains <script> tags, extract the Three.js script body
        if (cleaned.includes('<script') || cleaned.includes('<!DOCTYPE') || cleaned.includes('<html')) {
            const scriptMatches = cleaned.match(/<script\b[^>]*>([\s\S]*?)<\/script>/gi);
            if (scriptMatches && scriptMatches.length > 0) {
                let candidate = '';
                for (const sm of scriptMatches) {
                    const inner = sm.replace(/<script\b[^>]*>/i, '').replace(/<\/script>/i, '').trim();
                    if (inner.includes('THREE') || inner.includes('createModel') || inner.includes('scene.add') || inner.includes('from \'three\'') || inner.includes('from "three"')) {
                        candidate += '\n' + inner;
                    }
                }
                if (candidate.trim().length > 0) {
                    cleaned = candidate.trim();
                }
            }
        }

        // Sanitize ES module import/export statements
        cleaned = this.sanitizeModuleCode(cleaned);

        // Auto-balance unclosed functions, braces, and repair truncated statements
        cleaned = this.balanceAndRepairCode(cleaned);

        return cleaned;
    },

    /**
     * Balances unclosed braces, brackets, and parentheses in generated JavaScript code.
     * Also detects and repairs unclosed strings/template literals and truncated trailing lines.
     */
    balanceAndRepairCode(code) {
        if (!code || typeof code !== 'string') return '';
        const cleaned = code.trim();

        function tryBalance(source) {
            let inSingle = false;
            let inDouble = false;
            let inTemplate = false;
            let inLineComment = false;
            let inBlockComment = false;
            const stack = [];

            for (let i = 0; i < source.length; i++) {
                const c = source[i];
                const next = source[i + 1] || '';

                if (inLineComment) {
                    if (c === '\n') inLineComment = false;
                    continue;
                }
                if (inBlockComment) {
                    if (c === '*' && next === '/') {
                        inBlockComment = false;
                        i++;
                    }
                    continue;
                }

                // Count preceding backslashes to check if quote is escaped
                let backslashCount = 0;
                for (let j = i - 1; j >= 0 && source[j] === '\\'; j--) {
                    backslashCount++;
                }
                const isEscaped = (backslashCount % 2 === 1);

                if (inSingle) {
                    if (c === '\'' && !isEscaped) inSingle = false;
                    continue;
                }
                if (inDouble) {
                    if (c === '"' && !isEscaped) inDouble = false;
                    continue;
                }
                if (inTemplate) {
                    if (c === '`' && !isEscaped) inTemplate = false;
                    continue;
                }

                // Comment starts
                if (c === '/' && next === '/') {
                    inLineComment = true;
                    i++;
                    continue;
                }
                if (c === '/' && next === '*') {
                    inBlockComment = true;
                    i++;
                    continue;
                }

                // String starts
                if (c === '\'') { inSingle = true; continue; }
                if (c === '"') { inDouble = true; continue; }
                if (c === '`') { inTemplate = true; continue; }

                if (c === '{' || c === '(' || c === '[') {
                    stack.push(c);
                } else if (c === '}') {
                    if (stack.length > 0 && stack[stack.length - 1] === '{') stack.pop();
                } else if (c === ')') {
                    if (stack.length > 0 && stack[stack.length - 1] === '(') stack.pop();
                } else if (c === ']') {
                    if (stack.length > 0 && stack[stack.length - 1] === '[') stack.pop();
                }
            }

            let completion = '';
            if (inSingle) completion += '\'';
            if (inDouble) completion += '"';
            if (inTemplate) completion += '`';

            while (stack.length > 0) {
                const open = stack.pop();
                if (open === '{') completion += '\n}';
                else if (open === '[') completion += ']';
                else if (open === '(') completion += ')';
            }

            return source + completion;
        }

        function testParses(src) {
            try {
                new Function('scene', 'THREE', 'inputImage', 'helpers', src);
                return true;
            } catch (e) {
                return false;
            }
        }

        // 1. Direct bracket balancing
        const candidate = tryBalance(cleaned);
        if (testParses(candidate)) {
            return candidate;
        }

        // 2. Truncation recovery: if the snippet was cut off mid-statement, peel back incomplete trailing lines
        const lines = cleaned.split('\n');
        for (let removeCount = 1; removeCount <= Math.min(10, lines.length - 1); removeCount++) {
            const truncatedLines = lines.slice(0, lines.length - removeCount);
            const truncatedCandidate = tryBalance(truncatedLines.join('\n'));
            if (testParses(truncatedCandidate)) {
                console.log(`[3D Studio] Recovered truncated code by trimming ${removeCount} trailing unparsed line(s).`);
                return truncatedCandidate;
            }
        }

        return candidate;
    },

    loadModelCode(code) {
        this.init();
        if (!this.modelGroup) return;

        this.currentCode = code || '';

        // Clean model group
        while (this.modelGroup.children.length > 0) {
            const obj = this.modelGroup.children.pop();
            if (obj.geometry) obj.geometry.dispose();
        }
        this.clearVertexHandles();
        this.clearErrorNotification();

        const cleaned = this.extractModelCode(code);
        if (!cleaned) return;

        const targetGroup = this.modelGroup;

        // Create proxy around THREE to intercept scene creation and renderer
        const ThreeProxy = new Proxy(THREE, {
            get(target, prop, receiver) {
                if (prop === 'Scene') {
                    return function() {
                        const grp = new THREE.Group();
                        targetGroup.add(grp);
                        return grp;
                    };
                }
                if (prop === 'WebGLRenderer') {
                    return function() {
                        return {
                            setSize: () => {},
                            render: () => {},
                            setPixelRatio: () => {},
                            shadowMap: {},
                            domElement: document.createElement('div')
                        };
                    };
                }
                // Common LLM curve/geometry naming fallbacks
                if (prop === 'CatmullRomCurve' && target.CatmullRomCurve3) return target.CatmullRomCurve3;
                if (prop === 'SplineCurve3' && target.CatmullRomCurve3) return target.CatmullRomCurve3;
                if (prop === 'CubicBezierCurve' && target.CubicBezierCurve3) return target.CubicBezierCurve3;
                if (prop === 'QuadraticBezierCurve' && target.QuadraticBezierCurve3) return target.QuadraticBezierCurve3;
                if (prop === 'LineCurve' && target.LineCurve3) return target.LineCurve3;
                if (prop === 'Geometry' && target.BufferGeometry) return target.BufferGeometry;

                // Auto-close open Lathe profiles to prevent holes
                if (prop === 'LatheGeometry' || prop === 'LatheBufferGeometry') {
                    return function(points, segments, phiStart, phiLength) {
                        if (Array.isArray(points) && points.length > 1) {
                            return ThreeStudio.helpers.createWatertightLathe(points, segments, phiStart, phiLength);
                        }
                        const ctor = target.LatheBufferGeometry || target.LatheGeometry;
                        return new ctor(points, segments, phiStart, phiLength);
                    };
                }

                // In Three.js r128, automatically route legacy Geometry constructors to modern BufferGeometry constructors
                if (typeof prop === 'string' && prop.endsWith('Geometry') && !prop.includes('Buffer')) {
                    const bufProp = prop.replace('Geometry', 'BufferGeometry');
                    if (typeof target[bufProp] === 'function') {
                        return target[bufProp];
                    }
                }

                if (prop === 'SRGBColorSpace') return target.sRGBEncoding || 'srgb';
                if (prop === 'helpers') return ThreeStudio.helpers;
                if (prop === 'BufferGeometryUtils') {
                    return target.BufferGeometryUtils || (typeof window !== 'undefined' && window.THREE?.BufferGeometryUtils) || null;
                }

                return Reflect.get(target, prop, receiver);
            }
        });

        // Dummy environment preventing HTML scripts from polluting the DOM or hanging
        const dummyElement = document.createElement('div');
        const fakeDoc = {
            body: {
                appendChild: () => {},
                style: {}
            },
            createElement: (tag) => document.createElement(tag),
            getElementById: (id) => dummyElement,
            querySelector: (sel) => dummyElement,
            querySelectorAll: (sel) => [],
            addEventListener: () => {},
            removeEventListener: () => {}
        };
        const fakeWin = {
            innerWidth: 600,
            innerHeight: 400,
            addEventListener: () => {},
            removeEventListener: () => {}
        };
        const fakeRaf = (cb) => {
            try { cb(0); } catch(e) {}
            return 1;
        };

        const hasThreeDecl = /(?:const|let|var)\s+THREE\b/.test(cleaned);
        const threeInjection = hasThreeDecl ? '' : 'let THREE = __studioThree;\n';

        const hasSceneDecl = /(?:const|let|var)\s+scene\b/.test(cleaned);
        const sceneInjection = hasSceneDecl ? '' : 'let scene = __studioScene;\n';

        const hasInputImageDecl = /(?:const|let|var)\s+inputImage\b/.test(cleaned);
        const inputImageInjection = hasInputImageDecl ? '' : 'let inputImage = __studioInputImage;\n';

        const hasHelpersDecl = /(?:const|let|var)\s+helpers\b/.test(cleaned);
        const helpersInjection = hasHelpersDecl ? '' : 'let helpers = __studioHelpers;\n';

        const inputPhoto = typeof getLastUploadedPhoto === 'function' ? getLastUploadedPhoto() : (window.__lastUploadedImage || '');

        try {
            const runner = new Function(
                '__studioScene', '__studioThree', '__studioInputImage', '__studioHelpers', 'document', 'window', 'requestAnimationFrame',
                `
                try {
                    ${threeInjection}
                    ${sceneInjection}
                    ${inputImageInjection}
                    ${helpersInjection}
                    ${cleaned}
                    let res = null;
                    if (typeof createModel === 'function') {
                        res = createModel(__studioScene, THREE, __studioInputImage, __studioHelpers);
                    } else if (typeof createScene === 'function') {
                        res = createScene(__studioScene, THREE, __studioInputImage, __studioHelpers);
                    } else if (typeof initModel === 'function') {
                        res = initModel(__studioScene, THREE, __studioInputImage, __studioHelpers);
                    }
                    if (Array.isArray(res)) {
                        for (const item of res) {
                            if (item && item.isObject3D && item !== __studioScene && !__studioScene.children.includes(item)) {
                                __studioScene.add(item);
                            }
                        }
                    } else if (res && res.isObject3D && res !== __studioScene && !__studioScene.children.includes(res)) {
                        __studioScene.add(res);
                    }
                } catch(e) {
                    console.error('[3D Studio Runtime Error]', e);
                    if (typeof ThreeStudio !== 'undefined' && ThreeStudio.showErrorNotification) {
                        ThreeStudio.showErrorNotification('3D Runtime Error: ' + (e.message || e));
                    }
                }
                `
            );
            runner(targetGroup, ThreeProxy, inputPhoto, (this.getSafeHelpers ? this.getSafeHelpers() : this.helpers), fakeDoc, fakeWin, fakeRaf);
        } catch (err) {
            console.error('[3D Studio Compilation Error]', err);
            this.showErrorNotification(`3D Code Syntax Error: ${err.message}`);
            return;
        }

        if (this.modelGroup.children.length === 0) {
            this.showErrorNotification('Notice: 3D model script executed, but no objects were added to the scene.');
        } else {
            this.clearErrorNotification();
        }

        // Apply automatic geometry healing: weld duplicate vertices, cap open tubes, compute smooth normals, double-side
        this.postProcessModelGeometries(this.modelGroup);

        // Center and fit model
        const box = new THREE.Box3().setFromObject(this.modelGroup);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        this.modelGroup.position.sub(center);

        const maxDim = Math.max(size.x, size.y, size.z);
        if (maxDim > 0 && this.camera && this.controls) {
            const dist = maxDim * 2.2;
            this.camera.position.set(dist * 0.7, dist * 0.7, dist);
            this.controls.target.set(0, 0, 0);
            this.controls.update();
        }

        // Hide empty state
        const emptyState = document.getElementById('three-empty-state');
        if (emptyState) emptyState.style.display = 'none';

        this.updateOutliner();

        // Double-side all meshes and find primary mesh / textures
        let firstMesh = null;
        let hasAnyTexture = false;
        this.modelGroup.traverse(c => {
            if (c.isMesh) {
                if (!firstMesh) firstMesh = c;
                if (c.material) {
                    const mats = Array.isArray(c.material) ? c.material : [c.material];
                    mats.forEach(m => {
                        if (m && m.map) hasAnyTexture = true;
                        if (m && m.side === THREE.FrontSide) {
                            m.side = THREE.DoubleSide;
                        }
                    });
                }
            }
        });

        // Auto-apply chat photo texture if uploaded and model lacks textures
        if (!hasAnyTexture && inputPhoto && firstMesh && firstMesh.material) {
            console.log('[3D Studio] Auto-applying chat photo texture to primary mesh:', firstMesh.name || 'Body');
            const mat = Array.isArray(firstMesh.material) ? firstMesh.material[0] : firstMesh.material;
            const loader = new THREE.TextureLoader();
            loader.load(inputPhoto, (texture) => {
                texture.wrapS = THREE.RepeatWrapping;
                texture.wrapT = THREE.RepeatWrapping;
                texture.repeat.set(1, 1);
                texture.center.set(0.5, 0.5);
                mat.map = texture;
                mat.needsUpdate = true;
                if (this.selectedObject === firstMesh) {
                    this.updateInspectorUI(firstMesh);
                }
                if (typeof showNotification === 'function') {
                    showNotification('📷 Applied chat photo texture to model', 'success');
                }
            });
        }

        if (firstMesh) {
            this.selectObject(firstMesh);
        } else {
            this.updateInspectorUI(null);
        }
    },

    openWithCode(code) {
        openPreviewPanel();
        switchPreviewTab('tab-3d');
        this.loadModelCode(code);
    },

    loadDemoModel() {
        const demoCode = `
            function createModel(scene, THREE) {
                // Sci-Fi Energy Crate with Glowing Core
                const root = new THREE.Group();
                root.name = 'SciFiCrate';

                // Outer Armor Frame
                const frameGeom = new THREE.BoxGeometry(2, 2, 2);
                const frameMat = new THREE.MeshStandardMaterial({
                    color: 0x1e293b,
                    metalness: 0.85,
                    roughness: 0.25
                });
                const frame = new THREE.Mesh(frameGeom, frameMat);
                frame.name = 'OuterFrame';
                frame.castShadow = true;
                frame.receiveShadow = true;
                root.add(frame);

                // Glowing Reactor Core
                const coreGeom = new THREE.SphereGeometry(0.7, 32, 32);
                const coreMat = new THREE.MeshStandardMaterial({
                    color: 0x38bdf8,
                    emissive: 0x0284c7,
                    emissiveIntensity: 0.9,
                    metalness: 0.1,
                    roughness: 0.1,
                    transparent: true,
                    opacity: 0.92
                });
                const core = new THREE.Mesh(coreGeom, coreMat);
                core.name = 'PlasmaCore';
                root.add(core);

                // Corner Bevel Accents
                const cornerGeom = new THREE.BoxGeometry(0.4, 0.4, 0.4);
                const cornerMat = new THREE.MeshStandardMaterial({
                    color: 0xf59e0b,
                    metalness: 0.9,
                    roughness: 0.2
                });
                const offsets = [-1, 1];
                offsets.forEach(x => {
                    offsets.forEach(y => {
                        offsets.forEach(z => {
                            const corner = new THREE.Mesh(cornerGeom, cornerMat);
                            corner.position.set(x, y, z);
                            corner.name = \`Corner_\${x}_\${y}_\${z}\`;
                            root.add(corner);
                        });
                    });
                });

                scene.add(root);
            }
        `;
        this.loadModelCode(demoCode);
    },

    exportGLTF() {
        if (!this.modelGroup || typeof THREE.GLTFExporter === 'undefined') {
            alert('GLTFExporter is not available or scene is empty.');
            return;
        }
        const exporter = new THREE.GLTFExporter();
        exporter.parse(this.modelGroup, (gltf) => {
            const blob = new Blob([gltf], { type: 'application/octet-stream' });
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = 'model_3d.glb';
            link.click();
            URL.revokeObjectURL(link.href);
        }, { binary: true });
    },

    exportOBJ() {
        if (!this.modelGroup || typeof THREE.OBJExporter === 'undefined') {
            alert('OBJExporter is not available or scene is empty.');
            return;
        }
        const exporter = new THREE.OBJExporter();
        const result = exporter.parse(this.modelGroup);
        const blob = new Blob([result], { type: 'text/plain' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = 'model_3d.obj';
        link.click();
        URL.revokeObjectURL(link.href);
    }
};

// ════════════════════════════════════════════════════════════════════════════════
//  Voice Recognition & Text-To-Speech (Web Speech API)
// ════════════════════════════════════════════════════════════════════════════════

let voiceSettings = {
    lang: localStorage.getItem('moecher_voice_lang') || 'auto',
    autoSend: localStorage.getItem('moecher_voice_auto_send') === 'true',
    autoRead: localStorage.getItem('moecher_voice_auto_read') === 'true',
    ttsVoice: localStorage.getItem('moecher_tts_voice') || 'default',
    ttsRate: parseFloat(localStorage.getItem('moecher_tts_rate') || '1.0')
};

let speechRecognitionInstance = null;
let isVoiceRecording = false;
let preSpeechInputValue = '';
let speechFinalTranscript = '';
let speechSilenceTimer = null;
let currentPlayingUtterance = null;
let currentPlayingMessageDiv = null;

function isSpeechRecognitionSupported() {
    return typeof window !== 'undefined' && ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window);
}

function getSpeechRecognitionClass() {
    if (typeof window === 'undefined') return null;
    return window.SpeechRecognition || window.webkitSpeechRecognition || null;
}

function initVoiceRecognitionUI() {
    const micBtn = document.getElementById('mic-btn');
    const langSelect = document.getElementById('voice-lang-select');
    const autoSendCheck = document.getElementById('voice-auto-send');
    const autoReadCheck = document.getElementById('voice-auto-read');
    const rateSlider = document.getElementById('tts-rate-slider');
    const rateVal = document.getElementById('tts-rate-val');

    if (langSelect) langSelect.value = voiceSettings.lang;
    if (autoSendCheck) autoSendCheck.checked = voiceSettings.autoSend;
    if (autoReadCheck) autoReadCheck.checked = voiceSettings.autoRead;
    if (rateSlider) rateSlider.value = voiceSettings.ttsRate;
    if (rateVal) rateVal.textContent = `${voiceSettings.ttsRate.toFixed(1)}x`;

    updateVoiceModeButtonUI();

    if (!isSpeechRecognitionSupported()) {
        if (micBtn) {
            micBtn.title = "Voice recognition not supported in this browser (Use Chrome, Edge, or Safari)";
            micBtn.style.opacity = '0.5';
        }
    }

    initTtsVoices();
}

function updateVoiceModeButtonUI() {
    const voiceModeBtn = document.getElementById('voice-mode-btn');
    const voiceModeIcon = document.getElementById('voice-mode-icon');
    if (!voiceModeBtn || !voiceModeIcon) return;

    if (voiceSettings.autoRead) {
        voiceModeBtn.classList.add('active');
        voiceModeIcon.textContent = 'volume_up';
        voiceModeBtn.title = 'Voice Mode ON (Model responses are read aloud)';
    } else {
        voiceModeBtn.classList.remove('active');
        voiceModeIcon.textContent = 'volume_off';
        voiceModeBtn.title = 'Voice Mode OFF (Click to enable auto read-aloud)';
    }
}

function toggleVoiceMode() {
    voiceSettings.autoRead = !voiceSettings.autoRead;
    localStorage.setItem('moecher_voice_auto_read', voiceSettings.autoRead);
    const autoReadCheck = document.getElementById('voice-auto-read');
    if (autoReadCheck) autoReadCheck.checked = voiceSettings.autoRead;
    updateVoiceModeButtonUI();
    showToast(voiceSettings.autoRead ? "Voice Mode enabled: Responses will be read aloud." : "Voice Mode disabled.");
    if (!voiceSettings.autoRead) {
        stopTtsAudio();
    }
}

function onVoiceLanguageChange(val) {
    voiceSettings.lang = val;
    localStorage.setItem('moecher_voice_lang', val);
    if (isVoiceRecording && speechRecognitionInstance) {
        stopVoiceRecognition();
        setTimeout(startVoiceRecognition, 200);
    }
}

function onVoiceAutoSendChange(checked) {
    voiceSettings.autoSend = checked;
    localStorage.setItem('moecher_voice_auto_send', checked);
}

function onVoiceAutoReadChange(checked) {
    voiceSettings.autoRead = checked;
    localStorage.setItem('moecher_voice_auto_read', checked);
    updateVoiceModeButtonUI();
}

function onTtsVoiceChange(val) {
    voiceSettings.ttsVoice = val;
    localStorage.setItem('moecher_tts_voice', val);
}

function onTtsRateChange(val) {
    voiceSettings.ttsRate = parseFloat(val) || 1.0;
    localStorage.setItem('moecher_tts_rate', voiceSettings.ttsRate);
    const rateVal = document.getElementById('tts-rate-val');
    if (rateVal) rateVal.textContent = `${voiceSettings.ttsRate.toFixed(1)}x`;
}

function toggleVoiceRecognition() {
    if (isVoiceRecording) {
        stopVoiceRecognition();
    } else {
        startVoiceRecognition();
    }
}

async function startVoiceRecognition() {
    if (isGenerating) return;
    const SpeechClass = getSpeechRecognitionClass();
    if (!SpeechClass) {
        showToast("Voice recognition is not supported in this browser. Please use Chrome, Edge, or Safari.", "error", 5000);
        return;
    }

    // Stop any ongoing speech playback so microphone doesn't pick it up
    stopTtsAudio();

    // Check secure context
    if (typeof window !== 'undefined' && window.isSecureContext === false) {
        showToast("Microphone dictation requires a secure origin (http://localhost or https://).", "error", 6000);
        return;
    }

    // Check for connected audio input devices first if enumerateDevices is available
    if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const audioInputs = devices.filter(d => d.kind === 'audioinput');
            if (devices.length > 0 && audioInputs.length === 0) {
                showToast("No microphone detected on this computer. Please connect a microphone, AirPods, or headset.", "error", 7000);
                return;
            }
        } catch (e) {
            // Proceed to getUserMedia check
        }
    }

    // Explicitly prompt for mic permission via getUserMedia if available
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            stream.getTracks().forEach(track => track.stop());
        } catch (permErr) {
            console.warn("getUserMedia permission error:", permErr);
            if (permErr.name === 'NotFoundError' || permErr.name === 'DevicesNotFoundError') {
                showToast("No microphone detected on this computer. Please connect a microphone, AirPods, or headset.", "error", 7000);
                return;
            }
            if (permErr.name === 'NotAllowedError' || permErr.name === 'PermissionDeniedError') {
                showToast("Microphone permission blocked. Click the lock/tune icon in the browser address bar to allow microphone access.", "warn", 6000);
                return;
            }
            showToast("Microphone access error: " + (permErr.message || permErr.name), "error", 6000);
            return;
        }
    }

    try {
        if (speechRecognitionInstance) {
            try { speechRecognitionInstance.abort(); } catch (e) {}
        }

        speechRecognitionInstance = new SpeechClass();
        speechRecognitionInstance.continuous = true;
        speechRecognitionInstance.interimResults = true;
        speechRecognitionInstance.maxAlternatives = 1;

        if (voiceSettings.lang && voiceSettings.lang !== 'auto') {
            speechRecognitionInstance.lang = voiceSettings.lang;
        } else {
            speechRecognitionInstance.lang = navigator.language || 'en-US';
        }

        preSpeechInputValue = chatInput.value;
        speechFinalTranscript = '';

        speechRecognitionInstance.onstart = () => {
            isVoiceRecording = true;
            updateVoiceRecordingUI(true);
        };

        speechRecognitionInstance.onresult = (event) => {
            let interimTranscript = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    speechFinalTranscript += (speechFinalTranscript ? ' ' : '') + transcript.trim();
                } else {
                    interimTranscript += transcript;
                }
            }

            const currentSpoken = (speechFinalTranscript + (interimTranscript ? ' ' + interimTranscript : '')).trim();
            const prefix = preSpeechInputValue.trim();
            
            if (prefix && currentSpoken) {
                chatInput.value = prefix + ' ' + currentSpoken;
            } else {
                chatInput.value = currentSpoken || prefix;
            }

            chatInput.style.height = 'auto';
            chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + 'px';
            sendBtn.disabled = chatInput.value.trim() === '';

            // Handle silence auto-send if enabled
            if (voiceSettings.autoSend && currentSpoken.length > 0) {
                if (speechSilenceTimer) clearTimeout(speechSilenceTimer);
                speechSilenceTimer = setTimeout(() => {
                    if (isVoiceRecording && chatInput.value.trim().length > 0) {
                        stopVoiceRecognition();
                        sendMessage();
                    }
                }, 2000);
            }
        };

        speechRecognitionInstance.onerror = (event) => {
            console.warn("Speech recognition error:", event.error);
            if (event.error === 'not-allowed') {
                if (typeof window !== 'undefined' && window.isSecureContext === false) {
                    showToast("Microphone requires a secure origin (http://localhost or https://).", "error", 6000);
                } else {
                    showToast("Microphone access blocked. Click the lock/tune icon in the browser address bar to allow microphone access.", "warn", 6000);
                }
                stopVoiceRecognition();
            } else if (event.error === 'network') {
                showToast("Speech service network error. (Note: Chromium browsers require an internet connection for built-in speech recognition).", "warn", 6000);
                stopVoiceRecognition();
            } else if (event.error === 'audio-capture') {
                showToast("No microphone detected. Please check your system audio input device.", "error", 5000);
                stopVoiceRecognition();
            } else if (event.error === 'service-not-allowed') {
                showToast("Speech recognition service is disabled or blocked in your browser.", "error", 5000);
                stopVoiceRecognition();
            } else if (event.error === 'no-speech') {
                // Ignore silent intervals or brief pauses without spamming the user
            } else if (event.error === 'aborted') {
                // Handled gracefully when user or stop() cancels
            } else {
                showToast(`Speech error: ${event.error}`, "warn", 4000);
                stopVoiceRecognition();
            }
        };

        speechRecognitionInstance.onend = () => {
            if (isVoiceRecording) {
                isVoiceRecording = false;
                updateVoiceRecordingUI(false);
            }
        };

        speechRecognitionInstance.start();
    } catch (err) {
        console.error("Failed to start speech recognition:", err);
        isVoiceRecording = false;
        updateVoiceRecordingUI(false);
        showToast("Could not start microphone: " + (err.message || err), "error");
    }
}

function stopVoiceRecognition() {
    if (speechSilenceTimer) {
        clearTimeout(speechSilenceTimer);
        speechSilenceTimer = null;
    }
    if (speechRecognitionInstance) {
        try { speechRecognitionInstance.stop(); } catch (e) {}
    }
    isVoiceRecording = false;
    updateVoiceRecordingUI(false);
    chatInput.focus();
}

function cancelVoiceRecognition() {
    if (speechSilenceTimer) {
        clearTimeout(speechSilenceTimer);
        speechSilenceTimer = null;
    }
    if (speechRecognitionInstance) {
        try { speechRecognitionInstance.abort(); } catch (e) {}
    }
    isVoiceRecording = false;
    chatInput.value = preSpeechInputValue;
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + 'px';
    sendBtn.disabled = chatInput.value.trim() === '';
    updateVoiceRecordingUI(false);
    chatInput.focus();
}

function updateVoiceRecordingUI(recording) {
    const micBtn = document.getElementById('mic-btn');
    const micIcon = document.getElementById('mic-icon');
    const indicator = document.getElementById('voice-indicator');

    if (recording) {
        if (micBtn) {
            micBtn.classList.add('recording');
            micBtn.title = "Listening... (Click to stop)";
        }
        if (micIcon) micIcon.textContent = 'mic';
        if (indicator) indicator.classList.remove('hidden');
    } else {
        if (micBtn) {
            micBtn.classList.remove('recording');
            micBtn.title = "Voice input / Dictation (Click to talk)";
        }
        if (micIcon) micIcon.textContent = 'mic';
        if (indicator) indicator.classList.add('hidden');
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Text-To-Speech (TTS) Engine
// ─────────────────────────────────────────────────────────────────────────────

function cleanTextForSpeech(raw) {
    if (!raw) return '';
    let text = raw;

    // Remove tool activity markup if present
    text = text.replace(/<div class="tool-activity-block[\s\S]*?<\/div>/gi, '');
    
    // Replace full code blocks with a brief spoken placeholder
    text = text.replace(/```[\s\S]*?```/g, ' [code snippet omitted] ');
    
    // Remove inline code ticks
    text = text.replace(/`([^`]+)`/g, '$1');
    
    // Remove HTML tags
    text = text.replace(/<\/?[^>]+(>|$)/g, ' ');
    
    // Clean markdown links [label](url) -> label
    text = text.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');
    
    // Remove markdown headers, bold, italics, quotes
    text = text.replace(/#{1,6}\s+/g, '');
    text = text.replace(/[*_]{1,3}([^*_]+)[*_]{1,3}/g, '$1');
    text = text.replace(/^\s*>\s*/gm, '');
    text = text.replace(/^[-*+]\s+/gm, '');
    
    // Clean URLs
    text = text.replace(/https?:\/\/\S+/gi, 'web link');
    
    // Collapse excess whitespace
    text = text.replace(/\s+/g, ' ').trim();
    
    // Truncate if gigantic to avoid endless speaking
    if (text.length > 2500) {
        text = text.slice(0, 2500) + '... and more details in the message.';
    }
    
    return text;
}

function initTtsVoices() {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;

    const select = document.getElementById('tts-voice-select');
    if (!select) return;

    function populate() {
        const voices = window.speechSynthesis.getVoices();
        if (!voices || voices.length === 0) return;

        select.innerHTML = '<option value="default">Default System Voice</option>';
        voices.forEach(voice => {
            const opt = document.createElement('option');
            opt.value = voice.voiceURI;
            opt.textContent = `${voice.name} (${voice.lang})${voice.default ? ' — Default' : ''}`;
            if (voice.voiceURI === voiceSettings.ttsVoice) {
                opt.selected = true;
            }
            select.appendChild(opt);
        });
    }

    populate();
    if (window.speechSynthesis.onvoiceschanged !== undefined) {
        window.speechSynthesis.onvoiceschanged = populate;
    }
}

function stopTtsAudio() {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel();
    }
    currentPlayingUtterance = null;
    const voiceModeBtn = document.getElementById('voice-mode-btn');
    if (voiceModeBtn) voiceModeBtn.classList.remove('speaking');

    if (currentPlayingMessageDiv) {
        const playBtn = currentPlayingMessageDiv.querySelector('.read-aloud-btn');
        if (playBtn) {
            playBtn.classList.remove('playing');
            const icon = playBtn.querySelector('.material-symbols-outlined');
            if (icon) icon.textContent = 'volume_up';
        }
        currentPlayingMessageDiv = null;
    }
}

function speakAssistantMessage(rawText, messageElement = null) {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
        showToast("Text-to-speech is not supported in this browser.");
        return;
    }

    stopTtsAudio();

    const spokenText = cleanTextForSpeech(rawText);
    if (!spokenText) return;

    const utterance = new SpeechSynthesisUtterance(spokenText);
    utterance.rate = voiceSettings.ttsRate || 1.0;

    // Pick voice if selected
    if (voiceSettings.ttsVoice && voiceSettings.ttsVoice !== 'default') {
        const voices = window.speechSynthesis.getVoices();
        const matched = voices.find(v => v.voiceURI === voiceSettings.ttsVoice);
        if (matched) utterance.voice = matched;
    } else if (voiceSettings.lang && voiceSettings.lang !== 'auto') {
        utterance.lang = voiceSettings.lang;
    }

    const voiceModeBtn = document.getElementById('voice-mode-btn');
    let btnIcon = null;
    let playBtn = null;

    if (messageElement) {
        currentPlayingMessageDiv = messageElement;
        playBtn = messageElement.querySelector('.read-aloud-btn');
        if (playBtn) {
            playBtn.classList.add('playing');
            btnIcon = playBtn.querySelector('.material-symbols-outlined');
            if (btnIcon) btnIcon.textContent = 'pause';
        }
    }

    utterance.onstart = () => {
        if (voiceModeBtn) voiceModeBtn.classList.add('speaking');
    };

    utterance.onend = () => {
        stopTtsAudio();
    };

    utterance.onerror = (e) => {
        console.warn("TTS error:", e);
        stopTtsAudio();
    };

    currentPlayingUtterance = utterance;
    window.speechSynthesis.speak(utterance);
}

function toggleReadAloudMessage(btn) {
    const msgDiv = btn.closest('.message.assistant');
    if (!msgDiv) return;

    // If this message is already playing, toggle stop
    if (currentPlayingMessageDiv === msgDiv) {
        stopTtsAudio();
        return;
    }

    const contentEl = msgDiv.querySelector('.msg-content');
    const textToSpeak = contentEl ? contentEl.innerText : '';
    if (textToSpeak) {
        speakAssistantMessage(textToSpeak, msgDiv);
    }
}

function copyMessageText(btn) {
    const msgDiv = btn.closest('.message');
    if (!msgDiv) return;
    const contentEl = msgDiv.querySelector('.msg-content');
    if (!contentEl) return;

    // Get plain text but omit footer actions
    const clone = contentEl.cloneNode(true);
    const footer = clone.querySelector('.msg-footer-actions');
    if (footer) footer.remove();
    const text = clone.innerText.trim();

    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(() => {
            btn.classList.add('copied');
            const origHtml = btn.innerHTML;
            btn.innerHTML = `<span class="material-symbols-outlined">done</span> Copied!`;
            setTimeout(() => {
                btn.classList.remove('copied');
                btn.innerHTML = origHtml;
            }, 1800);
        });
    }
}

function attachAssistantMessageActions(assistantMsgDiv, rawContent) {
    if (!assistantMsgDiv) return;
    const contentWrapper = assistantMsgDiv.querySelector('.msg-content');
    if (!contentWrapper) return;

    // Avoid duplicate actions bar
    if (contentWrapper.querySelector('.msg-footer-actions')) return;

    const footer = document.createElement('div');
    footer.className = 'msg-footer-actions';
    footer.innerHTML = `
        <button type="button" class="msg-action-btn read-aloud-btn" onclick="toggleReadAloudMessage(this)" title="Read message aloud (Text-to-Speech)">
            <span class="material-symbols-outlined">volume_up</span>
            <span>Read</span>
        </button>
        <button type="button" class="msg-action-btn copy-msg-btn" onclick="copyMessageText(this)" title="Copy message text">
            <span class="material-symbols-outlined">content_copy</span>
            <span>Copy</span>
        </button>
    `;
    contentWrapper.appendChild(footer);
}

// Run initialization on DOM load
document.addEventListener('DOMContentLoaded', () => {
    initProxyServiceWorker();
    initPreviewPanel();
    initExpertProfileUI();
    initAgenticSettingsUI();
    initSystemPromptSync();
    initModelSelector();
    initMCPUI();
    setupVisionDragAndDrop();
    ThreeStudio.init();
    initVoiceRecognitionUI();
});

