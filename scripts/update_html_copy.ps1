# Script to update copy mechanism in HTML files to support offline / file:/// protocol execution
$wpMd = [System.IO.File]::ReadAllText("$PSScriptRoot\..\WHITE_PAPER_CEREBELLUM_ZDB.md", [System.Text.Encoding]::UTF8)
$wpHtml = [System.IO.File]::ReadAllText("$PSScriptRoot\..\WHITE_PAPER_CEREBELLUM_ZDB.html", [System.Text.Encoding]::UTF8)

# 1. Update buttons in WHITE_PAPER_CEREBELLUM_ZDB.html
$wpHtml = $wpHtml.Replace('<button class="nav-btn primary" onclick="copyLinkedInMarkdown()" id="copy-btn">', '<button class="nav-btn primary copy-trigger" onclick="copyLinkedInMarkdown()" id="copy-btn">')
$wpHtml = $wpHtml.Replace('<button class="nav-btn primary" onclick="copyLinkedInMarkdown()" style="margin: 0 auto; font-size: 1rem; padding: 12px 28px;">', '<button class="nav-btn primary copy-trigger" onclick="copyLinkedInMarkdown()" style="margin: 0 auto; font-size: 1rem; padding: 12px 28px;">')

# 2. Build the script block with embedded markdown and dual-layer clipboard fallback
$copyJsBlock = @"
        async function copyToClipboard(text) {
            // Layer 1: Modern async clipboard API
            if (navigator.clipboard && window.isSecureContext) {
                try {
                    await navigator.clipboard.writeText(text);
                    return true;
                } catch (e) {
                    console.warn('navigator.clipboard failed, falling back to execCommand', e);
                }
            }
            // Layer 2: execCommand fallback (works in local file:/// URLs)
            try {
                const ta = document.createElement('textarea');
                ta.value = text;
                ta.style.position = 'fixed';
                ta.style.left = '-9999px';
                ta.style.top = '-9999px';
                ta.setAttribute('readonly', '');
                document.body.appendChild(ta);
                ta.select();
                ta.setSelectionRange(0, 999999);
                const successful = document.execCommand('copy');
                document.body.removeChild(ta);
                if (successful) return true;
            } catch (e) {
                console.error('execCommand failed', e);
            }
            return false;
        }

        async function copyLinkedInMarkdown() {
            const rawEl = document.getElementById('raw-markdown-content');
            let text = rawEl ? rawEl.textContent.trim() : '';
            if (!text) {
                text = document.querySelector('.article-container') ? document.querySelector('.article-container').innerText : document.body.innerText;
            }

            const buttons = document.querySelectorAll('.copy-trigger');
            const ok = await copyToClipboard(text);

            if (ok) {
                buttons.forEach(btn => {
                    const orig = btn.getAttribute('data-original') || btn.innerHTML;
                    btn.setAttribute('data-original', orig);
                    btn.innerHTML = '<span>✅</span> Copied to Clipboard!';
                    btn.style.background = '#22c55e';
                    btn.style.color = '#ffffff';
                    setTimeout(() => {
                        btn.innerHTML = orig;
                        btn.style.background = '';
                        btn.style.color = '';
                    }, 2500);
                });
            } else {
                prompt('Copy Markdown manually (Ctrl+C, Enter):', text.substring(0, 500) + '...');
            }
        }
    </script>

    <script id="raw-markdown-content" type="text/plain">
$wpMd
    </script>
</body>
</html>
"@

# Replace old copyLinkedInMarkdown until end of file
$oldFuncStart = $wpHtml.IndexOf('        async function copyLinkedInMarkdown() {')
if ($oldFuncStart -gt 0) {
    $newWpHtml = $wpHtml.Substring(0, $oldFuncStart) + $copyJsBlock
    [System.IO.File]::WriteAllText("$PSScriptRoot\..\WHITE_PAPER_CEREBELLUM_ZDB.html", $newWpHtml, [System.Text.Encoding]::UTF8)
    Write-Host "Updated WHITE_PAPER_CEREBELLUM_ZDB.html successfully!"
} else {
    Write-Error "Could not find copyLinkedInMarkdown in WHITE_PAPER_CEREBELLUM_ZDB.html"
}

# Now update ARTICLE_FOLLOWUP_ZERO_DECODE_BYPASS.html
$artMd = [System.IO.File]::ReadAllText("$PSScriptRoot\..\ARTICLE_FOLLOWUP_ZERO_DECODE_BYPASS.md", [System.Text.Encoding]::UTF8)
$artHtml = [System.IO.File]::ReadAllText("$PSScriptRoot\..\ARTICLE_FOLLOWUP_ZERO_DECODE_BYPASS.html", [System.Text.Encoding]::UTF8)

$artHtml = $artHtml.Replace('<button class="nav-btn primary" onclick="copyLinkedInMarkdown()" id="copy-btn">', '<button class="nav-btn primary copy-trigger" onclick="copyLinkedInMarkdown()" id="copy-btn">')
$artHtml = $artHtml.Replace('<button class="nav-btn primary" onclick="copyLinkedInMarkdown()" style="margin: 0 auto; font-size: 1rem; padding: 12px 24px;">', '<button class="nav-btn primary copy-trigger" onclick="copyLinkedInMarkdown()" style="margin: 0 auto; font-size: 1rem; padding: 12px 24px;">')

$artCopyJsBlock = @"
        async function copyToClipboard(text) {
            // Layer 1: Modern async clipboard API
            if (navigator.clipboard && window.isSecureContext) {
                try {
                    await navigator.clipboard.writeText(text);
                    return true;
                } catch (e) {
                    console.warn('navigator.clipboard failed, falling back to execCommand', e);
                }
            }
            // Layer 2: execCommand fallback (works in local file:/// URLs)
            try {
                const ta = document.createElement('textarea');
                ta.value = text;
                ta.style.position = 'fixed';
                ta.style.left = '-9999px';
                ta.style.top = '-9999px';
                ta.setAttribute('readonly', '');
                document.body.appendChild(ta);
                ta.select();
                ta.setSelectionRange(0, 999999);
                const successful = document.execCommand('copy');
                document.body.removeChild(ta);
                if (successful) return true;
            } catch (e) {
                console.error('execCommand failed', e);
            }
            return false;
        }

        async function copyLinkedInMarkdown() {
            const rawEl = document.getElementById('raw-markdown-content');
            let text = rawEl ? rawEl.textContent.trim() : '';
            if (!text) {
                text = document.querySelector('.article-container') ? document.querySelector('.article-container').innerText : document.body.innerText;
            }

            const buttons = document.querySelectorAll('.copy-trigger');
            const ok = await copyToClipboard(text);

            if (ok) {
                buttons.forEach(btn => {
                    const orig = btn.getAttribute('data-original') || btn.innerHTML;
                    btn.setAttribute('data-original', orig);
                    btn.innerHTML = '<span>✅</span> Copied to Clipboard!';
                    btn.style.background = '#22c55e';
                    btn.style.color = '#ffffff';
                    setTimeout(() => {
                        btn.innerHTML = orig;
                        btn.style.background = '';
                        btn.style.color = '';
                    }, 2500);
                });
            } else {
                prompt('Copy Markdown manually (Ctrl+C, Enter):', text.substring(0, 500) + '...');
            }
        }
    </script>

    <script id="raw-markdown-content" type="text/plain">
$artMd
    </script>
</body>
</html>
"@

$oldArtFuncStart = $artHtml.IndexOf('        async function copyLinkedInMarkdown() {')
if ($oldArtFuncStart -gt 0) {
    $newArtHtml = $artHtml.Substring(0, $oldArtFuncStart) + $artCopyJsBlock
    [System.IO.File]::WriteAllText("$PSScriptRoot\..\ARTICLE_FOLLOWUP_ZERO_DECODE_BYPASS.html", $newArtHtml, [System.Text.Encoding]::UTF8)
    Write-Host "Updated ARTICLE_FOLLOWUP_ZERO_DECODE_BYPASS.html successfully!"
} else {
    Write-Error "Could not find copyLinkedInMarkdown in ARTICLE_FOLLOWUP_ZERO_DECODE_BYPASS.html"
}
