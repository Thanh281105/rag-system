/**
 * ArXiv RAG Frontend - App Logic
 * Handles chat interactions with the Rust backend
 */

const API_BASE = '';
let isProcessing = false;
let chatHistory = []; // Lưu lịch sử hội thoại

// ─── Init ────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initParticles();
    autoResizeTextarea();
    checkHealth();
});

// ─── Particles Background ────────────────────────────
function initParticles() {
    const container = document.getElementById('particles');
    for (let i = 0; i < 30; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDuration = (Math.random() * 15 + 10) + 's';
        particle.style.animationDelay = Math.random() * 10 + 's';
        particle.style.width = (Math.random() * 3 + 1) + 'px';
        particle.style.height = particle.style.width;
        container.appendChild(particle);
    }
}

// ─── Auto-resize Textarea ────────────────────────────
function autoResizeTextarea() {
    const textarea = document.getElementById('queryInput');
    textarea.addEventListener('input', () => {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    });
    textarea.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
        }
    });
}

// ─── Health Check ────────────────────────────────────
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        const badge = document.getElementById('badge-status');
        if (data.status === 'ok') {
            badge.innerHTML = '<span class="status-dot"></span> Online';
            badge.className = 'badge badge-green';
        }
    } catch {
        const badge = document.getElementById('badge-status');
        badge.innerHTML = '<span class="status-dot" style="background:var(--accent-amber)"></span> Offline';
        badge.className = 'badge badge-green';
        badge.style.background = 'rgba(245, 158, 11, 0.15)';
        badge.style.color = 'var(--accent-amber)';
        badge.style.borderColor = 'rgba(245, 158, 11, 0.2)';
    }
}

// ─── Submit Handler ──────────────────────────────────
async function handleSubmit(e) {
    if (e && e.preventDefault) e.preventDefault();
    if (isProcessing) return;

    const input = document.getElementById('queryInput');
    const question = input.value.trim();
    if (!question) return;

    // Hide welcome card
    const welcome = document.getElementById('welcomeCard');
    if (welcome) welcome.style.display = 'none';

    // Add user message
    addMessage(question, 'user');
    input.value = '';
    input.style.height = 'auto';

    // Show typing indicator
    const typingId = showTyping();
    isProcessing = true;
    setSubmitState(false);

    try {
        const res = await fetch(`${API_BASE}/api/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, top_k: 5, history: chatHistory }),
        });

        const data = await res.json();
        removeTyping(typingId);

        if (data.error) {
            addBotMessage(`❌ Lỗi: ${data.error}`, [], null, 0);
        } else {
            // Lưu vào lịch sử hội thoại
            chatHistory.push({ role: 'user', content: question });
            chatHistory.push({ role: 'assistant', content: data.answer });
            // Giữ tối đa 10 tin nhắn gần nhất
            if (chatHistory.length > 10) chatHistory = chatHistory.slice(-10);

            addBotMessage(
                data.answer,
                data.sources || [],
                data.agent_trace,
                data.processing_time_ms
            );
        }
    } catch (err) {
        removeTyping(typingId);
        addBotMessage(
            `⚠️ Không thể kết nối đến server. Vui lòng kiểm tra:\n` +
            `1. Qdrant đang chạy: \`docker-compose up -d\`\n` +
            `2. Backend đang chạy: \`cd rust_backend && cargo run\`\n\n` +
            `Lỗi: ${err.message}`,
            [], null, 0
        );
    }

    isProcessing = false;
    setSubmitState(true);
}

// ─── Add Messages ────────────────────────────────────
function addMessage(text, type) {
    const area = document.getElementById('messagesArea');
    const div = document.createElement('div');
    div.className = `message message-${type}`;
    div.innerHTML = `<div class="message-content">${escapeHtml(text)}</div>`;
    area.appendChild(div);
    scrollToBottom();
}

function addBotMessage(answer, sources, trace, timeMs) {
    const area = document.getElementById('messagesArea');
    const div = document.createElement('div');
    div.className = 'message message-bot';

    let sourcesHtml = '';
    if (sources && sources.length > 0) {
        sourcesHtml = `
            <div class="sources-section">
                <div class="sources-title">📚 ArXiv Sources (${sources.length})</div>
                ${sources.map(s => `
                    <div class="source-item">
                        <span class="source-title">${escapeHtml(s.doc_title || 'Paper')}</span>
                        <span class="source-meta">${escapeHtml(s.authors || '')} (${s.year || ''})</span>
                        ${s.arxiv_id ? `<a href="https://arxiv.org/abs/${s.arxiv_id}" target="_blank" class="arxiv-link">arXiv:${s.arxiv_id}</a>` : ''}
                        <br><span class="source-preview">${escapeHtml(s.text.substring(0, 150))}...</span>
                    </div>
                `).join('')}
            </div>`;
    }

    let actionsHtml = '';
    if (trace) {
        const traceId = 'trace-' + Date.now();
        actionsHtml = `
            <div class="response-actions">
                <button class="action-btn" onclick="showTrace('${traceId}')">🔍 Agent Trace</button>
                <button class="action-btn" onclick="copyText(this)">📋 Copy</button>
            </div>
            <div id="${traceId}" style="display:none">${JSON.stringify(trace)}</div>`;
    }

    let timeHtml = timeMs > 0 ? `<div class="processing-time">⚡ ${timeMs}ms</div>` : '';

    div.innerHTML = `
        <div class="bot-avatar">🧠</div>
        <div class="message-content">
            ${formatAnswer(answer)}
            ${sourcesHtml}
            ${actionsHtml}
            ${timeHtml}
        </div>`;

    area.appendChild(div);
    scrollToBottom();
}

// ─── Typing Indicator ────────────────────────────────
function showTyping() {
    const area = document.getElementById('messagesArea');
    const id = 'typing-' + Date.now();
    const div = document.createElement('div');
    div.id = id;
    div.className = 'message message-bot';
    div.innerHTML = `
        <div class="bot-avatar">🧠</div>
        <div class="message-content">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
            <span style="font-size:12px; color:var(--text-muted)">Đang phân tích câu hỏi...</span>
        </div>`;
    area.appendChild(div);
    scrollToBottom();
    return id;
}

function removeTyping(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// ─── Agent Trace Modal ───────────────────────────────
function showTrace(dataId) {
    const dataEl = document.getElementById(dataId);
    if (!dataEl) return;

    const trace = JSON.parse(dataEl.textContent);
    const modal = document.getElementById('traceModal');
    const content = document.getElementById('traceContent');

    const reviewerClass = trace.reviewer_result?.is_approved ? 'compliance-pass' : 'compliance-fail';
    const reviewerText = trace.reviewer_result?.is_approved ? '✅ APPROVED' : '❌ REJECTED';

    content.innerHTML = `
        <div class="trace-step">
            <h3>1. 🔀 Agent 1: RAG-Router</h3>
            <p>Decision: <strong>${trace.router_decision}</strong></p>
        </div>
        <div class="trace-step">
            <h3>2. 🌐 Translation (VN → EN)</h3>
            <pre>${escapeHtml(trace.translated_query || 'N/A')}</pre>
        </div>
        <div class="trace-step">
            <h3>3. 📝 Multi-Query Expansion (EN)</h3>
            <pre>${escapeHtml(trace.expanded_queries || 'N/A')}</pre>
        </div>
        <div class="trace-step">
            <h3>4. 🔍 Retrieval</h3>
            <p>Retrieved: ${trace.retrieved_count} docs → Reranked: ${trace.reranked_count} docs</p>
        </div>
        <div class="trace-step">
            <h3>5. 🧠 Agent 2: Analyst + Self-check</h3>
            <pre>${escapeHtml((trace.analyst_answer || 'N/A').substring(0, 500))}</pre>
        </div>
        <div class="trace-step">
            <h3>6. ⚖️ Agent 3: Reviewer ${trace.reviewer_triggered ? '(TRIGGERED)' : '(SKIPPED)'}</h3>
            <p class="${reviewerClass}">${trace.reviewer_triggered ? reviewerText : '⚡ Skipped (not needed)'}</p>
            ${trace.reviewer_result?.issues?.length > 0
            ? `<p>Issues: ${trace.reviewer_result.issues.join(', ')}</p>` : ''}
            ${trace.reviewer_triggered ? `<p>Retry count: ${trace.reviewer_result?.retry_count || 0}</p>` : ''}
        </div>`;

    modal.classList.add('active');
}

function closeTrace(e) {
    if (e && e.target !== document.getElementById('traceModal')) return;
    document.getElementById('traceModal').classList.remove('active');
}

// ─── Utilities ───────────────────────────────────────
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatAnswer(text) {
    // Simple markdown-like formatting
    return text
        .split('\n')
        .map(line => `<p>${escapeHtml(line)}</p>`)
        .join('')
        .replace(/<p><\/p>/g, '<br>');
}

function scrollToBottom() {
    const container = document.getElementById('chatContainer');
    setTimeout(() => container.scrollTop = container.scrollHeight, 50);
}

function setSubmitState(enabled) {
    document.getElementById('submitBtn').disabled = !enabled;
}

function askExample(btn) {
    document.getElementById('queryInput').value = btn.textContent;
    handleSubmit();
}

function copyText(btn) {
    const content = btn.closest('.message-content');
    const text = content.querySelector('p')?.textContent || content.textContent;
    navigator.clipboard.writeText(text);
    btn.textContent = '✅ Copied!';
    setTimeout(() => btn.textContent = '📋 Copy', 2000);
}

// ─── Keyboard shortcut ──────────────────────────────
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeTrace();
});
