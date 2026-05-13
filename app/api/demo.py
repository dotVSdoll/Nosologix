# ruff: noqa: E501
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["demo"])


DEMO_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Med RAG Agent Demo</title>
  <style>
    :root {
      --ink: #12211c;
      --muted: #5e716b;
      --paper: #fffaf0;
      --blue: #2f6f9f;
      --green: #1e8a63;
      --amber: #d88a25;
      --red: #bd4a3d;
      --line: rgba(18, 33, 28, 0.16);
      --shadow: 0 26px 80px rgba(42, 70, 64, 0.18);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      color: var(--ink);
      font-family: "Aptos Display", "Segoe UI", Georgia, serif;
      background:
        radial-gradient(circle at 12% 18%, rgba(47, 111, 159, 0.23), transparent 28rem),
        radial-gradient(circle at 84% 4%, rgba(216, 138, 37, 0.18), transparent 22rem),
        linear-gradient(135deg, #f8f0dd 0%, #eef7ef 52%, #e6f2f8 100%);
      min-height: 100vh;
    }

    .shell {
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
      padding: 34px 0 46px;
    }

    .hero {
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 24px;
      align-items: stretch;
      margin-bottom: 24px;
    }

    .panel {
      background: rgba(255, 250, 240, 0.82);
      border: 1px solid var(--line);
      border-radius: 30px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
    }

    .intro {
      padding: 34px;
      position: relative;
      overflow: hidden;
    }

    .intro:after {
      content: "";
      position: absolute;
      width: 210px;
      height: 210px;
      right: -72px;
      bottom: -82px;
      border: 28px solid rgba(30, 138, 99, 0.16);
      border-radius: 50%;
    }

    .eyebrow {
      color: var(--green);
      font-size: 13px;
      font-weight: 800;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      margin: 0 0 12px;
    }

    h1 {
      max-width: 820px;
      font-size: clamp(38px, 7vw, 72px);
      line-height: 0.92;
      letter-spacing: -0.06em;
      margin: 0 0 18px;
    }

    .intro p {
      max-width: 680px;
      color: var(--muted);
      font-size: 18px;
      line-height: 1.55;
      margin: 0;
    }

    .metrics {
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
      padding: 18px;
    }

    .metric {
      padding: 18px;
      border-radius: 22px;
      background: rgba(255, 255, 255, 0.55);
      border: 1px solid var(--line);
    }

    .metric strong {
      display: block;
      font-size: 26px;
      letter-spacing: -0.04em;
    }

    .metric span {
      color: var(--muted);
      font-size: 13px;
    }

    .grid {
      display: grid;
      grid-template-columns: 420px 1fr;
      gap: 24px;
      align-items: start;
    }

    .card { padding: 22px; }

    label {
      display: block;
      font-size: 13px;
      font-weight: 800;
      margin: 16px 0 8px;
      color: #1d3b33;
    }

    textarea, input, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.76);
      color: var(--ink);
      font: inherit;
      padding: 13px 14px;
      outline: none;
    }

    textarea { min-height: 140px; resize: vertical; }

    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }

    .toggle {
      display: flex;
      align-items: center;
      gap: 10px;
      margin: 16px 0;
      color: var(--muted);
    }

    .toggle input { width: auto; }

    button {
      width: 100%;
      border: 0;
      border-radius: 20px;
      padding: 15px 18px;
      color: #fff;
      background: linear-gradient(135deg, var(--green), var(--blue));
      font-weight: 900;
      letter-spacing: 0.02em;
      cursor: pointer;
      box-shadow: 0 14px 26px rgba(47, 111, 159, 0.22);
    }

    button:disabled {
      cursor: wait;
      opacity: 0.68;
    }

    .output {
      min-height: 560px;
      padding: 22px;
    }

    .status {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 18px;
    }

    .pill {
      border-radius: 999px;
      padding: 8px 12px;
      background: rgba(255, 255, 255, 0.7);
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
      font-weight: 800;
    }

    .answer {
      white-space: pre-wrap;
      font-size: 18px;
      line-height: 1.62;
      padding: 20px;
      border-radius: 24px;
      background: rgba(255, 255, 255, 0.62);
      border: 1px solid var(--line);
    }

    h2 {
      font-size: 19px;
      letter-spacing: -0.03em;
      margin: 24px 0 12px;
    }

    .items {
      display: grid;
      gap: 10px;
    }

    .item {
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.56);
    }

    .item-title {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-weight: 900;
      margin-bottom: 6px;
    }

    .item p {
      color: var(--muted);
      line-height: 1.48;
      margin: 0;
    }

    .runs {
      margin-top: 24px;
      padding: 22px;
    }

    .run-button {
      width: auto;
      padding: 8px 11px;
      border-radius: 999px;
      font-size: 12px;
      box-shadow: none;
    }

    .error {
      color: var(--red);
      font-weight: 900;
    }

    @media (max-width: 900px) {
      .hero, .grid { grid-template-columns: 1fr; }
      .row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="panel intro">
        <p class="eyebrow">Interview Demo Console</p>
        <h1>Traceable Healthcare Agentic RAG</h1>
        <p>Run the LangGraph workflow, inspect citations, and review each agent step without leaving the browser.</p>
      </div>
      <aside class="panel metrics">
        <div class="metric"><strong id="metricStatus">ready</strong><span>workflow status</span></div>
        <div class="metric"><strong id="metricLatency">-</strong><span>total latency</span></div>
        <div class="metric"><strong id="metricRuns">-</strong><span>recent runs</span></div>
      </aside>
    </section>

    <section class="grid">
      <form class="panel card" id="askForm">
        <label for="question">Question</label>
        <textarea id="question">What is hypertension?</textarea>
        <div class="row">
          <div>
            <label for="engine">Workflow</label>
            <select id="engine">
              <option value="langgraph">LangGraph</option>
              <option value="linear">Linear</option>
            </select>
          </div>
          <div>
            <label for="topK">Top K</label>
            <input id="topK" type="number" min="1" max="10" value="3" />
          </div>
        </div>
        <div class="row">
          <div>
            <label for="minScore">Min Score</label>
            <input id="minScore" type="number" min="0" max="1" step="0.01" value="0.02" />
          </div>
          <div>
            <label for="minCitations">Min Citations</label>
            <input id="minCitations" type="number" min="1" max="5" value="1" />
          </div>
        </div>
        <label class="toggle"><input id="useLlm" type="checkbox" /> Use real LLM</label>
        <button id="submitBtn" type="submit">Run Agentic RAG</button>
      </form>

      <section class="panel output">
        <div class="status" id="status"></div>
        <div class="answer" id="answer">Submit a question to inspect the grounded answer and agent trace.</div>
        <h2>Citations</h2>
        <div class="items" id="citations"></div>
        <h2>Agent Steps</h2>
        <div class="items" id="steps"></div>
      </section>
    </section>

    <section class="panel runs">
      <div class="item-title">
        <h2 style="margin:0">Recent Runs</h2>
        <button class="run-button" id="refreshRuns" type="button">Refresh</button>
      </div>
      <div class="items" id="runs"></div>
    </section>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);

    function pill(text) {
      return `<span class="pill">${escapeHtml(text)}</span>`;
    }

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
      }[char]));
    }

    function renderRun(payload) {
      $("metricStatus").textContent = payload.workflow_status || "done";
      $("metricLatency").textContent = `${Math.round(payload.total_latency_ms || 0)} ms`;
      $("status").innerHTML = [
        pill(`status: ${payload.workflow_status}`),
        pill(`engine: ${payload.workflow_engine}`),
        pill(`risk: ${payload.answer?.risk_level}`),
        pill(`confidence: ${payload.answer?.confidence}`),
        pill(`provider: ${payload.answer?.provider}`)
      ].join("");
      $("answer").textContent = payload.answer?.answer || "No answer returned.";

      const citations = payload.answer?.citations || [];
      $("citations").innerHTML = citations.length ? citations.map((item) => `
        <article class="item">
          <div class="item-title">
            <span>${escapeHtml(item.citation_id)} · ${escapeHtml(item.title || item.document_id)}</span>
            <span>${Number(item.score || 0).toFixed(3)}</span>
          </div>
          <p>${escapeHtml(item.excerpt)}</p>
        </article>
      `).join("") : `<p class="error">No citations returned.</p>`;

      const steps = payload.steps || [];
      $("steps").innerHTML = steps.length ? steps.map((step, index) => `
        <article class="item">
          <div class="item-title">
            <span>${index + 1}. ${escapeHtml(step.name)} · ${escapeHtml(step.status)}</span>
            <span>${Math.round(step.latency_ms || 0)} ms</span>
          </div>
          <p>${escapeHtml(step.output_summary || step.summary || "")}</p>
        </article>
      `).join("") : `<p class="error">Trace was not included.</p>`;
    }

    async function runAgent(event) {
      event.preventDefault();
      $("submitBtn").disabled = true;
      $("submitBtn").textContent = "Running...";
      try {
        const response = await fetch("/agents/rag", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question: $("question").value,
            top_k: Number($("topK").value),
            min_score: Number($("minScore").value),
            min_citations: Number($("minCitations").value),
            use_llm: $("useLlm").checked,
            include_trace: true,
            workflow_engine: $("engine").value
          })
        });
        if (!response.ok) {
          throw new Error(`${response.status} ${await response.text()}`);
        }
        renderRun(await response.json());
        await loadRuns();
      } catch (error) {
        $("answer").innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
      } finally {
        $("submitBtn").disabled = false;
        $("submitBtn").textContent = "Run Agentic RAG";
      }
    }

    async function loadRuns() {
      const response = await fetch("/agents/runs?limit=8");
      if (!response.ok) return;
      const payload = await response.json();
      $("metricRuns").textContent = payload.count;
      $("runs").innerHTML = payload.runs.length ? payload.runs.map((run) => `
        <article class="item">
          <div class="item-title">
            <span>${escapeHtml(run.question)}</span>
            <button class="run-button" type="button" data-run-id="${escapeHtml(run.run_id)}">Open</button>
          </div>
          <p>${escapeHtml(run.workflow_engine)} · ${escapeHtml(run.workflow_status)} · ${Math.round(run.total_latency_ms || 0)} ms · ${escapeHtml(run.created_at)}</p>
        </article>
      `).join("") : `<p>No runs yet.</p>`;
    }

    async function openRun(runId) {
      const response = await fetch(`/agents/runs/${encodeURIComponent(runId)}`);
      if (response.ok) renderRun(await response.json());
    }

    $("askForm").addEventListener("submit", runAgent);
    $("refreshRuns").addEventListener("click", loadRuns);
    $("runs").addEventListener("click", (event) => {
      const runId = event.target?.dataset?.runId;
      if (runId) openRun(runId);
    });
    loadRuns();
  </script>
</body>
</html>
"""


@router.get("/demo", response_class=HTMLResponse)
def demo_page() -> HTMLResponse:
    return HTMLResponse(DEMO_HTML)
