import { useMemo, useState } from "react";
import styles from "./App.module.css";

const TABS = ["Ask", "AI", "Search", "Inspect", "Dead code"];
const WORKSPACE_TABS = ["Overview", "Ask", "Search", "Impact", "History"];
const SUGGESTIONS = [
  "How does authentication work?",
  "What is the main request flow?",
  "Which functions are related to payment processing?"
];
const LANGUAGE_COLORS = {
  python: "#4B8BBE",
  javascript: "#D4A200",
  typescript: "#3178C6",
  go: "#00ADD8",
  java: "#E76F51"
};
const TAB_DESCRIPTIONS = {
  Ask: "Grounded repository answers with streaming responses, retrieval evidence, and graph context.",
  AI: "File-aware semantic analysis for indexed repositories using the existing AI query flow.",
  Search: "Low-friction repository search for files, functions, and retrieval signals.",
  Inspect: "Reverse-call impact analysis for understanding caller blast radius.",
  "Dead code": "Functions with no inbound callers in the indexed graph."
};

export default function App() {
  const [repoUrl, setRepoUrl] = useState("");
  const [stats, setStats] = useState(null);
  const [activeTab, setActiveTab] = useState("Ask");
  const [workspaceView, setWorkspaceView] = useState("Overview");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [callChain, setCallChain] = useState([]);
  const [graphEdges, setGraphEdges] = useState([]);
  const [graphNodes, setGraphNodes] = useState([]);
  const [aiQuestion, setAiQuestion] = useState("");
  const [aiApiKey, setAiApiKey] = useState("");
  const [aiFilePath, setAiFilePath] = useState("");
  const [aiResult, setAiResult] = useState(null);
  const [impactFunction, setImpactFunction] = useState("");
  const [impactRows, setImpactRows] = useState([]);
  const [deadCodeRows, setDeadCodeRows] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState(null);
  const [sourcesData, setSourcesData] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [previewMode, setPreviewMode] = useState("nodes");
  const [activityLog, setActivityLog] = useState([]);
  const [copyTarget, setCopyTarget] = useState("");

  const hasIndexedRepo = Boolean(stats?.repo_path);
  const indexedFiles = stats?.files || [];
  const topFiles = indexedFiles.slice(0, 6);
  const previewLanguage = selectedNode ? inferLanguage(selectedNode.file) : null;
  const previewBreadcrumbs = selectedNode ? selectedNode.file.replace(/\\/g, "/").split("/") : [];
  const nodeBadges = selectedNode
    ? [
        previewLanguage?.label || "Code",
        `Line ${selectedNode.start_line}`,
        `${selectedNode.calls?.length || 0} calls`
      ]
    : [];

  const previewCandidates = useMemo(() => {
    const seen = new Set();
    const candidates = [];
    const pushCandidate = (item) => {
      if (!item?.node_id || seen.has(item.node_id)) {
        return;
      }
      seen.add(item.node_id);
      candidates.push(item);
    };

    graphNodes.forEach(pushCandidate);
    (sourcesData?.fused || []).forEach(pushCandidate);
    (aiResult?.matches || []).forEach(pushCandidate);
    impactRows.forEach((row) =>
      pushCandidate({ node_id: row.node_id, name: row.name, file: row.file, start_line: row.line })
    );
    deadCodeRows.forEach((row) =>
      pushCandidate({ node_id: row.node_id, name: row.name, file: row.file, start_line: row.line })
    );

    return candidates;
  }, [aiResult, deadCodeRows, graphNodes, impactRows, sourcesData]);

  const summaryCards = [
    {
      eyebrow: "Repository map",
      title: hasIndexedRepo ? `${indexedFiles.length} indexed files` : "Repository map",
      body: hasIndexedRepo
        ? `${stats?.function_count ?? 0} functions are connected through ${stats?.edge_count ?? 0} edges in the current graph.`
        : "Index a repository to populate structure, relationships, and graph-backed analysis."
    },
    {
      eyebrow: "Repository overview",
      title: selectedNode?.name || activeTab,
      body: selectedNode
        ? `${shortPath(selectedNode.file)} is open in the inspector preview.`
        : TAB_DESCRIPTIONS[activeTab]
    }
  ];

  const workspaceTab =
    workspaceView === "History" ? "History" : workspaceView === "Overview" ? "Overview" : mapTopTabToWorkspace(activeTab);
  const pageTitle =
    workspaceView === "History"
      ? "Recent analysis activity"
      : workspaceView === "Overview"
        ? "Repository analysis workspace"
        : activeTab === "Dead code"
          ? "Dead code review"
          : activeTab === "Inspect"
            ? "Impact inspection"
            : activeTab === "AI"
              ? "AI repository analysis"
              : activeTab === "Search"
                ? "Repository search"
                : "Codebase Q&A";
  const pageSubtitle =
    workspaceView === "History"
      ? "A timeline of indexing and exploration events in this frontend session."
      : workspaceView === "Overview"
        ? hasIndexedRepo
          ? "Browse the indexed repository, inspect graph coverage, and jump into code-aware workflows."
          : "Start by indexing a repository path or URL to unlock the rest of the workspace."
        : TAB_DESCRIPTIONS[activeTab];
  const breadcrumbs = ["Workspace", workspaceTab];

  function recordActivity(kind, title, detail) {
    setActivityLog((current) => [
      { id: `${Date.now()}-${Math.random().toString(16).slice(2)}`, kind, title, detail },
      ...current
    ].slice(0, 12));
  }

  async function handleIndex() {
    setLoading(true);
    setError("");
    try {
      const response = await fetch("/index", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repo_url: repoUrl })
      });
      const data = await parseJson(response);
      if (!response.ok) {
        throw new Error(readApiError(data, "Indexing failed."));
      }
      setStats(data);
      setWorkspaceView("Overview");
      resetExplorationState();
      recordActivity("index", "Repository indexed", shortRepoName(data.repo_path || repoUrl));
      await loadDeadCode(data.dead_code_count);
    } catch (err) {
      setError(err.message || "Indexing failed.");
    } finally {
      setLoading(false);
    }
  }

  async function loadDeadCode(deadCount) {
    const deadCodeResponse = await fetch("/deadcode");
    const deadCodeData = await parseJson(deadCodeResponse);
    if (deadCodeResponse.ok) {
      setDeadCodeRows(deadCodeData);
      if (typeof deadCount === "number") {
        recordActivity("dead-code", "Dead code loaded", `${deadCount} functions flagged`);
      }
    } else {
      setDeadCodeRows([]);
    }
  }

  async function handleAsk(nextQuestion = question) {
    if (!hasIndexedRepo) {
      setError("Index a repository first before using Ask.");
      return;
    }
    if (!nextQuestion.trim()) {
      return;
    }
    setLoading(true);
    setError("");
    setActiveTab("Ask");
    setWorkspaceView(null);
    setPreviewMode("nodes");
    setMessages((current) => [...current, { role: "user", content: nextQuestion }, { role: "assistant", content: "" }]);
    recordActivity("ask", "Ask query submitted", nextQuestion);
    try {
      const response = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: nextQuestion })
      });
      if (!response.ok) {
        const errorData = await parseJson(response);
        throw new Error(readApiError(errorData, "Question failed."));
      }

      const debugResponse = await fetch(`/debug/retrieval?q=${encodeURIComponent(nextQuestion)}`);
      if (debugResponse.ok) {
        setSourcesData(await debugResponse.json());
      } else {
        setSourcesData(null);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n");
        buffer = parts.pop() || "";
        for (const part of parts) {
          if (!part.trim()) {
            continue;
          }
          const payload = JSON.parse(part);
          if (payload.type === "context") {
            setCallChain(payload.call_chain || []);
            setGraphEdges(payload.edges || []);
            setGraphNodes(payload.matches || []);
          }
          if (payload.type === "token") {
            setMessages((current) => {
              const next = [...current];
              next[next.length - 1] = {
                ...next[next.length - 1],
                content: next[next.length - 1].content + payload.token
              };
              return next;
            });
          }
        }
      }
      setQuestion("");
    } catch (err) {
      setMessages((current) => current.slice(0, -1));
      setError(err.message || "Question failed.");
    } finally {
      setLoading(false);
    }
  }

  async function handleAiQuery() {
    if (!hasIndexedRepo) {
      setError("Index a repository first before using AI.");
      return;
    }
    if (!aiQuestion.trim()) {
      return;
    }
    setLoading(true);
    setError("");
    setActiveTab("AI");
    setWorkspaceView(null);
    recordActivity("ai", "AI analysis submitted", aiFilePath ? `${shortPath(aiFilePath)} - ${aiQuestion}` : aiQuestion);
    try {
      const response = await fetch("/ai/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: aiQuestion, api_key: aiApiKey, file_path: aiFilePath })
      });
      const data = await parseJson(response);
      if (!response.ok) {
        throw new Error(readApiError(data, "AI query failed."));
      }
      setAiResult(data);
    } catch (err) {
      setError(err.message || "AI query failed.");
    } finally {
      setLoading(false);
    }
  }

  async function handleSearch(nextQuery = searchQuery) {
    if (!hasIndexedRepo) {
      setError("Index a repository first before using Search.");
      return;
    }
    if (!nextQuery.trim()) {
      return;
    }
    setLoading(true);
    setError("");
    setActiveTab("Search");
    setWorkspaceView(null);
    recordActivity("search", "Repository search", nextQuery);
    try {
      const response = await fetch(`/debug/retrieval?q=${encodeURIComponent(nextQuery)}`);
      const data = await parseJson(response);
      if (!response.ok) {
        throw new Error(readApiError(data, "Search failed."));
      }
      const normalizedQuery = nextQuery.trim().toLowerCase();
      const fileMatches = indexedFiles.filter((file) => {
        const path = file.path?.toLowerCase() || "";
        const stem = shortPath(file.path || "").toLowerCase();
        return path.includes(normalizedQuery) || stem.includes(normalizedQuery);
      });
      setSearchResults({
        query: nextQuery,
        files: fileMatches.slice(0, 12),
        fused: data.fused || [],
        vector: data.vector || [],
        keyword: data.keyword || []
      });
    } catch (err) {
      setError(err.message || "Search failed.");
    } finally {
      setLoading(false);
    }
  }

  async function handleInspect(nextFunction = impactFunction || selectedNode?.name || "") {
    if (!hasIndexedRepo) {
      setError("Index a repository first before using Inspect.");
      return;
    }
    const targetFunction = nextFunction.trim();
    if (!targetFunction) {
      setError("Select a function or enter a function name before running Inspect.");
      return;
    }
    if (targetFunction.includes(".py")) {
      setError("Inspect impact expects a function name, not a file name.");
      return;
    }
    setLoading(true);
    setError("");
    setActiveTab("Inspect");
    setWorkspaceView(null);
    setImpactFunction(targetFunction);
    recordActivity("impact", "Impact analysis", targetFunction);
    try {
      const [impactResponse, nodesResponse] = await Promise.all([
        fetch(`/impact/${encodeURIComponent(targetFunction)}`),
        fetch(`/nodes/by-name/${encodeURIComponent(targetFunction)}`)
      ]);
      const data = await parseJson(impactResponse);
      if (!impactResponse.ok) {
        throw new Error(readApiError(data, "Impact lookup failed."));
      }
      const nodes = await parseJson(nodesResponse);
      if (!nodesResponse.ok) {
        throw new Error(readApiError(nodes, "Unable to resolve function for preview."));
      }
      setImpactRows(data);
      if (nodes.length) {
        await openNodeReference(nodes[0], { preserveLoading: true, silent: true });
      }
    } catch (err) {
      setError(err.message || "Impact lookup failed.");
    } finally {
      setLoading(false);
    }
  }

  async function handleNodeClick(nodeId, options = {}) {
    const { preserveLoading = false, silent = false } = options;
    if (!preserveLoading) {
      setLoading(true);
    }
    setError("");
    setPreviewMode("nodes");
    try {
      const [nodeResponse, graphResponse] = await Promise.all([
        fetch(`/node/${encodeURIComponent(nodeId)}`),
        fetch(`/graph/${encodeURIComponent(nodeId)}`)
      ]);
      const data = await parseJson(nodeResponse);
      if (!nodeResponse.ok) {
        throw new Error(readApiError(data, "Unable to load function preview."));
      }
      const graphData = await parseJson(graphResponse);
      if (!graphResponse.ok) {
        throw new Error(readApiError(graphData, "Unable to load graph context."));
      }
      setSelectedNode(data);
      setImpactFunction(data.name || "");
      setCallChain(graphData.call_chain || []);
      setGraphEdges(graphData.edges || []);
      setGraphNodes(graphData.nodes || []);
      if (!silent) {
        recordActivity("preview", "Node preview opened", `${data.name} - ${shortPath(data.file)}`);
      }
    } catch (err) {
      setError(err.message || "Unable to load function preview.");
    } finally {
      if (!preserveLoading) {
        setLoading(false);
      }
    }
  }

  async function openNodeReference(nodeRef, options = {}) {
    const directNodeId = typeof nodeRef === "string" ? nodeRef : nodeRef?.node_id;
    if (directNodeId) {
      await handleNodeClick(directNodeId, options);
      return;
    }

    const { preserveLoading = false, silent = false } = options;
    const fallbackName = typeof nodeRef === "object" ? nodeRef?.name : "";
    if (!fallbackName) {
      setError("Unable to open function preview.");
      return;
    }

    if (!preserveLoading) {
      setLoading(true);
    }
    setError("");
    setPreviewMode("nodes");

    try {
      const response = await fetch(`/nodes/by-name/${encodeURIComponent(fallbackName)}`);
      const nodes = await parseJson(response);
      if (!response.ok) {
        throw new Error(readApiError(nodes, "Unable to resolve function for preview."));
      }

      const preferredNode =
        typeof nodeRef === "object" && nodeRef?.file
          ? nodes.find((item) => item.file === nodeRef.file) || nodes[0]
          : nodes[0];

      if (!preferredNode?.node_id) {
        throw new Error("Unable to resolve function for preview.");
      }

      await handleNodeClick(preferredNode.node_id, { preserveLoading: true, silent });
    } catch (err) {
      setError(err.message || "Unable to open function preview.");
    } finally {
      if (!preserveLoading) {
        setLoading(false);
      }
    }
  }

  function handleFileSelect(filePath) {
    setAiFilePath(filePath);
    setActiveTab("AI");
    setWorkspaceView(null);
    recordActivity("file", "File selected for AI", shortPath(filePath));
  }

  async function handleCopy(text, target = "copy") {
    if (!text) {
      setError("Nothing to copy.");
      return;
    }
    try {
      if (navigator.clipboard?.writeText && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
      } else {
        fallbackCopyText(text);
      }
      setCopyTarget(target);
      window.setTimeout(() => {
        setCopyTarget((current) => (current === target ? "" : current));
      }, 1500);
    } catch {
      setError("Copy failed. Try selecting the text manually.");
    }
  }

  function handlePreviewBack() {
    setSelectedNode(null);
    setPreviewMode("nodes");
  }

  function handleTopTabChange(nextTab) {
    setActiveTab(nextTab);
    setWorkspaceView(null);
    setError("");
    if (nextTab === "Dead code" && hasIndexedRepo && !deadCodeRows.length) {
      loadDeadCode().catch(() => {});
    }
    if (nextTab === "Inspect" && selectedNode?.name && !impactFunction) {
      setImpactFunction(selectedNode.name);
    }
    if (nextTab !== "Ask") {
      setMessages([]);
      setSourcesData(null);
    }
    if (nextTab !== "AI") {
      setAiResult(null);
    }
    if (nextTab !== "Search") {
      setSearchResults(null);
    }
    if (nextTab !== "Inspect") {
      setImpactRows([]);
    }
  }

  function handleWorkspaceTabChange(nextTab) {
    setError("");
    if (nextTab === "Overview" || nextTab === "History") {
      setWorkspaceView(nextTab);
      return;
    }
    if (nextTab === "Ask") {
      handleTopTabChange("Ask");
      return;
    }
    if (nextTab === "Search") {
      handleTopTabChange("Search");
      return;
    }
    if (nextTab === "Impact") {
      handleTopTabChange("Inspect");
    }
  }

  function resetExplorationState() {
    setMessages([]);
    setCallChain([]);
    setGraphEdges([]);
    setGraphNodes([]);
    setSourcesData(null);
    setSelectedNode(null);
    setAiResult(null);
    setSearchResults(null);
    setImpactRows([]);
    setQuestion("");
    setAiQuestion("");
    setAiFilePath("");
    setSearchQuery("");
    setImpactFunction("");
    setPreviewMode("nodes");
  }

  function renderWorkspaceContent() {
    if (workspaceView === "Overview") {
      return (
        <div className={styles.contentStack}>
          <div className={styles.summaryGrid}>
            <div className={styles.card}>
              <div className={styles.cardTopRow}>
                <div>
                  <div className={styles.sectionEyebrow}>{summaryCards[0].eyebrow}</div>
                  <h2 className={styles.cardTitle}>Repository map</h2>
                </div>
                <span className={styles.metaPill}>{topFiles.length}</span>
              </div>
              <p className={styles.cardText}>{summaryCards[0].body}</p>
              {topFiles.length ? (
                <div className={styles.mapList}>
                  {topFiles.map((file, index) => (
                    <button
                      key={file.path}
                      className={styles.mapRow}
                      onClick={() => handleFileSelect(file.path)}
                    >
                      <span className={styles.mapIndex}>{String(index + 1).padStart(2, "0")}</span>
                      <div className={styles.mapTrack}>
                        <div className={styles.mapFill} style={{ width: `${Math.max(24, 100 - index * 11)}%` }} />
                      </div>
                      <span className={styles.mapLabel}>{shortPath(file.path)}</span>
                    </button>
                  ))}
                </div>
              ) : (
                <div className={styles.emptyState}>Index a repository to populate file coverage and analysis context.</div>
              )}
            </div>

            <div className={styles.card}>
              <div className={styles.cardTopRow}>
                <div>
                  <div className={styles.sectionEyebrow}>{summaryCards[1].eyebrow}</div>
                  <h2 className={styles.cardTitle}>Repository overview</h2>
                </div>
                <span className={styles.metaPill}>{activeTab}</span>
              </div>
              <p className={styles.cardText}>{summaryCards[1].body}</p>
              <div className={styles.headerStats}>
                <HeaderStat label="Functions" value={stats?.function_count ?? 0} />
                <HeaderStat label="Edges" value={stats?.edge_count ?? 0} />
                <HeaderStat label="Dead code" value={stats?.dead_code_count ?? 0} />
              </div>
              <div className={styles.recommendationPanel}>
                <div className={styles.recommendationLabel}>Current workflow</div>
                <h3 className={styles.recommendationTitle}>{TAB_DESCRIPTIONS[activeTab]}</h3>
                <p className={styles.summaryText}>
                  {activeTab === "AI"
                    ? "Select an indexed file, enter an API key, and submit a file-focused analysis request."
                    : activeTab === "Inspect"
                      ? "Open a node preview or enter a function name to trace inbound callers."
                      : activeTab === "Dead code"
                        ? "Review unreachable functions and open them in the inspector for validation."
                        : activeTab === "Search"
                          ? "Search across files, symbols, and retrieval signals without changing backend behavior."
                          : "Ask natural-language questions to stream grounded explanations from the existing /ask flow."}
                </p>
              </div>
            </div>
          </div>

          <div className={`${styles.card} ${styles.askCard}`}>
            <div className={styles.askCardHeader}>
              <div className={styles.askBadge}>ASK</div>
              <div>
                <h2 className={styles.cardTitle}>Codebase Q&amp;A</h2>
                <p className={styles.cardText}>Ask the indexed repository a question without changing the current backend ask flow.</p>
              </div>
            </div>

            <div className={styles.messageStream}>
              {messages.length ? (
                messages.map((message, index) => (
                  <div
                    key={`${message.role}-${index}`}
                    className={message.role === "user" ? styles.userMessage : styles.assistantMessage}
                  >
                    {message.role === "assistant" ? renderStructuredMessage(message.content) : message.content}
                  </div>
                ))
              ) : (
                <div className={styles.emptyState}>Ask a repository question after indexing to see streamed answers and grounded retrieval sources.</div>
              )}
            </div>

            <textarea
              className={styles.textarea}
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask how a feature works..."
            />

            <div className={styles.badgeRow}>
              {SUGGESTIONS.map((suggestion) => (
                <button
                  key={suggestion}
                  className={styles.suggestionChip}
                  onClick={() => handleAsk(suggestion)}
                  disabled={!hasIndexedRepo}
                >
                  {suggestion}
                </button>
              ))}
            </div>

            <div className={styles.actionRow}>
              <button className={styles.primaryButton} onClick={() => handleAsk()} disabled={loading || !hasIndexedRepo}>
                <span>Send</span>
                <span className={styles.buttonArrow}>→</span>
              </button>
            </div>
          </div>
        </div>
      );
    }

    if (workspaceView === "History") {
      return (
        <div className={styles.contentStack}>
          <div className={styles.cardHeader}>
            <div>
              <div className={styles.sectionEyebrow}>History</div>
              <h2 className={styles.cardTitle}>Recent analysis activity</h2>
            </div>
            <p className={styles.cardText}>This timeline reflects real interactions in the current frontend session.</p>
          </div>
          {activityLog.length ? (
            <div className={styles.activityList}>
              {activityLog.map((item) => (
                <div key={item.id} className={styles.activityItem}>
                  <span className={styles.activityKind}>{item.kind}</span>
                  <div>
                    <div className={styles.activityTitle}>{item.title}</div>
                    <div className={styles.activityDetail}>{item.detail}</div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className={styles.emptyState}>Session activity appears here once you index a repository or run analysis.</div>
          )}
        </div>
      );
    }

    if (activeTab === "Ask") {
      return (
        <div className={styles.contentStack}>
          <div className={styles.cardHeader}>
            <div>
              <div className={styles.sectionEyebrow}>Ask</div>
              <h2 className={styles.cardTitle}>Codebase Q&amp;A</h2>
            </div>
            <p className={styles.cardText}>Streaming answers remain connected to the existing Ask endpoint and retrieval debug data.</p>
          </div>

          <div className={styles.messageStream}>
            {messages.length ? (
              messages.map((message, index) => (
                <div
                  key={`${message.role}-${index}`}
                  className={message.role === "user" ? styles.userMessage : styles.assistantMessage}
                >
                  {message.role === "assistant" ? renderStructuredMessage(message.content) : message.content}
                </div>
              ))
            ) : (
              <div className={styles.emptyState}>Ask a repository question after indexing to see streamed answers and grounded retrieval sources.</div>
            )}
          </div>

          <div className={styles.badgeRow}>
            {SUGGESTIONS.map((suggestion) => (
              <button
                key={suggestion}
                className={styles.suggestionChip}
                onClick={() => handleAsk(suggestion)}
                disabled={!hasIndexedRepo}
              >
                {suggestion}
              </button>
            ))}
          </div>

          <div className={styles.formStack}>
            <textarea
              className={styles.textarea}
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask how a feature works..."
            />
            <div className={styles.actionRow}>
              <button className={styles.primaryButton} onClick={() => handleAsk()} disabled={loading || !hasIndexedRepo}>
                <span>Send</span>
                <span className={styles.buttonArrow}>→</span>
              </button>
            </div>
          </div>

          {sourcesData ? (
            <div className={styles.card}>
              <div className={styles.subCardHeader}>
                <div className={styles.sectionEyebrow}>Sources</div>
                <span className={styles.metaPill}>{sourcesData.fused?.length ?? 0} fused hits</span>
              </div>
              <div className={styles.resultList}>
                {sourcesData.fused?.slice(0, 8).map((entry) => (
                  <div key={entry.node_id} className={styles.resultRow}>
                    <button className={styles.linkButton} onClick={() => openNodeReference(entry)}>
                      {entry.name}
                    </button>
                    <div className={styles.badgeRow}>
                      {sourcesData.vector?.some((item) => item.node_id === entry.node_id) ? (
                        <span className={styles.signalBadge}>Vector</span>
                      ) : null}
                      {sourcesData.graph?.some((item) => item.node_id === entry.node_id) ? (
                        <span className={styles.signalBadge}>Graph</span>
                      ) : null}
                      {sourcesData.keyword?.some((item) => item.node_id === entry.node_id) ? (
                        <span className={styles.signalBadge}>Keyword</span>
                      ) : null}
                      <span className={styles.metaPill}>{Number(entry.score || 0).toFixed(3)}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      );
    }

    if (activeTab === "AI") {
      return (
        <div className={styles.contentStack}>
          <div className={styles.cardHeader}>
            <div>
              <div className={styles.sectionEyebrow}>AI</div>
              <h2 className={styles.cardTitle}>File-focused semantic analysis</h2>
            </div>
            <p className={styles.cardText}>Select an indexed file, enter an API key, and submit a query using the current /ai/query payload.</p>
          </div>

          <div className={styles.formStack}>
            <div className={styles.formGrid}>
              <label className={styles.fieldGroup}>
                <span className={styles.fieldLabel}>Indexed file</span>
                <select className={styles.select} value={aiFilePath} onChange={(event) => setAiFilePath(event.target.value)}>
                  <option value="">Ask across the indexed repository</option>
                  {indexedFiles.map((file) => (
                    <option key={file.path} value={file.path}>
                      {file.path}
                    </option>
                  ))}
                </select>
              </label>

              <label className={styles.fieldGroup}>
                <span className={styles.fieldLabel}>API key</span>
                <input
                  className={styles.input}
                  type="password"
                  value={aiApiKey}
                  onChange={(event) => setAiApiKey(event.target.value)}
                  placeholder="Google AI API key"
                />
              </label>
            </div>

            <label className={styles.fieldGroup}>
              <span className={styles.fieldLabel}>Question</span>
              <textarea
                className={styles.textarea}
                value={aiQuestion}
                onChange={(event) => setAiQuestion(event.target.value)}
                placeholder="Ask the AI tab about the indexed codebase..."
              />
            </label>

            <div className={styles.actionRow}>
              <button className={styles.primaryButton} onClick={handleAiQuery} disabled={loading || !hasIndexedRepo}>
                Run AI analysis
              </button>
            </div>
          </div>

          {aiResult ? (
            <div className={styles.contentStack}>
              <div className={styles.card}>
                <div className={styles.subCardHeader}>
                  <div className={styles.sectionEyebrow}>Answer</div>
                  <button className={styles.secondaryButton} onClick={() => handleCopy(aiResult.answer, "answer")}>
                    {copyTarget === "answer" ? "Copied" : "Copy"}
                  </button>
                </div>
                {renderStructuredMessage(aiResult.answer)}
              </div>

              <div className={styles.dualGrid}>
                <div className={styles.card}>
                  <div className={styles.subCardHeader}>
                    <div className={styles.sectionEyebrow}>Semantic matches</div>
                    <span className={styles.metaPill}>{aiResult.matches?.length ?? 0}</span>
                  </div>
                  {aiResult.matches?.length ? (
                    <div className={styles.edgeListCompact}>
                      {aiResult.matches.map((match) => (
                        <button key={match.node_id} className={styles.edgeRow} onClick={() => openNodeReference(match)}>
                          <span>{match.name}</span>
                          <span>{shortPath(match.file)}</span>
                        </button>
                      ))}
                    </div>
                  ) : (
                    <div className={styles.emptyState}>No semantic matches found.</div>
                  )}
                </div>

                <div className={styles.card}>
                  <div className={styles.subCardHeader}>
                    <div className={styles.sectionEyebrow}>Retrieved context</div>
                    <span className={styles.metaPill}>{aiFilePath ? shortPath(aiFilePath) : "Repo-wide"}</span>
                  </div>
                  <pre className={styles.contextBlock}>
                    <code>{aiResult.context}</code>
                  </pre>
                </div>
              </div>
            </div>
          ) : (
            <div className={styles.emptyState}>Run an AI analysis to review the answer, semantic matches, and retrieved context.</div>
          )}
        </div>
      );
    }

    if (activeTab === "Search") {
      return (
        <div className={styles.contentStack}>
          <div className={styles.cardHeader}>
            <div>
              <div className={styles.sectionEyebrow}>Search</div>
              <h2 className={styles.cardTitle}>Repository search</h2>
            </div>
            <p className={styles.cardText}>Search results combine indexed file matches with the existing retrieval debug endpoint.</p>
          </div>

          <div className={styles.searchBar}>
            <input
              className={styles.input}
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              placeholder="Search by function name, file path, or symbol"
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  handleSearch();
                }
              }}
            />
            <button
              className={styles.primaryButton}
              onClick={() => handleSearch()}
              disabled={loading || !hasIndexedRepo || !searchQuery.trim()}
            >
              Search repository
            </button>
          </div>

          {searchResults ? (
            <div className={styles.tripleGrid}>
              <div className={styles.card}>
                <div className={styles.subCardHeader}>
                  <div className={styles.sectionEyebrow}>Files</div>
                  <span className={styles.metaPill}>{searchResults.files.length}</span>
                </div>
                {searchResults.files.length ? (
                  <div className={styles.resultList}>
                    {searchResults.files.map((file) => (
                      <button key={file.path} className={styles.fileSearchRow} onClick={() => handleFileSelect(file.path)}>
                        <span>{shortPath(file.path)}</span>
                        <span>{file.path}</span>
                      </button>
                    ))}
                  </div>
                ) : (
                  <div className={styles.emptyState}>No indexed files matched this search.</div>
                )}
              </div>

              <div className={styles.card}>
                <div className={styles.subCardHeader}>
                  <div className={styles.sectionEyebrow}>Functions / symbols</div>
                  <span className={styles.metaPill}>{searchResults.fused.length}</span>
                </div>
                {searchResults.fused.length ? (
                  <div className={styles.resultList}>
                    {searchResults.fused.map((entry) => (
                      <div key={entry.node_id} className={styles.resultRow}>
                        <button className={styles.linkButton} onClick={() => openNodeReference(entry)}>
                          {entry.name}
                        </button>
                        <span className={styles.metaPill}>{Number(entry.score || 0).toFixed(3)}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className={styles.emptyState}>No symbol matches found.</div>
                )}
              </div>

              <div className={styles.card}>
                <div className={styles.subCardHeader}>
                  <div className={styles.sectionEyebrow}>Retrieval signals</div>
                  <span className={styles.metaPill}>
                    {(searchResults.vector?.length || 0) + (searchResults.keyword?.length || 0)}
                  </span>
                </div>
                {[...(searchResults.vector || []), ...(searchResults.keyword || [])].length ? (
                  <div className={styles.resultList}>
                    {[...(searchResults.vector || []), ...(searchResults.keyword || [])].slice(0, 12).map((entry, index) => (
                      <div key={`${entry.node_id}-${index}`} className={styles.resultRow}>
                        <button className={styles.linkButton} onClick={() => openNodeReference(entry)}>
                          {entry.name}
                        </button>
                        <span className={styles.metaPill}>{Number(entry.score || 0).toFixed(3)}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className={styles.emptyState}>No retrieval signals available yet.</div>
                )}
              </div>
            </div>
          ) : (
            <div className={styles.emptyState}>Search for a function, file path, or symbol after indexing the repository.</div>
          )}
        </div>
      );
    }

    if (activeTab === "Inspect") {
      return (
        <div className={styles.contentStack}>
          <div className={styles.cardHeader}>
            <div>
              <div className={styles.sectionEyebrow}>Inspect</div>
              <h2 className={styles.cardTitle}>Caller impact view</h2>
            </div>
            <p className={styles.cardText}>Run reverse-call analysis for a function and inspect the most relevant preview automatically.</p>
          </div>

          <div className={styles.searchBar}>
            <input
              className={styles.input}
              value={impactFunction}
              onChange={(event) => setImpactFunction(event.target.value)}
              placeholder={selectedNode?.name ? `Function name (${selectedNode.name})` : "Function name"}
            />
            <button className={styles.primaryButton} onClick={() => handleInspect()} disabled={loading || !hasIndexedRepo}>
              Analyze impact
            </button>
          </div>

          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Caller</th>
                  <th>File</th>
                  <th>Line</th>
                  <th>Risk</th>
                </tr>
              </thead>
              <tbody>
                {impactRows.length ? (
                  impactRows.map((row) => (
                    <tr key={row.node_id} onClick={() => openNodeReference(row)} className={styles.clickableRow}>
                      <td>{row.name}</td>
                      <td>{row.file}</td>
                      <td>{row.line}</td>
                      <td>
                        <span className={styles[`risk${row.risk}`]}>{row.risk}</span>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="4">
                      <div className={styles.emptyState}>
                        {impactFunction
                          ? `No callers were found for ${impactFunction} in the current indexed graph.`
                          : "Search for a function to see its callers and risk levels."}
                      </div>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      );
    }

    return (
      <div className={styles.contentStack}>
        <div className={styles.cardHeader}>
          <div>
            <div className={styles.sectionEyebrow}>Dead code</div>
            <h2 className={styles.cardTitle}>Unreachable functions</h2>
          </div>
          <p className={styles.cardText}>Review functions with no inbound callers and open each one in the inspector preview.</p>
        </div>

        {deadCodeRows.length ? (
          <div className={styles.edgeListCompact}>
            {deadCodeRows.map((row) => (
              <button key={row.node_id} className={styles.edgeRow} onClick={() => openNodeReference(row)}>
                <span>{row.name}</span>
                <span>
                  {shortPath(row.file)}:{row.line}
                </span>
              </button>
            ))}
          </div>
        ) : (
          <div className={styles.emptyState}>No dead code results yet.</div>
        )}
      </div>
    );
  }

  return (
    <div className={styles.page}>
      <header className={styles.topbar}>
        <div className={styles.topbarLeft}>
          <div className={styles.brandBlock}>
            <div className={styles.brandMark}>CL</div>
            <div className={styles.brandName}>CodeLens</div>
          </div>

          <nav className={styles.topTabs} aria-label="Primary">
            {TABS.map((tab) => (
              <button
                key={tab}
                className={tab === activeTab && !workspaceView ? styles.topTabActive : styles.topTab}
                onClick={() => handleTopTabChange(tab)}
              >
                {tab}
              </button>
            ))}
          </nav>
        </div>

        <div className={styles.topbarRight}>
          <button className={styles.ghostButton} type="button">
            Docs
          </button>
          <button className={styles.ghostButton} type="button">
            Settings
          </button>
          <button className={styles.avatarButton} type="button" aria-label="Workspace profile">
            CL
          </button>
        </div>
      </header>

      <div className={styles.appShell}>
        <aside className={styles.sidebar}>
          <section className={styles.sidebarSection}>
            <div className={styles.sectionHeaderStack}>
              <div className={styles.sectionEyebrow}>Repository</div>
              <h2 className={styles.sidebarTitle}>Source workspace</h2>
              <p className={styles.sidebarText}>Use a GitHub URL or local path, then index the existing repository graph.</p>
            </div>

            <div className={styles.inputShell}>
              <span className={styles.inputIcon}>⌘</span>
              <input
                className={styles.input}
                value={repoUrl}
                onChange={(event) => setRepoUrl(event.target.value)}
                placeholder="GitHub URL or local path"
              />
            </div>

            <button className={styles.primaryButton} onClick={handleIndex} disabled={loading || !repoUrl.trim()}>
              {loading ? "Indexing repository..." : "Index repository"}
            </button>
          </section>

          <section className={styles.sidebarSection}>
            <div className={styles.sectionHeader}>
              <div>
                <div className={styles.sectionEyebrow}>Analysis metrics</div>
                <h3 className={styles.sectionTitle}>Current coverage</h3>
              </div>
              <span className={styles.metaPill}>{hasIndexedRepo ? "Live" : "Idle"}</span>
            </div>

            <div className={styles.sidebarMetrics}>
              <Stat label="Functions" value={stats?.function_count ?? 0} />
              <Stat label="Edges" value={stats?.edge_count ?? 0} />
              <Stat label="Dead code" value={stats?.dead_code_count ?? 0} />
              <Stat label="Files" value={indexedFiles.length} />
              <Stat label="Last indexed" value={hasIndexedRepo ? "ready" : "pending"} />
            </div>
          </section>

          <section className={styles.sidebarSection}>
            <div className={styles.sectionHeader}>
              <div>
                <div className={styles.sectionEyebrow}>Indexed files</div>
                <h3 className={styles.sectionTitle}>Drive AI analysis</h3>
              </div>
              <span className={styles.metaPill}>{indexedFiles.length}</span>
            </div>
            {indexedFiles.length ? (
              <div className={styles.fileList}>
                {indexedFiles.map((file) => (
                  <button
                    key={file.path}
                    className={file.path === aiFilePath ? styles.fileRowActive : styles.fileRow}
                    onClick={() => handleFileSelect(file.path)}
                  >
                    <span
                      className={styles.languageDot}
                      style={{ backgroundColor: LANGUAGE_COLORS[file.language] || "#A0AEC0" }}
                    />
                    <span className={styles.fileTextWrap}>
                      <span className={styles.fileName}>{shortPath(file.path)}</span>
                      <span className={styles.filePath}>{file.path}</span>
                    </span>
                  </button>
                ))}
              </div>
            ) : (
              <div className={styles.emptySidebar}>Indexed files will appear here after a repository scan.</div>
            )}
          </section>
        </aside>

        <main className={styles.workspace}>
          <div className={styles.workspaceCanvas}>
            <section className={styles.workspaceHeader}>
              <div className={styles.breadcrumbRow}>
                {breadcrumbs.map((crumb) => (
                  <span key={crumb} className={styles.breadcrumbChip}>
                    {crumb}
                  </span>
                ))}
              </div>

              <div className={styles.workspaceHeaderRow}>
                <div>
                  <h1 className={styles.workspaceTitle}>{pageTitle}</h1>
                  <p className={styles.workspaceSubtitle}>{pageSubtitle}</p>
                </div>

                <div className={styles.workspaceActions}>
                  <span className={styles.statusInline}>
                    <span className={styles.statusDot} />
                    {hasIndexedRepo ? "Indexed workspace ready" : "No repository indexed"}
                  </span>
                  {selectedNode ? (
                    <button className={styles.secondaryButton} onClick={() => handleCopy(selectedNode.code, "code")}>
                      {copyTarget === "code" ? "Code copied" : "Copy code"}
                    </button>
                  ) : null}
                </div>
              </div>
            </section>

            <section className={styles.subnav}>
              <div className={styles.workspaceTabs} aria-label="Workspace">
                {WORKSPACE_TABS.map((tab) => (
                  <button
                    key={tab}
                    className={workspaceTab === tab ? styles.workspaceTabActive : styles.workspaceTab}
                    onClick={() => handleWorkspaceTabChange(tab)}
                  >
                    {tab}
                  </button>
                ))}
              </div>
            </section>

            {error ? <div className={styles.errorBanner}>{error}</div> : null}

            <section className={styles.mainPanel}>{renderWorkspaceContent()}</section>
          </div>
        </main>

        <aside className={styles.previewPanel}>
          <div className={styles.previewHeader}>
            <div className={styles.sectionEyebrow}>Preview</div>
            <h2 className={styles.previewTitle}>Code inspector</h2>
            <p className={styles.previewText}>Node preview and graph context stay wired to the existing backend endpoints.</p>
          </div>

          <div className={styles.previewMetrics}>
            <MetricCard label="Nodes" value={callChain.length} />
            <MetricCard label="Edges" value={graphEdges.length} />
          </div>

          <div className={styles.previewModes}>
            <button
              className={previewMode === "nodes" ? styles.previewModeActive : styles.previewMode}
              onClick={() => setPreviewMode("nodes")}
            >
              Nodes
            </button>
            <button
              className={previewMode === "edges" ? styles.previewModeActive : styles.previewMode}
              onClick={() => setPreviewMode("edges")}
            >
              Edges
            </button>
          </div>

          <div className={styles.previewSurface}>
            {previewMode === "nodes" ? (
              selectedNode ? (
                <div className={styles.previewContent}>
                  <div className={styles.previewMeta}>
                    <div className={styles.breadcrumbs}>
                      {previewBreadcrumbs.map((crumb, index) => (
                        <span key={`${crumb}-${index}`} className={styles.breadcrumb}>
                          {crumb}
                        </span>
                      ))}
                    </div>
                    <h3 className={styles.previewNodeName}>{selectedNode.name}</h3>
                    <div className={styles.badgeRow}>
                      {nodeBadges.map((badge) => (
                        <span key={badge} className={styles.metaPill}>
                          {badge}
                        </span>
                      ))}
                    </div>
                    <div className={styles.previewActions}>
                      <button className={styles.secondaryButton} onClick={handlePreviewBack}>
                        Back
                      </button>
                      <button className={styles.secondaryButton} onClick={() => handleInspect(selectedNode.name)}>
                        Inspect impact
                      </button>
                    </div>
                  </div>
                  <pre className={styles.codeBlock}>
                    <code>{selectedNode.code}</code>
                  </pre>
                </div>
              ) : previewCandidates.length ? (
                <div className={styles.edgeList}>
                  {previewCandidates.slice(0, 12).map((node) => (
                    <button key={node.node_id} className={styles.edgeRow} onClick={() => openNodeReference(node)}>
                      <span>{node.name}</span>
                      <span>{shortPath(node.file)}</span>
                    </button>
                  ))}
                </div>
              ) : (
                <div className={styles.previewEmptyWrap}>
                  <div className={styles.previewEmptyIcon}>{"</>"}</div>
                  <div className={styles.previewEmptyTitle}>Nothing selected yet</div>
                  <div className={styles.previewEmpty}>Open results from Ask, AI, Search, Inspect, or Dead code to inspect functions here.</div>
                </div>
              )
            ) : graphEdges.length ? (
              <div className={styles.edgeList}>
                {graphEdges.map((edge, index) => (
                  <button
                    key={`${edge.source}-${edge.target}-${index}`}
                    className={styles.edgeRow}
                    onClick={() => openNodeReference(edge.target)}
                  >
                    <span>{shortName(edge.source)}</span>
                    <span className={styles.edgeArrow}>→</span>
                    <span>{shortName(edge.target)}</span>
                  </button>
                ))}
              </div>
            ) : (
              <div className={styles.previewEmptyWrap}>
                <div className={styles.previewEmptyIcon}>◎</div>
                <div className={styles.previewEmptyTitle}>No graph edges yet</div>
                <div className={styles.previewEmpty}>Run Ask, Search, AI, or Inspect to populate edge relationships for preview.</div>
              </div>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}

function renderStructuredMessage(content) {
  const blocks = content.split("\n\n").map((block) => block.trim()).filter(Boolean);
  let listKey = 0;

  return (
    <div className={styles.messageContent}>
      {blocks.map((block, index) => {
        if (block === "---") {
          return <hr key={`rule-${index}`} className={styles.messageRule} />;
        }
        if (block.startsWith("## ")) {
          return (
            <div key={`heading-${index}`} className={styles.messageHeading}>
              {block.slice(3)}
            </div>
          );
        }
        if (block.startsWith("_") && block.endsWith("_")) {
          return (
            <div key={`footer-${index}`} className={styles.messageFooter}>
              {block.slice(1, -1)}
            </div>
          );
        }
        const lines = block.split("\n").filter(Boolean);
        if (lines.every((line) => line.startsWith("- "))) {
          return (
            <ul key={`list-${index}`} className={styles.messageList}>
              {lines.map((line) => {
                listKey += 1;
                return <li key={`item-${listKey}`}>{line.slice(2)}</li>;
              })}
            </ul>
          );
        }
        return (
          <p key={`paragraph-${index}`} className={styles.messageParagraph}>
            {block}
          </p>
        );
      })}
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className={styles.statCard}>
      <span className={styles.statLabel}>{label}</span>
      <strong className={styles.statValue}>{value}</strong>
    </div>
  );
}

function HeaderStat({ label, value }) {
  return (
    <div className={styles.headerStat}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function MetricCard({ label, value }) {
  return (
    <div className={styles.metricCard}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function mapTopTabToWorkspace(tab) {
  if (tab === "Inspect") {
    return "Impact";
  }
  return tab;
}

function shortName(nodeId) {
  const parts = nodeId.split("::");
  return parts[parts.length - 1];
}

function shortPath(path) {
  if (!path) {
    return "Unknown file";
  }
  const normalized = path.replace(/\\/g, "/");
  const parts = normalized.split("/");
  return parts.slice(-2).join("/");
}

function shortRepoName(repoPath) {
  if (!repoPath) {
    return "Unknown repository";
  }
  const normalized = repoPath.replace(/\\/g, "/").replace(/\/$/, "");
  const parts = normalized.split("/");
  return parts[parts.length - 1] || normalized;
}

function inferLanguage(filePath) {
  if (!filePath) {
    return { label: "Source" };
  }
  const extension = filePath.split(".").pop()?.toLowerCase();
  switch (extension) {
    case "py":
      return { label: "Python" };
    case "js":
      return { label: "JavaScript" };
    case "ts":
      return { label: "TypeScript" };
    case "go":
      return { label: "Go" };
    case "java":
      return { label: "Java" };
    default:
      return { label: "Source" };
  }
}

async function parseJson(response) {
  try {
    return await response.json();
  } catch {
    return {};
  }
}

function readApiError(payload, fallback) {
  if (payload?.detail) {
    return payload.detail;
  }
  if (typeof payload === "string" && payload.trim()) {
    return payload;
  }
  return fallback;
}

function fallbackCopyText(text) {
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  textarea.setSelectionRange(0, textarea.value.length);
  const successful = document.execCommand("copy");
  document.body.removeChild(textarea);
  if (!successful) {
    throw new Error("Copy command failed.");
  }
}
