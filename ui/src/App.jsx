
import { useMemo, useState } from "react";
import styles from "./App.module.css";

const TABS = ["Ask", "AI", "Search", "Inspect", "Dead code"];
const SUGGESTIONS = [
  "How does authentication work?",
  "What is the main request flow?",
  "Which functions are related to payment processing?"
];
const QUICK_ACTIONS = [
  { label: "Start with Ask", tab: "Ask" },
  { label: "Open AI", tab: "AI" },
  { label: "Open Search", tab: "Search" },
  { label: "Inspect impact", tab: "Inspect" },
  { label: "Open dead code", tab: "Dead code" }
];
const LANGUAGE_COLORS = {
  python: "#4B8BBE",
  javascript: "#D4A200",
  typescript: "#3178C6",
  go: "#00ADD8",
  java: "#E76F51"
};
const TAB_DESCRIPTIONS = {
  Ask: "Architect-style answers grounded in retrieved code, graph expansion, and verified structure.",
  AI: "A separate semantic-search path that uses embeddings to retrieve the most relevant code before answering.",
  Search: "Simple repo search for function names, files, and retrieved symbols without changing the Ask flow.",
  Inspect: "Reverse-call analysis to see which callers depend on a function and where change risk may surface.",
  "Dead code": "A scan of functions that currently have no inbound callers in the indexed graph."
};

export default function App() {
  const [repoUrl, setRepoUrl] = useState("");
  const [stats, setStats] = useState(null);
  const [activeTab, setActiveTab] = useState("Search");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [callChain, setCallChain] = useState([]);
  const [graphEdges, setGraphEdges] = useState([]);
  const [graphNodes, setGraphNodes] = useState([]);
  const [agentQuery, setAgentQuery] = useState("");
  const [agentResult, setAgentResult] = useState(null);
  const [aiQuestion, setAiQuestion] = useState("");
  const [aiApiKey, setAiApiKey] = useState("");
  const [aiFilePath, setAiFilePath] = useState("");
  const [aiResult, setAiResult] = useState(null);
  const [issueText, setIssueText] = useState("");
  const [prDiff, setPrDiff] = useState("");
  const [githubIntel, setGithubIntel] = useState(null);
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

  const hasIndexedRepo = Boolean(stats?.repo_path);
  const indexedFiles = stats?.files || [];
  const topFiles = indexedFiles.slice(0, 6);
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
    impactRows.forEach((row) => pushCandidate({ node_id: row.node_id, name: row.name, file: row.file, start_line: row.line }));
    deadCodeRows.forEach((row) => pushCandidate({ node_id: row.node_id, name: row.name, file: row.file, start_line: row.line }));
    return candidates;
  }, [aiResult, deadCodeRows, graphNodes, impactRows, sourcesData]);
  const previewLanguage = selectedNode ? inferLanguage(selectedNode.file) : null;
  const nodeBadges = selectedNode
    ? [
        previewLanguage?.label || "Code",
        `Line ${selectedNode.start_line}`,
        `${selectedNode.calls?.length || 0} calls`
      ]
    : [];
  const previewBreadcrumbs = selectedNode ? selectedNode.file.replace(/\\/g, "/").split("/") : [];

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
      resetExplorationState();

      await loadDeadCode();
    } catch (err) {
      setError(err.message || "Indexing failed.");
    } finally {
      setLoading(false);
    }
  }

  async function loadDeadCode() {
    const deadCodeResponse = await fetch("/deadcode");
    const deadCodeData = await parseJson(deadCodeResponse);
    if (deadCodeResponse.ok) {
      setDeadCodeRows(deadCodeData);
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
    setPreviewMode("nodes");
    setMessages((current) => [...current, { role: "user", content: nextQuestion }, { role: "assistant", content: "" }]);
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

  async function handleAgentQuery() {
    if (!hasIndexedRepo) {
      setError("Index a repository first before using Agent.");
      return;
    }
    if (!agentQuery.trim()) {
      return;
    }
    setLoading(true);
    setError("");
    try {
      const response = await fetch("/agent/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: agentQuery })
      });
      const data = await parseJson(response);
      if (!response.ok) {
        throw new Error(readApiError(data, "Agent query failed."));
      }
      setAgentResult(data);
    } catch (err) {
      setError(err.message || "Agent query failed.");
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
        keyword: data.keyword || [],
      });
    } catch (err) {
      setError(err.message || "Search failed.");
    } finally {
      setLoading(false);
    }
  }

  async function handleGitHubIntelligence() {
    if (!hasIndexedRepo) {
      setError("Index a repository first before using GitHub analysis.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const response = await fetch("/intelligence/github", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ issue_text: issueText, pr_diff: prDiff })
      });
      const data = await parseJson(response);
      if (!response.ok) {
        throw new Error(readApiError(data, "GitHub intelligence failed."));
      }
      setGithubIntel(data);
    } catch (err) {
      setError(err.message || "GitHub intelligence failed.");
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
    setImpactFunction(targetFunction);
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
        await handleNodeClick(nodes[0].node_id);
      }
    } catch (err) {
      setError(err.message || "Impact lookup failed.");
    } finally {
      setLoading(false);
    }
  }

  async function handleNodeClick(nodeId) {
    setLoading(true);
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
    } catch (err) {
      setError(err.message || "Unable to load function preview.");
    } finally {
      setLoading(false);
    }
  }
  async function handleFileJump(filePath) {
    const candidate =
      githubIntel?.affected_functions?.find((fn) => fn.file === filePath)?.node_id ||
      agentResult?.tool_results?.search_functions?.find((fn) => fn.file === filePath)?.node_id;
    if (!candidate) {
      setError(`No function preview is available yet for ${filePath}.`);
      return;
    }
    await handleNodeClick(candidate);
  }

  async function handleCopy(text) {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      setError("Copy failed. Try selecting the text manually.");
    }
  }

  function handlePreviewBack() {
    setSelectedNode(null);
    setPreviewMode("nodes");
  }

  function handleTabChange(nextTab) {
    setActiveTab(nextTab);
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
    if (nextTab !== "GitHub") {
      setGithubIntel(null);
    }
    if (nextTab !== "Inspect") {
      setImpactRows([]);
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
    setAgentResult(null);
    setGithubIntel(null);
    setImpactRows([]);
    setQuestion("");
    setAiQuestion("");
    setAiFilePath("");
    setSearchQuery("");
    setAgentQuery("");
    setIssueText("");
    setPrDiff("");
    setPreviewMode("nodes");
  }

  return (
    <div className={styles.appShell}>
      <aside className={styles.sidebar}>
        <div className={styles.logoWrap}>
          <h1 className={styles.logo}>CodeLens</h1>
        </div>

        <div className={styles.sidebarSection}>
          <label className={styles.label}>Repository</label>
          <input
            className={styles.input}
            value={repoUrl}
            onChange={(event) => setRepoUrl(event.target.value)}
            placeholder="GitHub URL or local path"
          />
          <button className={styles.primaryButton} onClick={handleIndex} disabled={loading || !repoUrl.trim()}>
            {loading ? "Indexing..." : "Index"}
          </button>
          <p className={styles.helperText}>
            Index a repo first, then use Ask, AI, Search, Inspect, or Dead code against that indexed codebase.
          </p>
        </div>

        <div className={styles.sidebarStats}>
          <Stat label="FUNCTIONS" value={stats?.function_count ?? 0} />
          <Stat label="EDGES" value={stats?.edge_count ?? 0} />
          <Stat label="DEAD CODE" value={stats?.dead_code_count ?? 0} />
        </div>

        <div className={styles.indexedSection}>
          <div className={styles.label}>Indexed files</div>
          {indexedFiles.length ? (
            <div className={styles.fileList}>
              {indexedFiles.map((file) => (
                <div key={file.path} className={styles.fileRow}>
                  <span
                    className={styles.languageDot}
                    style={{ backgroundColor: LANGUAGE_COLORS[file.language] || "#9CA3AF" }}
                  />
                  <span className={styles.filePath}>{file.path}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className={styles.emptySidebar}>No indexed files yet.</div>
          )}
        </div>
      </aside>

      <main className={styles.mainColumn}>
        <header className={styles.header}>
          <div>
            <div className={styles.headerLabel}>ACTIVE WORKSPACE</div>
            <h2 className={styles.headerTitle}>
              {stats?.repo_path ? shortRepoName(stats.repo_path) : "Ready to index a repository"}
            </h2>
            <p className={styles.headerSubtitle}>A visual map of the currently retrieved execution path.</p>
          </div>
          <div className={styles.headerStats}>
            <HeaderStat label="FUNCTIONS" value={stats?.function_count ?? 0} />
            <HeaderStat label="EDGES" value={stats?.edge_count ?? 0} />
            <HeaderStat label="DEAD" value={stats?.dead_code_count ?? 0} />
          </div>
        </header>

        <div className={styles.tabBar}>
          {TABS.map((tab) => (
            <button
              key={tab}
              className={tab === activeTab ? styles.activeTab : styles.tab}
              onClick={() => handleTabChange(tab)}
            >
              {tab}
            </button>
          ))}
        </div>

        <div className={styles.commandBar}>
          <div className={styles.commandTitle}>Quick actions</div>
          <div className={styles.commandActions}>
            {QUICK_ACTIONS.map((action) => (
              <button
                key={action.label}
                className={styles.commandButton}
                onClick={() => handleTabChange(action.tab)}
                disabled={!hasIndexedRepo}
              >
                {action.label}
              </button>
            ))}
          </div>
        </div>

        <div className={styles.mainGrid}>
          <section className={styles.contentArea}>
            <div className={styles.overviewPanel}>
              <div>
                <div className={styles.label}>Repository overview</div>
                <h3 className={styles.overviewTitle}>
                  {stats?.repo_path ? shortRepoName(stats.repo_path) : "No repository indexed yet"}
                </h3>
                <p className={styles.overviewText}>
                  {TAB_DESCRIPTIONS[activeTab]}
                </p>
              </div>
              <div className={styles.overviewMap}>
                <div className={styles.overviewMapTitle}>Mini repository map</div>
                {topFiles.length ? topFiles.map((file, index) => (
                  <div key={file.path} className={styles.mapRow}>
                    <span className={styles.mapIndex}>{String(index + 1).padStart(2, "0")}</span>
                    <div className={styles.mapTrack}>
                      <div className={styles.mapFill} style={{ width: `${Math.max(24, 100 - index * 11)}%` }} />
                    </div>
                    <span className={styles.mapLabel}>{shortPath(file.path)}</span>
                  </div>
                )) : (
                  <div className={styles.emptyMini}>Index a repository to populate the overview map.</div>
                )}
              </div>
            </div>

            {activeTab === "Ask" && (
              <div className={styles.card}>
                <div className={styles.cardHeaderBlock}>
                  <div>
                    <div className={styles.label}>ASK</div>
                    <h3 className={styles.cardTitle}>Architect analysis</h3>
                  </div>
                  <p className={styles.cardDescription}>
                    Grounded answers with retrieval evidence, graph expansion, and static verification.
                  </p>
                </div>

                <div className={styles.messages}>
                  {messages.length ? messages.map((message, index) => (
                    <div
                      key={`${message.role}-${index}`}
                      className={message.role === "user" ? styles.userMessage : styles.assistantMessage}
                    >
                      {message.role === "assistant" ? renderStructuredMessage(message.content) : message.content}
                    </div>
                  )) : (
                    <div className={styles.emptyState}>Ask a question after indexing to explore the repo and see fused retrieval sources.</div>
                  )}
                </div>

                <div className={styles.pills}>
                  {SUGGESTIONS.map((suggestion) => (
                    <button
                      key={suggestion}
                      className={styles.pill}
                      onClick={() => handleAsk(suggestion)}
                      disabled={!hasIndexedRepo}
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>

                <div className={styles.composeBar}>
                  <textarea
                    className={styles.textarea}
                    value={question}
                    onChange={(event) => setQuestion(event.target.value)}
                    placeholder="Ask how a feature works..."
                  />
                  <button className={styles.primaryButton} onClick={() => handleAsk()} disabled={loading || !hasIndexedRepo}>
                    Send
                  </button>
                </div>

                {sourcesData && (
                  <div className={styles.sourcesPanel}>
                    <div className={styles.subpanelHeader}>
                      <div className={styles.label}>Sources</div>
                      <span className={styles.metaPill}>{sourcesData.fused?.length ?? 0} fused hits</span>
                    </div>
                    {sourcesData.fused?.slice(0, 8).map((entry) => (
                      <div key={entry.node_id} className={styles.sourceRow}>
                        <button className={styles.sourceButton} onClick={() => handleNodeClick(entry.node_id)}>
                          {entry.name}
                        </button>
                        <div className={styles.sourceMeta}>
                          {sourcesData.vector?.some((item) => item.node_id === entry.node_id) && (
                            <span className={styles.vectorBadge}>vector</span>
                          )}
                          {sourcesData.graph?.some((item) => item.node_id === entry.node_id) && (
                            <span className={styles.graphBadge}>graph</span>
                          )}
                          {sourcesData.keyword?.some((item) => item.node_id === entry.node_id) && (
                            <span className={styles.keywordBadge}>keyword</span>
                          )}
                          <span className={styles.scoreBadge}>{Number(entry.score).toFixed(3)}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
            {activeTab === "AI" && (
              <div className={styles.card}>
                <div className={styles.cardHeaderBlock}>
                  <div>
                    <div className={styles.label}>AI</div>
                    <h3 className={styles.cardTitle}>FAISS semantic analysis</h3>
                  </div>
                  <p className={styles.cardDescription}>
                    A separate AI path that can search the indexed repository semantically or read one selected file and ask Gemini about that file only.
                  </p>
                </div>
                <div className={styles.composeBar}>
                  <select
                    className={styles.input}
                    value={aiFilePath}
                    onChange={(event) => setAiFilePath(event.target.value)}
                  >
                    <option value="">Ask across the indexed repository</option>
                    {indexedFiles.map((file) => (
                      <option key={file.path} value={file.path}>
                        {file.path}
                      </option>
                    ))}
                  </select>
                  <input
                    className={styles.input}
                    type="password"
                    value={aiApiKey}
                    onChange={(event) => setAiApiKey(event.target.value)}
                    placeholder="Google AI API key"
                  />
                  <textarea
                    className={styles.textarea}
                    value={aiQuestion}
                    onChange={(event) => setAiQuestion(event.target.value)}
                    placeholder="Ask the AI tab about the indexed codebase..."
                  />
                  <button className={styles.primaryButton} onClick={handleAiQuery} disabled={loading || !hasIndexedRepo}>
                    Run AI
                  </button>
                </div>
                {aiResult ? (
                  <div className={styles.resultStack}>
                    <div className={styles.subCard}>
                      <div className={styles.subpanelHeader}>
                        <div className={styles.label}>Answer</div>
                        <button className={styles.secondaryButton} onClick={() => handleCopy(aiResult.answer)}>Copy</button>
                      </div>
                      {renderStructuredMessage(aiResult.answer)}
                    </div>
                    <div className={styles.subCard}>
                      <div className={styles.label}>Semantic matches</div>
                      {aiResult.matches?.length ? aiResult.matches.map((match) => (
                        <button key={match.node_id} className={styles.deadCodeItem} onClick={() => handleNodeClick(match.node_id)}>
                          <span>{match.name}</span>
                          <span>{match.file}:{match.start_line}</span>
                        </button>
                      )) : <div className={styles.emptyMini}>No semantic matches found.</div>}
                    </div>
                    <div className={styles.subCard}>
                      <div className={styles.label}>Retrieved context</div>
                      <pre className={styles.codeBlock}><code>{aiResult.context}</code></pre>
                    </div>
                  </div>
                ) : (
                  <div className={styles.emptyState}>Run the AI tab to use FAISS-based semantic retrieval against the indexed repository.</div>
                )}
              </div>
            )}
            {activeTab === "Search" && (
              <div className={styles.card}>
                <div className={styles.cardHeaderBlock}>
                  <div>
                    <div className={styles.label}>SEARCH</div>
                    <h3 className={styles.cardTitle}>Repository search</h3>
                  </div>
                  <p className={styles.cardDescription}>
                    Search indexed files and retrieved symbols without affecting Ask, AI, Inspect, or Dead code.
                  </p>
                </div>
                <div className={styles.inlineForm}>
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
                  <button className={styles.primaryButton} onClick={() => handleSearch()} disabled={loading || !hasIndexedRepo || !searchQuery.trim()}>
                    Search
                  </button>
                </div>
                {searchResults ? (
                  <div className={styles.resultStack}>
                    <div className={styles.subCard}>
                      <div className={styles.subpanelHeader}>
                        <div className={styles.label}>Files</div>
                        <span className={styles.metaPill}>{searchResults.files.length}</span>
                      </div>
                      {searchResults.files.length ? searchResults.files.map((file) => (
                        <div key={file.path} className={styles.listRow}>
                          <span>{shortPath(file.path)}</span>
                          <span className={styles.mutedText}>{file.path}</span>
                        </div>
                      )) : <div className={styles.emptyMini}>No indexed files matched this search.</div>}
                    </div>
                    <div className={styles.subCard}>
                      <div className={styles.subpanelHeader}>
                        <div className={styles.label}>Functions / Symbols</div>
                        <span className={styles.metaPill}>{searchResults.fused.length}</span>
                      </div>
                      {searchResults.fused.length ? searchResults.fused.map((entry) => (
                        <div key={entry.node_id} className={styles.sourceRow}>
                          <button className={styles.sourceButton} onClick={() => handleNodeClick(entry.node_id)}>
                            {entry.name}
                          </button>
                          <span className={styles.scoreBadge}>{Number(entry.score || 0).toFixed(3)}</span>
                        </div>
                      )) : <div className={styles.emptyMini}>No symbol matches found.</div>}
                    </div>
                    <div className={styles.subCard}>
                      <div className={styles.subpanelHeader}>
                        <div className={styles.label}>Retrieval signals</div>
                        <span className={styles.metaPill}>
                          {(searchResults.vector?.length || 0) + (searchResults.keyword?.length || 0)}
                        </span>
                      </div>
                      {[...(searchResults.vector || []), ...(searchResults.keyword || [])].length ? (
                        [...(searchResults.vector || []), ...(searchResults.keyword || [])].slice(0, 12).map((entry, index) => (
                          <div key={`${entry.node_id}-${index}`} className={styles.listRow}>
                            <button className={styles.sourceButton} onClick={() => handleNodeClick(entry.node_id)}>
                              {entry.name}
                            </button>
                            <span className={styles.scoreBadge}>{Number(entry.score || 0).toFixed(3)}</span>
                          </div>
                        ))
                      ) : (
                        <div className={styles.emptyMini}>No retrieval signals available yet.</div>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className={styles.emptyState}>Search for a function name, file path, or symbol after indexing the repository.</div>
                )}
              </div>
            )}
            {activeTab === "GitHub" && (
              <div className={styles.card}>
                <div className={styles.cardHeaderBlock}>
                  <div>
                    <div className={styles.label}>GITHUB</div>
                    <h3 className={styles.cardTitle}>PR / issue intelligence</h3>
                  </div>
                  <p className={styles.cardDescription}>
                    Paste the issue summary plus a real unified diff. The analysis is most accurate when the diff includes file paths and hunk ranges.
                  </p>
                </div>
                <div className={styles.composeBar}>
                  <label className={styles.fieldLabel}>Issue summary</label>
                  <textarea
                    className={styles.textarea}
                    value={issueText}
                    onChange={(event) => setIssueText(event.target.value)}
                    placeholder="Describe the bug, regression, or feature request here"
                  />
                  <label className={styles.fieldLabel}>PR diff</label>
                  <textarea
                    className={styles.textarea}
                    value={prDiff}
                    onChange={(event) => setPrDiff(event.target.value)}
                    placeholder="Paste the unified diff here: diff --git ... --- ... +++ ... @@ ..."
                  />
                  <button className={styles.primaryButton} onClick={handleGitHubIntelligence} disabled={loading || !hasIndexedRepo}>
                    Analyze PR / Issue
                  </button>
                </div>
                {githubIntel ? (
                  <div className={styles.resultStack}>
                    <div className={styles.subCard}>
                      <div className={styles.subpanelHeader}>
                        <div className={styles.label}>Summary</div>
                        <button className={styles.secondaryButton} onClick={() => handleCopy(githubIntel.summary)}>Copy</button>
                      </div>
                      {renderStructuredMessage(githubIntel.summary)}
                      <div className={styles.inlineChips}>
                        <span className={styles.metaPill}>Risk: {githubIntel.risk_level}</span>
                        <span className={styles.metaPill}>Files: {githubIntel.impacted_files?.length ?? 0}</span>
                        <span className={styles.metaPill}>Functions: {githubIntel.affected_functions?.length ?? 0}</span>
                      </div>
                    </div>
                    <div className={styles.subCard}>
                      <div className={styles.label}>Impacted files</div>
                      {githubIntel.impacted_files?.length ? githubIntel.impacted_files.map((file) => (
                        <button key={file} className={styles.fileJumpButton} onClick={() => handleFileJump(file)}>
                          {file}
                        </button>
                      )) : <div className={styles.emptyMini}>No impacted files detected.</div>}
                    </div>
                    <div className={styles.subCard}>
                      <div className={styles.label}>Affected functions</div>
                      {githubIntel.affected_functions?.length ? githubIntel.affected_functions.map((fn) => (
                        <button key={fn.node_id} className={styles.deadCodeItem} onClick={() => handleNodeClick(fn.node_id)}>
                          <span>{fn.name}</span>
                          <span>{fn.file}:{fn.start_line}</span>
                          <span className={styles.mutedText}>{fn.reason}</span>
                          <span className={styles.scoreBadge}>Match: {fn.match_score ?? 0}</span>
                        </button>
                      )) : <div className={styles.emptyMini}>No function-level hits yet.</div>}
                    </div>
                  </div>
                ) : (
                  <div className={styles.emptyState}>Paste an issue description and a PR diff after indexing the repo to get impacted files, affected functions, risk, and a short summary.</div>
                )}
              </div>
            )}

            {activeTab === "Inspect" && (
              <div className={styles.card}>
                <div className={styles.cardHeaderBlock}>
                  <div>
                    <div className={styles.label}>INSPECT</div>
                    <h3 className={styles.cardTitle}>Caller impact view</h3>
                  </div>
                  <p className={styles.cardDescription}>
                    Reverse-call analysis to see what could break when a function changes.
                  </p>
                </div>
                <div className={styles.inlineForm}>
                  <input
                    className={styles.input}
                    value={impactFunction}
                    onChange={(event) => setImpactFunction(event.target.value)}
                    placeholder={selectedNode?.name ? `Function name (${selectedNode.name})` : "Function name"}
                  />
                  <button className={styles.primaryButton} onClick={() => handleInspect()} disabled={loading || !hasIndexedRepo}>
                    Analyze
                  </button>
                </div>
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
                    {impactRows.length ? impactRows.map((row) => (
                      <tr key={row.node_id} onClick={() => handleNodeClick(row.node_id)} className={styles.clickableRow}>
                        <td>{row.name}</td>
                        <td>{row.file}</td>
                        <td>{row.line}</td>
                        <td><span className={styles[`risk${row.risk}`]}>{row.risk}</span></td>
                      </tr>
                    )) : (
                      <tr>
                        <td colSpan="4">
                          <div className={styles.emptyMini}>
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
            )}

            {activeTab === "Dead code" && (
              <div className={styles.card}>
                <div className={styles.cardHeaderBlock}>
                  <div>
                    <div className={styles.label}>DEAD CODE</div>
                    <h3 className={styles.cardTitle}>Unreachable functions</h3>
                  </div>
                  <p className={styles.cardDescription}>
                    These functions are currently defined without inbound callers in the indexed graph.
                  </p>
                </div>
                {deadCodeRows.length ? deadCodeRows.map((row) => (
                  <button key={row.node_id} className={styles.deadCodeItem} onClick={() => handleNodeClick(row.node_id)}>
                    <span>{row.name}</span>
                    <span>{row.file}:{row.line}</span>
                  </button>
                )) : <div className={styles.emptyState}>No dead code results yet.</div>}
              </div>
            )}
            {error && <div className={styles.errorBanner}>{error}</div>}
          </section>

          <aside className={styles.previewPanel}>
            <div className={styles.previewHeader}>
              <div>
                <div className={styles.label}>Code preview</div>
                <p className={styles.previewHint}>Jump here from Ask, AI, Inspect, or Dead code.</p>
              </div>
              <div className={styles.previewToggles}>
                {selectedNode ? (
                  <>
                    <button className={styles.secondaryButton} onClick={handlePreviewBack}>
                      Back
                    </button>
                    <button className={styles.secondaryButton} onClick={() => handleCopy(selectedNode.code)}>
                      Copy code
                    </button>
                    <button className={styles.secondaryButton} onClick={() => handleInspect(selectedNode.name)}>
                      Inspect impact
                    </button>
                  </>
                ) : null}
                <button
                  className={previewMode === "nodes" ? styles.activeToggle : styles.toggle}
                  onClick={() => setPreviewMode("nodes")}
                >
                  Nodes {callChain.length}
                </button>
                <button
                  className={previewMode === "edges" ? styles.activeToggle : styles.toggle}
                  onClick={() => setPreviewMode("edges")}
                >
                  Edges {graphEdges.length}
                </button>
              </div>
            </div>

            <div className={styles.previewBody}>
              {previewMode === "nodes" ? (
                selectedNode ? (
                  <>
                    <div className={styles.previewTitleWrap}>
                      <div className={styles.breadcrumbs}>
                        {previewBreadcrumbs.map((crumb, index) => (
                          <span key={`${crumb}-${index}`} className={styles.breadcrumb}>
                            {crumb}
                          </span>
                        ))}
                      </div>
                      <div className={styles.previewTitleRow}>
                        <h3 className={styles.previewName}>{selectedNode.name}</h3>
                      </div>
                      <div className={styles.previewBadges}>
                        {nodeBadges.map((badge) => <span key={badge} className={styles.symbolBadge}>{badge}</span>)}
                      </div>
                    </div>
                    <pre className={styles.codeBlock}><code>{selectedNode.code}</code></pre>
                  </>
                ) : (
                  previewCandidates.length ? (
                    <div className={styles.edgeList}>
                      {previewCandidates.slice(0, 12).map((node) => (
                        <button key={node.node_id} className={styles.edgeRow} onClick={() => handleNodeClick(node.node_id)}>
                          <span>{node.name}</span>
                          <span>{shortPath(node.file)}</span>
                        </button>
                      ))}
                    </div>
                  ) : (
                    <div className={styles.previewEmpty}>Click a function in Ask, AI, Inspect, or Dead code to inspect it.</div>
                  )
                )
              ) : graphEdges.length ? (
                <div className={styles.edgeList}>
                  {graphEdges.map((edge, index) => (
                    <button key={`${edge.source}-${edge.target}-${index}`} className={styles.edgeRow} onClick={() => handleNodeClick(edge.target)}>
                      <span>{shortName(edge.source)}</span>
                      <span className={styles.edgeArrow}>?</span>
                      <span>{shortName(edge.target)}</span>
                    </button>
                  ))}
                </div>
              ) : (
                <div className={styles.previewEmpty}>No graph edges yet. Ask a question to populate execution relationships.</div>
              )}
            </div>
          </aside>
        </div>
      </main>
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
          return <div key={`heading-${index}`} className={styles.messageHeading}>{block.slice(3)}</div>;
        }
        if (block.startsWith("_") && block.endsWith("_")) {
          return <div key={`footer-${index}`} className={styles.messageFooter}>{block.slice(1, -1)}</div>;
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
        return <p key={`paragraph-${index}`} className={styles.messageParagraph}>{block}</p>;
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

function buildTree(callChain, edges) {
  if (!callChain.length) {
    return null;
  }
  const nodes = new Map(callChain.map((nodeId) => [nodeId, { name: nodeId, attributes: { label: shortName(nodeId) }, children: [] }]));
  const children = new Set();

  edges.forEach((edge) => {
    const source = nodes.get(edge.source);
    const target = nodes.get(edge.target);
    if (source && target) {
      source.children.push(target);
      children.add(edge.target);
    }
  });

  const rootId = callChain.find((nodeId) => !children.has(nodeId)) || callChain[0];
  return nodes.get(rootId);
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
