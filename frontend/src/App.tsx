import { useCallback, useEffect, useRef, useState } from 'react';
import type { ChangeEvent, DragEvent, KeyboardEvent, PointerEvent, ReactNode } from 'react';
import ReactMarkdown from 'react-markdown';
import {
  Activity, ArrowDownToLine, ArrowRight, ArrowUp, BarChart3, BrainCircuit,
  Check, CheckCircle2, ChevronDown, ChevronLeft, ChevronRight, CircleAlert,
  CloudUpload, Copy, Database, FileBarChart2, FileSpreadsheet, FolderOpen,
  GripVertical, LayoutDashboard, LoaderCircle, Maximize2, MessageSquare,
  Minimize2, MoreHorizontal, PanelLeftClose, PanelLeftOpen, PanelRightClose,
  PanelRightOpen, Pin, Plus, RotateCcw, Search, Send, Settings2, ShieldCheck,
  Sparkles, Table2, Trash2, TrendingUp, UploadCloud, X
} from 'lucide-react';

type Page = 'Overview' | 'AI Analyst' | 'Datasets' | 'Visualization Studio' | 'ML Studio' | 'Reports';
type DataTable = { columns: string[]; rows: Record<string, unknown>[]; total: number };
type Dataset = { name: string; rows: number; columns: number; missing: number; missing_pct: number;
  duplicates: number; fields: { name: string; type: string; missing: number }[]; preview: DataTable };
type Message = { role: 'user' | 'assistant'; content: string; task?: string; chart?: string | null;
  table?: DataTable | null; has_chart?: boolean; has_table?: boolean };
type StoredSession = { id: string; title: string; updated: number };
type ApiResponse = { detail?: string; [key: string]: unknown };
type FileSystemEntryExt = { isFile: boolean; isDirectory: boolean; name: string;
  file?: (cb: (file: File) => void, error?: (e: Error) => void) => void;
  createReader?: () => { readEntries: (cb: (entries: FileSystemEntryExt[]) => void,
    error?: (e: Error) => void) => void } };

const PAGE_ITEMS = [
  { id: 'Overview' as const, icon: LayoutDashboard },
  { id: 'AI Analyst' as const, icon: MessageSquare },
  { id: 'Datasets' as const, icon: Database },
  { id: 'Visualization Studio' as const, icon: BarChart3 },
  { id: 'ML Studio' as const, icon: BrainCircuit },
  { id: 'Reports' as const, icon: FileBarChart2 },
];

function readStore<T>(key: string, defaultValue: T): T {
  try { const v = localStorage.getItem(key); return v ? JSON.parse(v) as T : defaultValue; }
  catch { return defaultValue; }
}
function useStore<T>(key: string, initial: T): [T, (value: T | ((old: T) => T)) => void] {
  const [value, setValue] = useState<T>(() => readStore(key, initial));
  useEffect(() => { localStorage.setItem(key, JSON.stringify(value)); }, [key, value]);
  return [value, setValue];
}
async function api<T>(url: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try { response = await fetch(url, init); }
  catch { throw new Error('Cannot reach the local backend. Start the FastAPI server on port 8000.'); }
  const data = await response.json().catch(() => ({})) as ApiResponse;
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `Request failed (${response.status}).`);
  return data as T;
}
function post<T>(url: string, data: unknown): Promise<T> {
  return api<T>(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
}
function shortNum(num: number): string {
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
  if (num >= 10000) return `${(num / 1000).toFixed(1)}K`;
  return num.toLocaleString();
}
function prettyTime(unix: number): string {
  return new Date(unix).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}
async function entryFiles(entry: FileSystemEntryExt): Promise<File[]> {
  if (entry.isFile && entry.file) {
    return new Promise(resolve => entry.file!(file => resolve([file]), () => resolve([])));
  }
  if (entry.isDirectory && entry.createReader) {
    const reader = entry.createReader();
    const all: FileSystemEntryExt[] = [];
    while (true) {
      const batch = await new Promise<FileSystemEntryExt[]>(resolve => reader.readEntries(resolve, () => resolve([])));
      if (!batch.length) break;
      all.push(...batch);
    }
    return (await Promise.all(all.map(entryFiles))).flat();
  }
  return [];
}
async function droppedFiles(event: DragEvent): Promise<File[]> {
  const items = Array.from(event.dataTransfer.items || []);
  const roots = items.map(item => (item as DataTransferItem & {
    webkitGetAsEntry?: () => FileSystemEntryExt | null }).webkitGetAsEntry?.()).filter(Boolean) as FileSystemEntryExt[];
  if (roots.length) return (await Promise.all(roots.map(entryFiles))).flat();
  return Array.from(event.dataTransfer.files);
}

function IconButton({ children, label, onClick, active, disabled }: {
  children: ReactNode; label: string; onClick: () => void; active?: boolean; disabled?: boolean
}) {
  return <button type="button" className={`icon-button ${active ? 'is-active' : ''}`} title={label}
    aria-label={label} onClick={onClick} disabled={disabled}>{children}</button>;
}

function DataTableView({ data, compact = false }: { data: DataTable | null; compact?: boolean }) {
  if (!data || !data.columns?.length) return <div className="empty-subtle">No table data to display yet.</div>;
  return <div className={`data-table-wrap ${compact ? 'compact' : ''}`}>
    <table className="data-table"><thead><tr>{data.columns.map((col, index) =>
      <th key={index}>{col}</th>)}</tr></thead><tbody>
      {data.rows.map((row, index) => <tr key={index}>{data.columns.map((col, j) =>
        <td key={j} title={String(row[col] ?? '')}>{String(row[col] ?? '—')}</td>)}</tr>)}
    </tbody></table>
    <div className="table-footer">Showing {data.rows.length.toLocaleString()} of {data.total.toLocaleString()} rows</div>
  </div>;
}

function UploadCard({ onFiles, uploading, dataset, onDownload, dense = false }: {
  onFiles: (files: File[]) => void; uploading: boolean;
  dataset: Dataset | null; onDownload: () => void; dense?: boolean;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const folderRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  useEffect(() => { folderRef.current?.setAttribute('webkitdirectory', ''); folderRef.current?.setAttribute('directory', ''); }, []);
  const handleFileInput = (ev: ChangeEvent<HTMLInputElement>) => {
    const selected = Array.from(ev.target.files || []);
    if (selected.length) onFiles(selected);
    ev.target.value = '';
  };
  return <div className={`panel-card upload-card ${dense ? 'dense' : ''}`}>
    <div className="card-heading"><span>{dataset ? 'Your data is ready' : 'Upload dataset'}</span>
      {dataset ? <CheckCircle2 size={18} className="success-icon" /> : <UploadCloud size={18} />}</div>
    <div className={`dropzone ${dragging ? 'drag-over' : ''} ${dataset ? 'uploaded-state' : ''}`}
      role="group" aria-label="Drop CSV, Excel or a folder to import datasets"
      onDragOver={e => { e.preventDefault(); setDragging(true); }}
      onDragLeave={e => { e.preventDefault(); setDragging(false); }}
      onDrop={e => { e.preventDefault(); setDragging(false); if (!uploading) void droppedFiles(e).then(onFiles); }}>
      {dataset ? <>
        <div className="upload-ready-line"><FileSpreadsheet size={18} /><span title={dataset.name}>{dataset.name}</span></div>
        <div className="drop-description">{dataset.rows.toLocaleString()} rows · {dataset.columns} columns</div>
        <button className="upload-download" type="button" onClick={onDownload}>
          <ArrowDownToLine size={16}/> Download current data
        </button>
        <div className="upload-subtext">CSV export includes any analysis changes</div>
        <div className="upload-actions replace-actions">
          <button type="button" disabled={uploading} onClick={() => fileRef.current?.click()}>Replace files</button>
          <button type="button" disabled={uploading} onClick={() => folderRef.current?.click()}>Replace folder</button>
        </div>
        {dragging && <span className="drop-replace-indicator">Drop to load a different dataset</span>}
      </> : <>
        <div className="upload-icon"><CloudUpload size={23} strokeWidth={1.7} /></div>
        <div className="drop-title">{uploading ? 'Importing your data…' : 'Drag files or folder here'}</div>
        <div className="drop-description">CSV, XLSX or XLS · up to 25MB/file</div>
        <div className="upload-actions">
          <button type="button" disabled={uploading} onClick={() => fileRef.current?.click()}>Browse files</button>
          <button type="button" disabled={uploading} onClick={() => folderRef.current?.click()}>Browse folder</button>
        </div>
      </>}
      <input ref={fileRef} type="file" accept=".csv,.xlsx,.xls" multiple hidden onChange={handleFileInput} />
      <input ref={folderRef} type="file" multiple hidden onChange={handleFileInput} />
    </div>
  </div>;
}

function App() {
  const [page, setPage] = useState<Page>('AI Analyst');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessions, setSessions] = useStore<StoredSession[]>('if-v4-sessions', []);
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [datasetOptions, setDatasetOptions] = useState<string[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [chart, setChart] = useState<string | null>(null);
  const [table, setTable] = useState<DataTable | null>(null);
  const [draft, setDraft] = useState('');
  const [studioDraft, setStudioDraft] = useState('');
  const [columnFilter, setColumnFilter] = useState('');
  const [status, setStatus] = useState<{ ollama: boolean; model: string; message: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [tab, setTab] = useState<'chart' | 'data'>('chart');
  const [datasetTab, setDatasetTab] = useState<'overview' | 'columns'>('overview');
  const [chartPinned, setChartPinned] = useState(false);
  const sendingRef = useRef(false);
  const [chartExpanded, setChartExpanded] = useState(false);
  const [rightVisible, setRightVisible] = useState(true);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useStore<number>('if-v4-sidebar-width', 232);
  const [rightWidth, setRightWidth] = useStore<number>('if-v4-right-width', 330);
  const [topHeight, setTopHeight] = useStore<number>('if-v4-top-height', 390);
  const beforeExpanded = useRef(330);
  const endRef = useRef<HTMLDivElement>(null);
  const messagesRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const activeId = useRef<string | null>(null);
  const workspaceRef = useRef<HTMLDivElement>(null);
  const rightRef = useRef<HTMLDivElement>(null);
  const studioStreamRef = useRef<HTMLDivElement>(null);

  const updateSessionList = useCallback((id: string, title?: string) => {
    setSessions(old => {
      const found = old.find(item => item.id === id);
      const next: StoredSession = { id, title: title ?? found?.title ?? 'New analysis', updated: Date.now() };
      return [next, ...old.filter(item => item.id !== id)].slice(0, 25);
    });
  }, [setSessions]);

  const newSession = useCallback(async () => {
    setError(''); setNotice(''); setBusy(false);
    const created = await post<{ session_id: string }>('/api/session', {});
    activeId.current = created.session_id;
    setSessionId(created.session_id);
    setDataset(null); setDatasetOptions([]); setMessages([]); setChart(null); setTable(null); setChartPinned(false);
    setDraft(''); setStudioDraft(''); setPage('AI Analyst');
    updateSessionList(created.session_id, 'New analysis');
  }, [updateSessionList]);

  const openSession = useCallback(async (id: string) => {
    setError('');
    try {
      const data = await api<{ dataset: Dataset | null; datasets: string[]; messages: Message[]; chart: string | null;
        table: DataTable | null }>(`/api/session/${encodeURIComponent(id)}`);
      activeId.current = id; setSessionId(id); setDataset(data.dataset);
      setMessages(data.messages); setChart(data.chart); setTable(data.table); setDatasetOptions(data.datasets || []);
      setChartPinned(false); setPage('AI Analyst');
      localStorage.setItem('if-v4-active-session', id);
    } catch (err) {
      setError(err instanceof Error ? `${err.message} Old sessions are held in local server memory.` : 'Cannot restore session.');
    }
  }, []);

  useEffect(() => {
    let aborted = false;
    const boot = async () => {
      try {
        const health = await api<{ ollama: boolean; model: string; message: string }>('/api/health');
        if (!aborted) setStatus(health);
      } catch { if (!aborted) setStatus(null); }
      if (aborted) return;
      const old = localStorage.getItem('if-v4-active-session');
      if (old) {
        try {
          const previous = await api<{ dataset: Dataset | null; datasets: string[]; messages: Message[]; chart: string | null;
            table: DataTable | null }>(`/api/session/${encodeURIComponent(old)}`);
          if (aborted) return;
          activeId.current = old; setSessionId(old); setDataset(previous.dataset);
          setMessages(previous.messages); setChart(previous.chart); setTable(previous.table); setDatasetOptions(previous.datasets || []);
          return;
        } catch { localStorage.removeItem('if-v4-active-session'); }
      }
      if (!aborted) {
        try {
          const fresh = await post<{ session_id: string }>('/api/session', {});
          if (aborted) return;
          activeId.current = fresh.session_id; setSessionId(fresh.session_id);
          setSessions(items => [{ id: fresh.session_id, title: 'New analysis', updated: Date.now() }, ...items].slice(0, 25));
        } catch { if (!aborted) setError('Cannot connect to the backend. Run the Python API server first.'); }
      }
    };
    void boot();
    const interval = window.setInterval(async () => {
      try { const health = await api<{ ollama: boolean; model: string; message: string }>('/api/health');
        if (!aborted) setStatus(health); } catch { if (!aborted) setStatus(null); }
    }, 20000);
    return () => { aborted = true; window.clearInterval(interval); };
    // Only run boot on mount; actions update sessions independently.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => { if (sessionId) localStorage.setItem('if-v4-active-session', sessionId); }, [sessionId]);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }); }, [messages, busy]);
  useEffect(() => {
    const el = studioStreamRef.current;
    if (page === 'Visualization Studio' && el) el.scrollTop = el.scrollHeight;
  }, [messages, busy, page]);

  const upload = useCallback(async (files: File[]) => {
    const id = activeId.current;
    if (!id || !files.length) return;
    const supported = files.filter(file => /\.(csv|xlsx|xls)$/i.test(file.name));
    if (!supported.length) { setError('This folder contains no supported CSV or Excel files.'); return; }
    if (supported.length > 20) { setError('Choose a folder with at most 20 supported files.'); return; }
    setUploading(true); setError(''); setNotice('');
    const data = new FormData(); data.append('session_id', id);
    supported.forEach(file => data.append('files', file, file.name));
    try {
      const response = await api<{ dataset: Dataset; imported: string[]; ignored: string[]; notice?: string }>('/api/upload', {
        method: 'POST', body: data,
      });
      if (activeId.current !== id) return;
      setDataset(response.dataset); setDatasetOptions(response.imported); setMessages([]); setChart(null); setTable(null); setChartPinned(false);
      updateSessionList(id, response.dataset.name);
      if (response.notice || response.ignored.length) setNotice([response.notice, ...response.ignored.slice(0, 2)].filter(Boolean).join(' '));
      setPage('AI Analyst');
    } catch (err) { setError(err instanceof Error ? err.message : 'Upload failed.'); }
    finally { setUploading(false); }
  }, [updateSessionList]);

  const loadSample = useCallback(async () => {
    const id = activeId.current;
    if (!id) return;
    setUploading(true); setError(''); setNotice('');
    try {
      const data = await post<{ dataset: Dataset; imported: string[] }>('/api/sample', { session_id: id });
      if (activeId.current !== id) return;
      setDataset(data.dataset); setDatasetOptions(data.imported); setMessages([]); setChart(null); setTable(null);
      updateSessionList(id, data.dataset.name); setPage('AI Analyst');
    } catch (err) { setError(err instanceof Error ? err.message : 'Could not load sample dataset.'); }
    finally { setUploading(false); }
  }, [updateSessionList]);

  const selectDataset = useCallback(async (name: string) => {
    const id = activeId.current;
    if (!id) return;
    setUploading(true); setError('');
    try {
      const response = await post<{ dataset: Dataset; imported: string[] }>('/api/select-dataset', { session_id: id, name });
      if (activeId.current !== id) return;
      setDataset(response.dataset); setDatasetOptions(response.imported);
      setMessages([]); setChart(null); setTable(null); setChartPinned(false);
      updateSessionList(id, name);
    } catch (err) { setError(err instanceof Error ? err.message : 'Cannot select dataset.'); }
    finally { setUploading(false); }
  }, [updateSessionList]);

  const send = useCallback(async (question = draft, source: 'analyst' | 'studio' = 'analyst') => {
    const id = activeId.current;
    const clean = question.trim();
    if (!id || !clean || busy || sendingRef.current) return;
    if (!dataset) { setError('Upload a dataset first, or use the sample dataset.'); return; }
    sendingRef.current = true;
    if (source !== 'studio') setPage('AI Analyst');
    setError(''); setNotice(''); setDraft(''); setStudioDraft(''); setBusy(true);
    setMessages(old => [...old, { role: 'user', content: clean }]);
    try {
      const result = await post<{ answer: string; chart: string | null; table: DataTable | null;
        dataset: Dataset; task: string }>('/api/query', { session_id: id, question: clean });
      if (activeId.current !== id) return;
      setMessages(old => [...old, { role: 'assistant', content: result.answer, chart: result.chart,
        table: result.table, task: result.task }]);
      setDataset(result.dataset);
      if (!chartPinned) {
        if (result.chart) { setChart(result.chart); setTab('chart'); }
        if (result.table) { setTable(result.table); if (!result.chart) setTab('data'); }
      }
      if (messages.length === 0) updateSessionList(id, clean.length > 30 ? clean.slice(0, 30) + '…' : clean);
      else updateSessionList(id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed.');
      setMessages(old => [...old, { role: 'assistant', content: 'I couldn’t complete that request. Please check the error above and try again.' }]);
    } finally { sendingRef.current = false; setBusy(false); }
  }, [busy, chartPinned, dataset, draft, messages.length, updateSessionList]);

  // Quick questions are one-click actions: they are sent immediately.
  const selectPrompt = (prompt: string) => { void send(prompt); }; 
  const askShortcut = (question: string) => { void send(question); };

  const dragDivider = (axis: 'sidebar' | 'right' | 'top') => (e: PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    const startX = e.clientX, startY = e.clientY;
    const startingSide = sidebarWidth, startingRight = rightWidth, startingTop = topHeight;
    const onMove = (event: globalThis.PointerEvent) => {
      const totalWidth = workspaceRef.current?.clientWidth ?? window.innerWidth;
      const sideLimit = sidebarCollapsed ? 70 : startingSide;
      if (axis === 'sidebar') setSidebarWidth(Math.max(186, Math.min(300, startingSide + event.clientX - startX)));
      if (axis === 'right') {
        const max = Math.max(275, totalWidth - (sidebarCollapsed ? 75 : sideLimit) - 430);
        setRightWidth(Math.max(275, Math.min(Math.min(540, max), startingRight + startX - event.clientX)));
        setChartExpanded(false);
      }
      if (axis === 'top') {
        const totalHeight = rightRef.current?.clientHeight ?? 620;
        setTopHeight(Math.max(225, Math.min(totalHeight - 190, startingTop + event.clientY - startY)));
      }
    };
    const onUp = () => { window.removeEventListener('pointermove', onMove); window.removeEventListener('pointerup', onUp); document.body.classList.remove('resizing'); };
    document.body.classList.add('resizing');
    window.addEventListener('pointermove', onMove); window.addEventListener('pointerup', onUp, { once: true });
  };
  const resetSizes = () => { setSidebarWidth(232); setRightWidth(330); setTopHeight(390); setChartExpanded(false); };
  const expandChart = () => {
    if (chartExpanded) { setRightWidth(beforeExpanded.current); setChartExpanded(false); }
    else {
      beforeExpanded.current = rightWidth;
      const total = workspaceRef.current?.clientWidth ?? window.innerWidth;
      const max = Math.max(275, total - (sidebarCollapsed ? 75 : sidebarWidth) - 430);
      setRightWidth(Math.min(520, max)); setChartExpanded(true);
    }
  };
  const showResult = (message: Message) => {
    if (!message.chart && !message.table) return;
    setChart(message.chart || null);
    setTable(message.table || null);
    setTab(message.chart ? 'chart' : 'data');
    setPage('Visualization Studio');
    setError('');
  };
  const copyText = (text: string) => { void navigator.clipboard?.writeText(text).then(() => setNotice('Copied to clipboard.')); };
  const saveImage = () => {
    if (!chart) return;
    const a = document.createElement('a'); a.href = chart; a.download = 'insightforge_chart.png'; a.click();
  };
  const exportDataset = () => {
    if (!sessionId || !dataset) { setError('Select a dataset first.'); return; }
    window.location.href = `/api/export/${encodeURIComponent(sessionId)}`;
  };

  const renderChat = () => <div className="chat-surface">
    <div className="chat-heading">
      <div><h2>{sessions.find(item => item.id === sessionId)?.title || 'New analysis'}</h2>
        <span><FileSpreadsheet size={13} /> {dataset ? dataset.name : 'No dataset selected'}</span></div>
      <div className="heading-actions">
        <IconButton label="New analysis" onClick={() => { void newSession().catch(err => setError(String(err))); }}><Plus size={17} /></IconButton>
        <IconButton label="More options" onClick={() => setShowSettings(true)}><MoreHorizontal size={19} /></IconButton>
      </div>
    </div>
    <div className="chat-messages" ref={messagesRef}>
      {messages.length === 0 && <div className="chat-empty">
        <div className="hero-symbol"><Sparkles size={26} strokeWidth={1.8} /></div>
        <h1>Your data, explained.</h1>
        <p>Ask questions, explore patterns, and turn your dataset into meaningful insights.</p>
        {!dataset ? <div className="empty-dataset-note"><Database size={16} /> Upload a dataset in the right panel, or <button onClick={() => void loadSample()} disabled={uploading}>load sample data</button>.</div>
          : <div className="dataset-ready"><CheckCircle2 size={16} /> {dataset.name} is ready for analysis</div>}
        <div className="suggested-grid">
          {['Summarize this dataset', 'Find missing values', 'Show a useful visualization', 'Identify unusual patterns'].map(prompt =>
            <button key={prompt} type="button" disabled={busy || !dataset} title={dataset ? 'Ask this question immediately' : 'Upload a dataset to use quick questions'} onClick={() => selectPrompt(prompt)}><Sparkles size={14} />{prompt}<ArrowRight size={13} /></button>)}
        </div>
      </div>}
      {messages.map((message, index) => <div key={index} className={`chat-message ${message.role}`}>
        {message.role === 'assistant' && <div className="assistant-icon"><Sparkles size={16} /></div>}
        <div className="message-content">
          {message.role === 'assistant' && <strong className="message-author">InsightForge AI</strong>}
          {message.role === 'user' ? <p>{message.content}</p> : <div className="answer-markdown"><ReactMarkdown>{message.content}</ReactMarkdown></div>}
          {message.role === 'assistant' && (message.chart || message.table) &&
            <button className="chart-result-link" type="button" onClick={() => showResult(message)}>
              <div className="result-link-icon">{message.chart ? <TrendingUp size={18} /> : <Table2 size={18} />}</div>
              <div><strong>{message.chart ? 'Analysis visualization' : 'Analysis data table'}</strong>
                <span>Open full-size {message.chart ? 'chart' : 'table'} in Visualization Studio</span></div><ArrowRight size={16} />
            </button>}
          {message.role === 'assistant' && <div className="message-tools">
            <IconButton label="Copy answer" onClick={() => copyText(message.content)}><Copy size={14} /></IconButton>
            {message.task && message.task !== 'conversation' && <span><ShieldCheck size={13} /> Pandas-verified</span>}
          </div>}
        </div>
      </div>)}
      {busy && <div className="chat-message assistant"><div className="assistant-icon"><Sparkles size={16} /></div>
        <div className="working-indicator"><LoaderCircle size={15} className="spin" /> Analyzing dataset with Qwen3 and Pandas…</div></div>}
      <div ref={endRef} />
    </div>
    <div className="composer-wrap">
      <div className="composer">
        <textarea ref={textareaRef} rows={2} value={draft} onChange={e => setDraft(e.target.value)}
          onKeyDown={(e: KeyboardEvent<HTMLTextAreaElement>) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void send(); } }}
          placeholder="Ask anything about your data…" aria-label="Ask about your dataset" />
        <div className="composer-bottom">
          <div className="composer-utilities">
            <IconButton label="Show upload panel" onClick={() => setRightVisible(true)}><UploadCloud size={17} /></IconButton>
            <span className="attached-file"><FileSpreadsheet size={13} />{dataset ? dataset.name : 'Attach a dataset'}</span>
          </div>
          <button className="send-button" type="button" onClick={() => void send()}
            disabled={!draft.trim() || busy || !dataset} title="Send question"><ArrowUp size={20} /></button>
        </div>
      </div>
      <span className="composer-footer">Local AI · Calculations verified with Pandas · Shift + Enter for new line</span>
    </div>
  </div>;

  const renderStudio = () => {
    const savedResults = messages.filter(m => m.role === 'assistant' && (m.chart || m.table));
    return <section className="studio-page" aria-label="Visualization Studio">
      <div className="studio-heading">
        <div><div className="eyebrow">VISUALIZATION WORKSPACE</div>
          <h1>Visualization Studio</h1>
          <p>Explore a full-size chart or data table and ask follow-up questions alongside it.</p></div>
        <div className="studio-top-actions">
          <button className="secondary-action" type="button" onClick={() => setPage('AI Analyst')}><ChevronLeft size={16}/> AI Analyst</button>
          <button className="secondary-action" type="button" aria-pressed={chartPinned} onClick={() => setChartPinned(v => !v)}><Pin size={16}/> {chartPinned ? 'Unpin result' : 'Pin result'}</button>
          <button className="secondary-action" type="button" disabled={!chart} onClick={saveImage}><ArrowDownToLine size={16}/> Save chart</button>
        </div>
      </div>
      <div className="studio-grid">
        <div className="studio-primary">
          <div className="studio-results-heading">
            <div><strong>{dataset?.name || 'No dataset loaded'}</strong><span>{savedResults.length} saved analysis {savedResults.length === 1 ? 'result' : 'results'} in this session</span></div>
            <div className="studio-view-switch" role="tablist" aria-label="Visualization type">
              <button type="button" role="tab" aria-selected={tab === 'chart'} className={tab === 'chart' ? 'on' : ''} onClick={() => setTab('chart')}><BarChart3 size={16}/> Chart</button>
              <button type="button" role="tab" aria-selected={tab === 'data'} className={tab === 'data' ? 'on' : ''} onClick={() => setTab('data')}><Table2 size={16}/> Table</button>
            </div>
          </div>
          <div className="studio-canvas">
            {tab === 'chart' ? (chart ? <img src={chart} alt="Full-size verified analytical visualization" className="studio-chart"/>
              : <div className="studio-empty"><BarChart3 size={40}/><h3>No chart selected</h3>
                <p>Ask InsightForge for a bar chart, histogram, correlation heatmap, or another visualization.</p>
                <button className="secondary-action" disabled={!dataset || busy} onClick={() => void send('Show a useful visualization for this dataset', 'studio')}>Generate a chart <ArrowRight size={15}/></button></div>)
              : table ? <div className="studio-table"><DataTableView data={table}/></div>
                : <div className="studio-empty"><Table2 size={40}/><h3>No result table selected</h3><p>Ask for grouped comparisons, missing values or top records to generate a data table.</p></div>}
          </div>
          <div className="studio-results-list">
            <div className="studio-results-title">Previous results <span>{savedResults.length}</span></div>
            <div className="studio-result-items">
              {savedResults.length ? savedResults.map((result, i) => <button key={i} type="button"
                className="studio-result-item" onClick={() => showResult(result)} title="Open this analysis result">
                {result.chart ? <BarChart3 size={15}/> : <Table2 size={15}/>}
                <span>{result.task && result.task !== 'conversation' ? result.task.replace(/_/g, ' ') : 'Analysis result'} #{i+1}</span><ArrowRight size={14}/>
              </button>) : <span className="studio-no-results">Visualizations created in AI Analyst appear here automatically.</span>}
            </div>
          </div>
        </div>
        <div className="studio-chat">
          <div className="studio-chat-heading"><Sparkles size={19}/><div><strong>Ask InsightForge</strong><span>Answers share your active dataset and history</span></div></div>
          <div className="studio-chat-stream" ref={studioStreamRef}>
            {messages.length ? messages.map((message, index) => <div key={index} className={`studio-message ${message.role}`}>
              {message.role === 'assistant' && <div className="studio-message-author"><Sparkles size={14}/> InsightForge AI</div>}
              {message.role === 'assistant' ? <div className="answer-markdown"><ReactMarkdown>{message.content}</ReactMarkdown></div>
                : <p>{message.content}</p>}
              {message.role === 'assistant' && (message.chart || message.table) &&
                <button className="studio-show-result" type="button" onClick={() => showResult(message)}>
                  {message.chart ? <BarChart3 size={14}/> : <Table2 size={14}/>} Show {message.chart ? 'chart' : 'table'} <ArrowRight size={13}/>
                </button>}
            </div>) : <div className="studio-chat-empty"><Sparkles size={23}/><strong>Analyze visually</strong><p>Ask about the chart or your uploaded dataset.</p></div>}
            {busy && <div className="working-indicator"><LoaderCircle className="spin" size={16}/> Thinking and calculating…</div>}
          </div>
          <div className="studio-chat-bottom">
            <form onSubmit={e => { e.preventDefault(); void send(studioDraft, 'studio'); }} className="studio-chat-composer">
              <textarea value={studioDraft} onChange={e => setStudioDraft(e.target.value)} rows={2}
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void send(studioDraft, 'studio'); } }}
                placeholder="Ask about the graph or your dataset…" aria-label="Ask from Visualization Studio"/>
              <button className="send-button" type="submit" disabled={!studioDraft.trim() || busy || !dataset} aria-label="Send visualization question"><ArrowUp size={18}/></button>
            </form>
            <span>Uses the same conversation and local Qwen3 model</span>
          </div>
        </div>
      </div>
    </section>;
  };

  const renderPage = () => {
    if (page === 'AI Analyst') return renderChat();
    if (page === 'Visualization Studio') return renderStudio();
    if (page === 'Overview') return <section className="content-page">
      <div className="eyebrow">WORKSPACE OVERVIEW</div><h1>Good to see you.</h1>
      <p className="page-lead">Your data analysis workspace, ready whenever you are.</p>
      <div className="overview-metrics">
        <div><Database size={19} /><span>Active dataset</span><strong>{dataset?.name ?? 'None selected'}</strong></div>
        <div><Table2 size={19} /><span>Dataset rows</span><strong>{dataset ? shortNum(dataset.rows) : '—'}</strong></div>
        <div><MessageSquare size={19} /><span>Analysis questions</span><strong>{messages.filter(m => m.role === 'user').length}</strong></div>
      </div>
      <div className="page-grid"><div className="feature-panel"><div className="feature-icon"><Sparkles size={21} /></div>
        <h3>Start exploring</h3><p>Ask plain-English questions and see verified results instantly.</p>
        <button className="primary-action" onClick={() => setPage('AI Analyst')}>Open AI Analyst <ArrowRight size={15} /></button></div>
        <div className="feature-panel"><div className="feature-icon"><FileSpreadsheet size={21} /></div>
          <h3>Try a sample dataset</h3><p>No dataset yet? Load the bundled employee data to explore.</p>
          <button className="secondary-action" onClick={() => void loadSample()} disabled={uploading}>Load sample data <ArrowRight size={15} /></button></div></div>
    </section>;
    if (page === 'Datasets') return <section className="content-page">
      <div className="eyebrow">DATA EXPLORER</div><h1>Datasets</h1>
      <p className="page-lead">Inspect the active data source and review its columns.</p>
      {!dataset ? <div className="feature-panel"><FolderOpen size={26} /><h3>Nothing uploaded yet</h3><p>Upload a CSV, Excel file or folder from the right panel to begin.</p>
        <button className="secondary-action" onClick={() => void loadSample()}>Load sample dataset</button></div> : <>
        <div className="dataset-hero"><FileSpreadsheet size={21} /><div><strong>{dataset.name}</strong><span>{dataset.rows.toLocaleString()} rows · {dataset.columns} columns · {dataset.missing.toLocaleString()} missing cells</span></div>
          <button className="text-button" onClick={exportDataset}><ArrowDownToLine size={15} /> Export CSV</button></div>
        <h3 className="subsection-title">Data preview</h3><DataTableView data={dataset.preview}/>
        <h3 className="subsection-title">Column types</h3><div className="field-list">
          {dataset.fields.map(field => <div key={field.name}><strong>{field.name}</strong><span>{field.type}</span><span>{field.missing} missing</span></div>)}
        </div></>}
    </section>;
    if (page === 'ML Studio') return <section className="content-page">
      <div className="eyebrow">MACHINE LEARNING</div><h1>ML Studio</h1><p className="page-lead">Use existing analysis-engine ML operations through the AI Analyst.</p>
      <div className="page-grid">{[
        { title: 'Correlation analysis', desc: 'Examine relationships between numerical features.', prompt: 'Find the strongest correlations in this dataset', icon: Activity },
        { title: 'Clustering', desc: 'Discover natural groups in your data.', prompt: 'Perform K-Means clustering on this dataset', icon: BarChart3 },
        { title: 'Classification', desc: 'Explore supervised model performance.', prompt: 'Train a classification model and show evaluation metrics', icon: BrainCircuit },
        { title: 'Anomaly detection', desc: 'Examine unusual numerical values.', prompt: 'Detect outliers in this dataset', icon: TrendingUp },
      ].map(item => <div className="feature-panel" key={item.title}><div className="feature-icon"><item.icon size={21}/></div>
        <h3>{item.title}</h3><p>{item.desc}</p><button className="secondary-action" onClick={() => selectPrompt(item.prompt)}>Ask AI Analyst <ArrowRight size={14}/></button></div>)}</div>
      <div className="note-block"><CircleAlert size={16}/> Dedicated AutoML experiments and SHAP explainability are planned enhancements, not yet implemented in v4.</div>
    </section>;
    return <section className="content-page">
      <div className="eyebrow">EXPORTS</div><h1>Reports & exports</h1>
      <p className="page-lead">Save your modified dataset and chart from the current analysis session.</p>
      <div className="page-grid"><div className="feature-panel"><FileSpreadsheet size={22}/><h3>Export dataset</h3>
        <p>Download your current data, including cleaning or transformation changes.</p>
        <button className="secondary-action" onClick={exportDataset} disabled={!dataset}><ArrowDownToLine size={14}/> Download CSV</button></div>
        <div className="feature-panel"><BarChart3 size={22}/><h3>Export latest chart</h3><p>Save the current visualization as a PNG image.</p>
        <button className="secondary-action" onClick={saveImage} disabled={!chart}><ArrowDownToLine size={14}/> Download PNG</button></div></div>
      <div className="note-block">PDF report generation is a future milestone. No report is invented from incomplete analysis.</div>
    </section>;
  };

  return <div className="app-shell">
    <header className="topbar"><div className="brand">
      <div className="brand-icon"><TrendingUp size={22} /></div>
      <div><strong>InsightForge</strong><small>AI DATA ANALYST</small></div></div>
      <div className="header-end"><div className={`model-pill ${status?.ollama ? 'online' : 'offline'}`} title={status?.message ?? 'Backend unavailable'}>
        <span className="model-dot" />{status?.ollama ? status.model : 'Local AI unavailable'}</div>
        {page !== 'Visualization Studio' && <IconButton label={rightVisible ? 'Hide right panel' : 'Show right panel'} onClick={() => setRightVisible(v => !v)}>
          {rightVisible ? <PanelRightClose size={19}/> : <PanelRightOpen size={19}/>}</IconButton>}
        </div></header>
    <div className="workspace" ref={workspaceRef}>
      <aside className={`sidebar ${sidebarCollapsed ? 'collapsed' : ''}`} style={{ width: sidebarCollapsed ? 65 : sidebarWidth }}>
        <div className="sidebar-top">
          <button className="new-analysis" type="button" onClick={() => { void newSession().catch(err => setError(String(err))); }} title="New analysis">
            <Plus size={17}/>{!sidebarCollapsed && <span>New Analysis</span>}</button>
          <nav className="primary-nav" aria-label="Main navigation">{PAGE_ITEMS.map(item =>
            <button key={item.id} title={item.id} className={page === item.id ? 'selected' : ''}
              onClick={() => setPage(item.id)}><item.icon size={18}/>{!sidebarCollapsed && <span>{item.id}</span>}</button>)}</nav>
        </div>
        <div className="sidebar-divider" />
        {!sidebarCollapsed && <div className="history-section"><div className="history-heading"><span>RECENT CHATS</span><Search size={15}/></div>
          <div className="history-items">{sessions.map(s =>
            <button key={s.id} className={s.id === sessionId ? 'current' : ''} title={s.title} onClick={() => { void openSession(s.id); }}>
              <span className="history-text">{s.title}</span><small>{prettyTime(s.updated)}</small></button>)}</div></div>}
        <div className="sidebar-bottom"><button onClick={() => setShowSettings(true)} title="Settings"><Settings2 size={18}/>{!sidebarCollapsed && <span>Settings</span>}</button>
          <button onClick={() => setSidebarCollapsed(v => !v)} title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}>
            {sidebarCollapsed ? <PanelLeftOpen size={18}/> : <PanelLeftClose size={18}/>}{!sidebarCollapsed && <span>Collapse sidebar</span>}</button></div>
      </aside>
      {!sidebarCollapsed && <div className="splitter splitter-vertical" title="Drag to resize sidebar, double click to reset"
        onPointerDown={dragDivider('sidebar')} onDoubleClick={resetSizes}><GripVertical size={12}/></div>}
      <main className="main-panel">{error && <div className="floating-alert error" role="alert"><CircleAlert size={16}/><span>{error}</span><button aria-label="Dismiss error" onClick={() => setError('')}><X size={15}/></button></div>}
        {notice && <div className="floating-alert info" role="status"><CheckCircle2 size={16}/><span>{notice}</span><button aria-label="Dismiss notice" onClick={() => setNotice('')}><X size={15}/></button></div>}
        {renderPage()}</main>
      {rightVisible && page !== 'Visualization Studio' && <>
        <div className="splitter splitter-vertical right-splitter" title="Drag to resize chart panel, double click to reset"
          onPointerDown={dragDivider('right')} onDoubleClick={resetSizes}><GripVertical size={12}/></div>
        <aside className="right-panel" style={{ width: rightWidth }} ref={rightRef}>
          <div className="right-top" style={{ height: topHeight }}>
            <UploadCard onFiles={files => { void upload(files); }} uploading={uploading} dataset={dataset} onDownload={exportDataset} dense/>
            <div className="panel-card dataset-card">
              <div className="card-heading"><span>Active dataset</span><ChevronDown size={16}/></div>
              {dataset ? <><div className="dataset-file"><FileSpreadsheet size={18}/><span title={dataset.name}>{dataset.name}</span><CheckCircle2 size={15}/></div>
                {datasetOptions.length > 1 && <select className="dataset-select" aria-label="Choose active dataset" value={dataset.name}
                  onChange={e => { void selectDataset(e.target.value); }} disabled={uploading}>
                  {datasetOptions.map(name => <option key={name} value={name}>{name}</option>)}
                </select>}
                <div className="mini-stats"><div><small>ROWS</small><strong>{shortNum(dataset.rows)}</strong></div>
                  <div><small>COLS</small><strong>{dataset.columns}</strong></div>
                  <div><small>MISSING</small><strong>{dataset.missing_pct}%</strong></div></div>
                <div className="mini-tabs" role="tablist" aria-label="Active dataset details">
                  <button type="button" role="tab" aria-selected={datasetTab==='overview'} className={datasetTab==='overview'?'on':''} onClick={()=>setDatasetTab('overview')}>Overview</button>
                  <button type="button" role="tab" aria-selected={datasetTab==='columns'} className={datasetTab==='columns'?'on':''} onClick={()=>setDatasetTab('columns')}>Columns</button></div>
                {datasetTab === 'overview' ? <div className="dataset-overview-details" role="tabpanel">
                  <div className="dataset-detail-row"><span>Duplicate rows</span><strong>{dataset.duplicates.toLocaleString()}</strong></div>
                  <div className="dataset-detail-row"><span>Missing cells</span><strong>{dataset.missing.toLocaleString()}</strong></div>
                  <div className="dataset-detail-row"><span>Numeric columns</span><strong>{dataset.fields.filter(f => /int|float|double|number/i.test(f.type)).length}</strong></div>
                  <button className="dataset-panel-link" onClick={() => setPage('Datasets')}>See dataset preview <ArrowRight size={14}/></button>
                </div> : <div className="dataset-columns-details" role="tabpanel">
                  <input type="search" className="column-search" value={columnFilter} onChange={e => setColumnFilter(e.target.value)}
                    placeholder={`Search ${dataset.fields.length} columns`} aria-label="Search dataset columns" />
                  <div className="column-scroll-list">{dataset.fields.filter(f => f.name.toLowerCase().includes(columnFilter.trim().toLowerCase()))
                    .map(f => <div className="column-row" key={f.name} title={`${f.name} (${f.type})`}>
                      <span>{f.name}</span><small>{f.type}{f.missing ? ` · ${f.missing} missing` : ''}</small>
                    </div>)}</div>
                  <button className="dataset-panel-link" onClick={() => setPage('Datasets')}>Explore all data <ArrowRight size={14}/></button>
                </div>}</>
                : <div className="no-dataset"><Database size={19}/><span>Your uploaded dataset will appear here.</span>
                  <button onClick={()=>void loadSample()} disabled={uploading}>Try sample dataset <ArrowRight size={13}/></button></div>}
            </div>
          </div>
          <div className="splitter splitter-horizontal" onPointerDown={dragDivider('top')} onDoubleClick={resetSizes}
            title="Drag to resize dataset and visualization panels"><div className="splitter-handle"/></div>
          <div className="panel-card visualization-card">
            <div className="visual-heading"><div><strong>Visualization Studio</strong><span>{chart ? 'Latest analysis result' : 'Waiting for a chart'}</span></div>
              <div className="visual-actions">
                <IconButton label="Open full Visualization Studio" onClick={() => setPage('Visualization Studio')}><Maximize2 size={16}/></IconButton>
                <IconButton label={chartExpanded ? 'Restore dock width' : 'Widen dock'} onClick={expandChart}>
                  {chartExpanded ? <Minimize2 size={16}/> : <ChevronLeft size={16}/>}</IconButton>
              </div></div>
            <div className="visual-tabs"><button className={tab === 'chart' ? 'on' : ''} onClick={() => setTab('chart')}>Chart</button>
              <button className={tab === 'data' ? 'on' : ''} onClick={() => setTab('data')}>Data</button></div>
            <div className="visual-content">
              {tab === 'chart' ? chart ? <img className="analysis-chart" src={chart} alt="Pandas verified analysis chart" /> :
                <div className="chart-empty"><div className="chart-empty-icon"><BarChart3 size={27}/></div><strong>Your charts appear here</strong>
                  <p>Ask the AI Analyst to create a visualization.</p></div>
                : <DataTableView data={table} compact/>}
            </div>
            <div className="visual-footer"><span><ShieldCheck size={13}/> Pandas verified</span>
              <div><IconButton active={chartPinned} label={chartPinned ? 'Unpin visualization' : 'Pin visualization'} onClick={() => setChartPinned(v=>!v)}><Pin size={15}/></IconButton>
                <IconButton label="Download visualization PNG" onClick={saveImage} disabled={!chart}><ArrowDownToLine size={15}/></IconButton></div></div>
          </div>
        </aside></>}
    </div>
    {showSettings && <div className="modal-backdrop" role="presentation" onClick={() => setShowSettings(false)}>
      <div className="settings-modal" role="dialog" aria-modal="true" aria-label="Settings" onClick={e => e.stopPropagation()}>
        <div className="modal-heading"><div><h2>Settings</h2><p>Workspace preferences and model status</p></div>
          <IconButton label="Close settings" onClick={() => setShowSettings(false)}><X size={19}/></IconButton></div>
        <div className="setting-row"><div><strong>Local model</strong><span>{status?.model ?? 'Unavailable'}</span></div><span className={`connection-label ${status?.ollama ? 'connected' : ''}`}>{status?.ollama ? 'Connected' : 'Offline'}</span></div>
        <div className="setting-row"><div><strong>Saved panel sizes</strong><span>Sidebar, visualization width, and panel split</span></div>
          <button className="secondary-action" onClick={resetSizes}><RotateCcw size={14}/> Reset layout</button></div>
        <div className="setting-row"><div><strong>Data storage</strong><span>Datasets and answers live in local server memory; panel settings and chat titles stay in this browser.</span></div></div>
        <button className="danger-action" onClick={() => { if (sessionId) void post('/api/reset',{session_id:sessionId}).then(()=>{setDataset(null);setDatasetOptions([]);setChart(null);setTable(null);setMessages([]);setShowSettings(false);}).catch(err=>setError(String(err))); }}>
          <Trash2 size={15}/> Clear current analysis</button>
      </div>
    </div>}
  </div>;
}

export default App;
