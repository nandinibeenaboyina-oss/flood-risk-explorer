import { useEffect, useState } from 'react';
import {
  Activity, AlertTriangle, ArrowDownToLine, BarChart3, Bell, BookOpen, BrainCircuit,
  Check, ChevronDown, ChevronLeft, ChevronRight, CircleHelp, Clock3, Database,
  FileSpreadsheet, FlaskConical, Gauge, GitCompareArrows, Layers3, ListFilter,
  MapPinned, Menu, Network, Play, Search, Settings2, ShieldAlert, SlidersHorizontal,
  Table2, Upload, X, Trash2, TreeDeciduous, Waves, Zap,
} from 'lucide-react';
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { CircleMarker, MapContainer, Popup, TileLayer } from 'react-leaflet';

const NAV = [
  { group: 'WORKSPACE', items: [
    ['dashboard', 'Dashboard', Gauge], ['dataset', 'Dataset', Database], ['preprocessing', 'Preprocessing', SlidersHorizontal],
    ['features', 'Feature Engineering', Layers3],
  ] },
  { group: 'MODELS', items: [
    ['decision-tree', 'Decision Tree', TreeDeciduous], ['random-forest', 'Random Forest', Network],
    ['xgboost', 'XGBoost', Zap], ['lstm', 'LSTM Forecasting', Activity], ['comparison', 'Model Comparison', GitCompareArrows],
  ] },
  { group: 'DECISION SUPPORT', items: [
    ['prediction', 'Flood Prediction', BrainCircuit], ['future', 'Future Risk Prediction', Clock3],
    ['map', 'Flood Risk Map', MapPinned], ['warning', 'Early Warning', Bell],
  ] },
  { group: 'RESEARCH', items: [
    ['research', 'Research Results', BookOpen], ['case-study', 'Case Study', FlaskConical],
    ['ablation', 'Ablation Study', BarChart3], ['performance', 'System Performance', Activity],
    ['usability', 'Usability', CircleHelp], ['history', 'Prediction History', Table2],
  ] },
];
const TITLES = Object.fromEntries(NAV.flatMap(group => group.items.map(([id, title]) => [id, title])));
const PAPER_RESULTS = [
  { model: 'Random Forest', accuracy: 64.56, precision: 0.4754, recall: 0.3894, f1: 0.3619 },
  { model: 'XGBoost', accuracy: 65.82, precision: 0.5008, recall: 0.4268, f1: 0.4387 },
  { model: 'LSTM', accuracy: 55.13, precision: 0.4034, recall: 0.4179, f1: 0.4007 },
];
const PAPER_TRAINING = [
  { model: 'Decision Tree', training_accuracy: 93.44, validation_accuracy: 52.16 },
  { model: 'Random Forest', training_accuracy: 100, validation_accuracy: 64.56 },
  { model: 'XGBoost', training_accuracy: 100, validation_accuracy: 65.82 },
  { model: 'LSTM', training_accuracy: 65.92, validation_accuracy: 55.13 },
];
const PAPER_CASE = [
  { attribute: 'Human Fatalities', reported_value: '28', prediction_value: '26' },
  { attribute: 'Human Displacement', reported_value: '176,715', prediction_value: '180,000' },
  { attribute: 'Crop Loss (hectares)', reported_value: '172,308 hectares', prediction_value: '170,500 hectares' },
  { attribute: 'Severity', reported_value: 'High', prediction_value: 'High' },
];
const PAPER_ABLATION = [
  { configuration: 'Full Feature Set', accuracy: 64.56, latency: 0.000816 },
  { configuration: 'Without Duration (Days)', accuracy: 63.29, latency: 0.000816 },
  { configuration: 'Without Human Displaced', accuracy: 51.90, latency: 0.000597 },
  { configuration: 'Without Area Affected', accuracy: 59.49, latency: 0.000573 },
  { configuration: 'Without Human Fatality', accuracy: 55.70, latency: 0.000629 },
];
const PAPER_LATENCY = [
  { component: 'Classification Model', seconds: 1.125 },
  { component: 'Forecasting Model (LSTM)', seconds: 0.496 },
  { component: 'Combined Pipeline', seconds: 1.668 },
];
const PAPER_USABILITY = [
  { metric: 'Task Completion Rate (%)', mean: 91.3, sd: 6.2, unit: '%' },
  { metric: 'Time on Task – Severity', mean: 38.4, sd: 11.7, unit: 'seconds' },
  { metric: 'Time on Task – Alerts', mean: 62.1, sd: 18.3, unit: 'seconds' },
  { metric: 'System Usability Scale', mean: 78.5, sd: 9.1, unit: 'points' },
  { metric: 'Net Promoter Score', mean: 42, sd: null, unit: 'NPS' },
];
const fmt = (value, digits = 0) => value === null || value === undefined || !Number.isFinite(Number(value)) ? '—' : new Intl.NumberFormat('en-IN', { maximumFractionDigits: digits }).format(Number(value));
const percent = value => value === null || value === undefined ? '—' : `${fmt(Number(value) * (Number(value) <= 1 ? 100 : 1), 2)}%`;
const api = async (url, options = {}) => {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status}).`);
  return payload;
};
const colorFor = label => {
  const colors = ['#2e8065', '#d47c4b', '#5b8798', '#b8963d', '#98758a', '#678759', '#c66655'];
  let hash = 0;
  for (const character of String(label)) hash = ((hash << 5) - hash + character.charCodeAt(0)) | 0;
  return colors[Math.abs(hash) % colors.length];
};

function Badge({ children, kind = 'neutral' }) { return <span className={`badge badge-${kind}`}>{children}</span>; }
function Metric({ label, value, note, icon: Icon, tone = '' }) {
  return <article className={`metric-card ${tone}`}><div className="metric-label">{label}<Icon size={15} aria-hidden="true" /></div><strong>{value ?? '—'}</strong><small>{note}</small></article>;
}
function Panel({ title, eyebrow, actions, children, className = '' }) {
  return <section className={`panel ${className}`}><div className="panel-heading"><div><div className="eyebrow">{eyebrow}</div><h2>{title}</h2></div>{actions}</div>{children}</section>;
}
function EmptyState({ title = 'Dataset not loaded', children, onUpload }) {
  return <div className="empty-state"><span className="empty-icon"><Database size={21} /></span><h2>{title}</h2><p>{children || 'Upload the research CSV to inspect its schema and calculate real results.'}</p>{onUpload && <button className="button button-primary" onClick={onUpload}><Upload size={14} />Upload dataset</button>}</div>;
}
function SourceHeading({ children, calculated = false }) {
  return <div className="source-heading"><div><div className="eyebrow">{calculated ? 'LIVE MODEL OUTPUT' : 'RESEARCH PAPER'}</div><h2>{children}</h2></div><Badge kind={calculated ? 'green' : 'paper'}>{calculated ? 'Calculated from Uploaded Dataset' : 'Reported in Research Paper'}</Badge></div>;
}
function Table({ columns, rows, empty = 'No records available.' }) {
  return <div className="table-scroll"><table><thead><tr>{columns.map(column => <th key={column.key}>{column.label}</th>)}</tr></thead><tbody>{rows.length ? rows.map((row, index) => <tr key={row.id ?? row.row_index ?? index}>{columns.map(column => <td key={column.key}>{column.render ? column.render(row[column.key], row) : row[column.key] ?? '—'}</td>)}</tr>) : <tr><td colSpan={columns.length} className="empty-row">{empty}</td></tr>}</tbody></table></div>;
}
function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return <div className="chart-tooltip"><strong>{label}</strong>{payload.map(item => <div key={item.dataKey}><span style={{ background: item.color }} />{item.name || item.dataKey}: {fmt(item.value, 3)}</div>)}</div>;
}

function App() {
  const [page, setPage] = useState('dashboard');
  const [dataset, setDataset] = useState(null);
  const [preview, setPreview] = useState({ columns: [], records: [], filtered_rows: 0, total_rows: 0 });
  const [models, setModels] = useState([]);
  const [history, setHistory] = useState([]);
  const [historyTotal, setHistoryTotal] = useState(0);
  const [apiConnected, setApiConnected] = useState(false);
  const [mapData, setMapData] = useState(null);
  const [selectedFeatures, setSelectedFeatures] = useState([]);
  const [target, setTarget] = useState('');
  const [activeModel, setActiveModel] = useState('random-forest');
  const [trainingResults, setTrainingResults] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [toast, setToast] = useState('');
  const [busy, setBusy] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [search, setSearch] = useState('');
  const [pageOffset, setPageOffset] = useState(0);
  const [sort, setSort] = useState({ column: '', descending: false });
  const [attribute, setAttribute] = useState('');
  const [trainConfig, setTrainConfig] = useState({ test_size: 0.2, max_depth: 8, n_estimators: 200, learning_rate: 0.05, subsample: 0.8, colsample_bytree: 0.8 });
  const [formValues, setFormValues] = useState({});
  const [futureYear, setFutureYear] = useState(new Date().getFullYear() + 1);
  const [futureState, setFutureState] = useState('');
  const [futureDistrict, setFutureDistrict] = useState('');
  const [preprocessOptions, setPreprocessOptions] = useState({ drop_duplicates: true, missing_strategy: 'impute', encode_categoricals: false, normalize_numeric: false });
  const [preprocessResult, setPreprocessResult] = useState(null);
  const [lstmConfig, setLstmConfig] = useState({ lookback: 7, horizon: 7, epochs: 50, units: 64 });
  const [lstmResult, setLstmResult] = useState(null);
  const [historySearch, setHistorySearch] = useState('');
  const [historyFilter, setHistoryFilter] = useState('');
  const [datasetSearch, setDatasetSearch] = useState('');

  const notify = message => { setToast(message); window.setTimeout(() => setToast(''), 4200); };
  const loadHistory = async () => {
    const params = new URLSearchParams();
    if (historySearch) params.set('search', historySearch);
    if (historyFilter) params.set('model', historyFilter);
    try { const result = await api(`/api/prediction-history?${params}`); setHistory(result.records); setHistoryTotal(result.total); } catch { setHistory([]); setHistoryTotal(0); }
  };
  const loadDataset = async (offset = pageOffset, query = datasetSearch, sortBy = sort) => {
    try {
      const info = await api('/api/dataset/info');
      const params = new URLSearchParams({ offset: String(offset), limit: '25' });
      if (query) params.set('search', query);
      if (sortBy.column) { params.set('sort_by', sortBy.column); params.set('descending', String(sortBy.descending)); }
      const [previewData, modelData, spatial] = await Promise.all([
        api(`/api/dataset/preview?${params}`), api('/api/models'), api('/api/map'),
      ]);
      setDataset(info);
      setPreview(previewData);
      setModels(modelData.trained);
      setTrainingResults(current => {
        const updated = { ...current };
        for (const result of modelData.trained) {
          const key = result.model === 'Decision Tree' ? 'decision-tree' : result.model === 'Random Forest' ? 'random-forest' : 'xgboost';
          updated[key] = result;
        }
        return updated;
      });
      setMapData(spatial);
      setTarget(current => current || info.target_candidate || '');
      setAttribute(current => current || info.columns[0] || '');
      setSelectedFeatures(current => {
        const selectedTarget = currentTarget(info) || info.target_candidate;
        const available = current.filter(column => info.columns.includes(column) && column !== selectedTarget);
        return available.length ? available : info.columns.filter(column => column !== selectedTarget);
      });
      return info;
    } catch (error) {
      if (!dataset && error.message.includes('No dataset')) { setPreview({ columns: [], records: [], filtered_rows: 0, total_rows: 0 }); }
      else notify(error.message);
      return null;
    }
  };
  function currentTarget(info) { return target && info.columns.includes(target) ? target : info.target_candidate; }
  useEffect(() => {
    api('/api/health').then(() => setApiConnected(true)).catch(() => setApiConnected(false));
    loadDataset(0, '', { column: '', descending: false });
    loadHistory();
  }, []);
  useEffect(() => { if (dataset) loadHistory(); }, [historySearch, historyFilter]);
  useEffect(() => { setMobileOpen(false); }, [page]);

  const doUpload = async file => {
    if (!file) return;
    setBusy(true);
    try {
      const body = new FormData(); body.append('file', file);
      const info = await api('/api/dataset/upload', { method: 'POST', body });
      setDataset(info); setTarget(info.target_candidate || '');
      setSelectedFeatures(info.columns.filter(column => column !== (info.target_candidate || '')));
      setTrainingResults({}); setModels([]); setPrediction(null); setLstmResult(null); setPreprocessResult(null);
      await loadDataset(0, '', { column: '', descending: false });
      notify(`Loaded ${fmt(info.rows)} records from ${file.name}.`);
    } catch (error) { notify(error.message); }
    finally { setBusy(false); }
  };
  const uploadButton = <label className="button button-primary upload-control"><Upload size={14} />{busy ? 'Loading…' : 'Upload dataset'}<input type="file" accept=".csv,text/csv" onChange={event => doUpload(event.target.files?.[0])} /></label>;
  const refreshPreview = async (offset = 0, query = datasetSearch, sortBy = sort) => {
    if (!dataset) return;
    setPageOffset(offset);
    const params = new URLSearchParams({ offset: String(offset), limit: '25' });
    if (query) params.set('search', query);
    if (sortBy.column) { params.set('sort_by', sortBy.column); params.set('descending', String(sortBy.descending)); }
    try { setPreview(await api(`/api/dataset/preview?${params}`)); }
    catch (error) { notify(error.message); }
  };
  const changeSort = column => {
    const next = { column, descending: sort.column === column ? !sort.descending : false };
    setSort(next); refreshPreview(0, datasetSearch, next);
  };
  const setTrainValue = event => setTrainConfig(current => ({ ...current, [event.target.name]: Number(event.target.value) }));
  const runTrain = async modelName => {
    if (!dataset) return notify('Upload a CSV dataset first.');
    if (!target) return notify('Choose the target attribute first.');
    setBusy(true); setActiveModel(modelName);
    try {
      const result = await api(`/api/train/${modelName}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...trainConfig, target, features: selectedFeatures }) });
      setTrainingResults(current => ({ ...current, [modelName]: result }));
      await loadDataset(pageOffset, datasetSearch, sort);
      notify(`${result.model} trained successfully on ${fmt(result.instances)} labelled records.`);
    } catch (error) { notify(error.message); }
    finally { setBusy(false); }
  };
  const runPrediction = async (modelName, future = false) => {
    const values = {};
    for (const feature of selectedFeatures) {
      const raw = formValues[feature];
      const detail = dataset?.column_details?.find(item => item.name === feature);
      values[feature] = raw === '' || raw === undefined ? null : detail?.kind === 'numeric' ? Number(raw) : raw;
    }
    const request = { model: modelName, features: values, state: future ? futureState || null : values.State || values.state || null, district: future ? futureDistrict || null : null };
    const url = future ? '/api/future-prediction' : '/api/predict';
    if (future) request.future_year = Number(futureYear);
    setBusy(true);
    try {
      const result = await api(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request) });
      setPrediction(result); await loadHistory(); notify(future ? 'Scenario classified by the trained historical model.' : 'Prediction calculated and saved.');
    } catch (error) { notify(error.message); }
    finally { setBusy(false); }
  };
  const runPreprocess = async () => {
    if (!dataset) return notify('Upload a CSV dataset first.');
    setBusy(true);
    try {
      const result = await api('/api/preprocess', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...preprocessOptions, target, features: selectedFeatures }) });
      setPreprocessResult(result);
      const updatedInfo = await loadDataset(0, datasetSearch, sort);
      if (updatedInfo) setSelectedFeatures(updatedInfo.columns.filter(column => column !== target));
      notify('Preprocessing applied to an in-memory copy.');
    } catch (error) { notify(error.message); }
    finally { setBusy(false); }
  };
  const runLstm = async () => {
    setBusy(true);
    try { const result = await api('/api/train/lstm', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(lstmConfig) }); setLstmResult(result); notify('LSTM trained on the uploaded time series.'); }
    catch (error) { setLstmResult({ error: error.message }); notify(error.message); }
    finally { setBusy(false); }
  };
  const deleteHistory = async id => {
    try { await api(`/api/prediction-history/${id}`, { method: 'DELETE' }); await loadHistory(); notify('Prediction removed from history.'); }
    catch (error) { notify(error.message); }
  };
  const handleDatasetSearch = value => { setDatasetSearch(value); refreshPreview(0, value, sort); };
  const navigate = id => { setPage(id); window.scrollTo({ top: 0, behavior: 'smooth' }); };

  const activeResult = trainingResults[page] || trainingResults[activeModel];
  const trainedModelKey = page === 'decision-tree' ? 'decision-tree' : page === 'random-forest' ? 'random-forest' : page === 'xgboost' ? 'xgboost' : activeModel;
  const modelOutput = trainingResults[trainedModelKey];
  const meta = dataset?.column_details || [];
  const severityDistribution = dataset?.target_distribution || [];
  const actualRiskClasses = severityDistribution.filter(item => ['high', 'critical'].includes(String(item.label).toLowerCase()));
  const actualHighCount = actualRiskClasses.reduce((sum, item) => sum + item.count, 0);
  const bestModel = [...models].sort((a, b) => b.metrics.accuracy - a.metrics.accuracy)[0];
  const numericFeature = name => meta.find(item => item.name === name)?.kind === 'numeric';

  const renderTargetSelect = () => <label className="field"><span>Target attribute</span><select value={target} onChange={event => { setTarget(event.target.value); setSelectedFeatures(dataset?.columns.filter(column => column !== event.target.value) || []); }}>{dataset?.columns.map(column => <option key={column} value={column}>{column}</option>)}</select></label>;
  const renderFeaturePicker = () => <div className="feature-picker">{dataset?.columns.filter(column => column !== target).map(column => <label key={column} className="check-row"><input type="checkbox" checked={selectedFeatures.includes(column)} onChange={event => setSelectedFeatures(current => event.target.checked ? [...current, column] : current.filter(item => item !== column))} /><span>{column}</span><small>{meta.find(item => item.name === column)?.kind}</small></label>)}</div>;
  const renderInputFields = () => <div className="prediction-fields">{selectedFeatures.map(feature => {
    const detail = meta.find(item => item.name === feature);
    const known = Object.keys(detail?.frequencies || {});
    return <label className="field" key={feature}><span>{feature}{detail?.kind === 'numeric' ? ' · numeric' : ' · category'}</span>{detail?.kind === 'categorical' && known.length && known.length <= 40 ? <select value={formValues[feature] ?? ''} onChange={event => setFormValues(current => ({ ...current, [feature]: event.target.value }))}><option value="">Select a value</option>{known.filter(value => value !== '<NA>').map(value => <option key={value}>{value}</option>)}</select> : <input type={detail?.kind === 'numeric' ? 'number' : 'text'} step="any" value={formValues[feature] ?? ''} onChange={event => setFormValues(current => ({ ...current, [feature]: event.target.value }))} placeholder={detail?.kind === 'numeric' ? 'Enter a value' : 'Enter category'} />}</label>;
  })}</div>;

  const renderDashboard = () => {
    const historyRows = history.slice(0, 5);
    const modelChart = models.map(item => ({ model: item.model, Accuracy: item.metrics.accuracy * 100, Precision: item.metrics.precision_macro * 100, Recall: item.metrics.recall_macro * 100, 'F1-score': item.metrics.f1_macro * 100 }));
    return <>
      <div className="page-intro"><div><div className="eyebrow">INDIA · HISTORICAL DISASTER INVENTORY</div><h1>Flood Risk Research System</h1><p>Explore the uploaded inventory, run reproducible classifiers and review paper-reported findings.</p></div>{uploadButton}</div>
      {!dataset && <div className="notice notice-warn"><AlertTriangle size={16} /><span>The research CSV is not present in this project yet. Upload <strong>India_Floods_Weka_Reduced(1).csv</strong> to enable dataset analysis and model training.</span></div>}
      <div className="metric-grid dashboard-metrics">
        <Metric label="DATASET RECORDS" value={dataset ? fmt(dataset.rows) : '—'} note={dataset?.dataset_name || 'Upload dataset to inspect'} icon={Database} />
        <Metric label="FEATURES" value={dataset ? fmt(dataset.columns_count) : '—'} note={target ? `Target candidate: ${target}` : 'No target inferred'} icon={Layers3} />
        <Metric label="SEVERITY CLASSES" value={dataset ? fmt(dataset.target_classes?.length || 0) : '—'} note="Observed values only" icon={Activity} />
        <Metric label="BEST LIVE MODEL" value={bestModel?.model || '—'} note={bestModel ? `${percent(bestModel.metrics.accuracy)} test accuracy` : 'Train a classifier to calculate'} icon={BrainCircuit} />
        <Metric label="HIGH / CRITICAL EVENTS" value={dataset ? fmt(actualHighCount) : '—'} note={actualRiskClasses.length ? 'Exact target labels High/Critical' : 'No ordered mapping inferred from labels'} icon={ShieldAlert} tone="metric-accent" />
        <Metric label="PREDICTIONS STORED" value={fmt(historyTotal)} note="Saved to local SQLite history" icon={ListFilter} />
      </div>
      <div className="dashboard-grid">
        <Panel title="Observed target distribution" eyebrow="CALCULATED FROM UPLOADED DATASET" actions={dataset ? <Badge kind="green">Live dataset</Badge> : <Badge>No data</Badge>}>
          {dataset ? <div className="chart-area"><ResponsiveContainer width="100%" height="100%"><BarChart data={severityDistribution} margin={{ top: 12, right: 10, left: -18, bottom: 4 }}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="label" tick={{ fontSize: 10 }} /><YAxis tick={{ fontSize: 9 }} /><Tooltip content={<ChartTooltip />} /><Bar dataKey="count" name="Records" radius={[3, 3, 0, 0]}>{severityDistribution.map(item => <Cell key={item.label} fill={colorFor(item.label)} />)}</Bar></BarChart></ResponsiveContainer></div> : <div className="chart-empty">Upload a dataset to compute its target class distribution.</div>}
        </Panel>
        <Panel title="Classifier comparison" eyebrow="CALCULATED AFTER TRAINING" actions={<Badge kind="green">Live results</Badge>}>
          {modelChart.length ? <div className="chart-area"><ResponsiveContainer width="100%" height="100%"><BarChart data={modelChart} margin={{ top: 12, right: 8, left: -18, bottom: 2 }}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="model" tick={{ fontSize: 9 }} /><YAxis domain={[0, 100]} tick={{ fontSize: 9 }} /><Tooltip content={<ChartTooltip />} /><Legend wrapperStyle={{ fontSize: 9 }} /><Bar dataKey="Accuracy" fill="#38795e" radius={[3, 3, 0, 0]} /><Bar dataKey="F1-score" fill="#d18a56" radius={[3, 3, 0, 0]} /></BarChart></ResponsiveContainer></div> : <div className="chart-empty">Live comparison appears after training at least one classifier.</div>}
        </Panel>
      </div>
      {dataset?.year_distribution?.length > 0 && <Panel title="Flood events by year" eyebrow="CALCULATED FROM ACTUAL DATE / YEAR COLUMN" actions={<Badge kind="green">Live dataset</Badge>}><div className="chart-area"><ResponsiveContainer width="100%" height="100%"><AreaChart data={dataset.year_distribution}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="year" tick={{ fontSize: 9 }} /><YAxis tick={{ fontSize: 9 }} /><Tooltip content={<ChartTooltip />} /><Area type="monotone" dataKey="count" name="Events" stroke="#477d63" fill="#dcebdd" /></AreaChart></ResponsiveContainer></div></Panel>}
      {dataset?.location_distribution?.length > 0 && <Panel title={`Events by ${dataset.geographic_columns.state || 'district'}`} eyebrow="TOP LOCATIONS IN THE UPLOADED DATASET" actions={<Badge kind="green">Live dataset</Badge>}><div className="chart-area chart-tall"><ResponsiveContainer width="100%" height="100%"><BarChart data={dataset.location_distribution.slice(0, 15)} layout="vertical" margin={{ left: 65, right: 16 }}><CartesianGrid strokeDasharray="3 3" horizontal={false} /><XAxis type="number" tick={{ fontSize: 9 }} /><YAxis type="category" dataKey="location" width={75} tick={{ fontSize: 8 }} /><Tooltip content={<ChartTooltip />} /><Bar dataKey="count" name="Events" fill="#6d9270" radius={[0, 3, 3, 0]} /></BarChart></ResponsiveContainer></div></Panel>}
      <Panel title="Recent predictions" eyebrow="PERSISTED IN SQLITE" actions={<button className="text-button" onClick={() => navigate('history')}>View history <ChevronRight size={13} /></button>}>
        <Table columns={[{ key: 'timestamp', label: 'TIME', render: value => value ? new Date(value).toLocaleString() : '—' }, { key: 'model', label: 'MODEL' }, { key: 'predicted_severity', label: 'PREDICTED CLASS' }, { key: 'probability', label: 'CONFIDENCE', render: value => percent(value) }, { key: 'risk_level', label: 'RISK MAPPING' }]} rows={historyRows} empty="Predictions will appear after you train a model and submit a scenario." />
      </Panel>
      <Panel title="Service status" eyebrow="APPLICATION HEALTH"><div className="service-grid"><StatusItem label="Dataset" ok={Boolean(dataset)} detail={dataset ? 'Loaded in API session' : 'Upload required'} /><StatusItem label="Classifiers" ok={models.length > 0} detail={models.length ? `${models.length} trained` : 'Not trained'} /><StatusItem label="Prediction service" ok detail="API online" /><StatusItem label="LSTM forecasting" ok={Boolean(dataset?.time_series?.suitable)} detail={dataset?.time_series?.suitable ? 'Chronology candidate detected' : 'Unavailable until suitable series is uploaded'} /></div></Panel>
      <Panel title="Reported benchmark" eyebrow="RESEARCH PAPER · NOT A LIVE MODEL RESULT" actions={<Badge kind="paper">Reported in Research Paper</Badge>}><div className="chart-area chart-short"><ResponsiveContainer width="100%" height="100%"><BarChart data={PAPER_RESULTS} margin={{ top: 8, right: 12, left: -18, bottom: 0 }}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="model" tick={{ fontSize: 10 }} /><YAxis domain={[0, 100]} tick={{ fontSize: 9 }} /><Tooltip content={<ChartTooltip />} /><Bar dataKey="accuracy" name="Accuracy (%)" fill="#73825f" radius={[3, 3, 0, 0]} /></BarChart></ResponsiveContainer></div></Panel>
    </>;
  };

  const renderDataset = () => !dataset ? <EmptyState onUpload={() => document.querySelector('#datasetUpload')?.click()}>Upload the primary CSV to inspect rows, columns, missing values, duplicates, data types, statistics, and target candidates. The original file is never edited.</EmptyState> : <>
    <div className="page-intro"><div><div className="eyebrow">DATASET EXPLORER · WEKA-STYLE ATTRIBUTE VIEW</div><h1>Dataset</h1><p>Schema and statistics are calculated from the uploaded CSV by the Python API.</p></div>{uploadButton}</div>
    <div className="metric-grid four-metrics"><Metric label="ROWS" value={fmt(dataset.rows)} note="Uploaded records" icon={Database} /><Metric label="ATTRIBUTES" value={fmt(dataset.columns_count)} note={`${dataset.numerical_columns.length} numeric · ${dataset.categorical_columns.length} categorical`} icon={Table2} /><Metric label="MISSING CELLS" value={fmt(dataset.missing_cells)} note={`${fmt(dataset.rows_with_missing)} rows contain missing values`} icon={CircleHelp} /><Metric label="DUPLICATES" value={fmt(dataset.duplicate_rows)} note="Exact duplicate records" icon={Layers3} /></div>
    <Panel title="Dataset details" eyebrow="SOURCE: UPLOADED FILE" actions={<div className="panel-actions"><Badge kind="green">{dataset.dataset_name}</Badge><a className="icon-button" href="/api/dataset/download" title="Download current dataset"><ArrowDownToLine size={15} /></a></div>}><div className="details-grid"><div><small>Rows × columns</small><strong>{fmt(dataset.rows)} × {fmt(dataset.columns_count)}</strong></div><div><small>Numerical attributes</small><strong>{fmt(dataset.numerical_columns.length)}</strong></div><div><small>Categorical attributes</small><strong>{fmt(dataset.categorical_columns.length)}</strong></div><div><small>Target candidate</small><strong>{dataset.target_candidate || 'Not detected; choose manually'}</strong></div><div><small>Geographic fields</small><strong>{Object.values(dataset.geographic_columns).filter(Boolean).join(', ') || 'Not available'}</strong></div><div><small>Year/date fields</small><strong>{dataset.temporal_columns.map(item => item.column).join(', ') || 'Not available'}</strong></div></div></Panel>
    <Panel title="Instances" eyebrow="SORT, SEARCH AND PAGE THROUGH SOURCE DATA" actions={<div className="table-controls"><label className="search-field"><Search size={14} /><input value={datasetSearch} onChange={event => handleDatasetSearch(event.target.value)} placeholder="Search all fields" /></label><a href="/api/dataset/download" className="icon-button" title="Download dataset"><ArrowDownToLine size={15} /></a></div>}>
      <div className="table-scroll"><table><thead><tr><th>#</th>{preview.columns.map(column => <th key={column} onClick={() => changeSort(column)} className="sortable">{column}{sort.column === column ? (sort.descending ? ' ↓' : ' ↑') : ''}</th>)}</tr></thead><tbody>{preview.records.length ? preview.records.map((record, index) => <tr key={index}><td className="row-index">{pageOffset + index + 1}</td>{preview.columns.map(column => <td key={column}>{record[column] === null || record[column] === undefined ? <span className="missing-value">?</span> : String(record[column])}</td>)}</tr>) : <tr><td colSpan={preview.columns.length + 1} className="empty-row">No matching rows.</td></tr>}</tbody></table></div>
      <div className="pagination"><span>{fmt(preview.filtered_rows)} matching · {fmt(preview.total_rows)} total</span><div><button className="icon-button" disabled={pageOffset <= 0} onClick={() => refreshPreview(Math.max(0, pageOffset - 25))}><ChevronLeft size={15} /></button><span>{Math.floor(pageOffset / 25) + 1} / {Math.max(1, Math.ceil(preview.filtered_rows / 25))}</span><button className="icon-button" disabled={pageOffset + 25 >= preview.filtered_rows} onClick={() => refreshPreview(pageOffset + 25)}><ChevronRight size={15} /></button></div></div>
    </Panel>
    <Panel title="Attribute profile" eyebrow="DISTRIBUTIONS AND DESCRIPTIVE STATISTICS" actions={<select value={attribute} onChange={event => setAttribute(event.target.value)}>{dataset.column_details.map(column => <option key={column.name}>{column.name}</option>)}</select>}><AttributeProfile column={dataset.column_details.find(item => item.name === attribute)} /></Panel>
  </>;

  const renderPreprocessing = () => !dataset ? <EmptyState onUpload={() => document.querySelector('#datasetUpload')?.click()} /> : <>
    <div className="page-intro"><div><div className="eyebrow">IN-MEMORY TRANSFORMATIONS · BACKEND EXECUTED</div><h1>Preprocessing</h1><p>Each setting is applied by Python to a working copy. The uploaded CSV is left untouched.</p></div></div>
    <Panel title="Workflow options" eyebrow="DATA CLEANING AND ENCODING"><div className="options-grid"><label className="toggle-option"><input type="checkbox" checked={preprocessOptions.drop_duplicates} onChange={event => setPreprocessOptions(current => ({ ...current, drop_duplicates: event.target.checked }))} /><span><strong>Remove exact duplicates</strong><small>Drop duplicate rows in the working copy.</small></span></label><label className="toggle-option"><input type="checkbox" checked={preprocessOptions.encode_categoricals} onChange={event => setPreprocessOptions(current => ({ ...current, encode_categoricals: event.target.checked }))} /><span><strong>One-hot encode categorical inputs</strong><small>Keep the selected target label unchanged.</small></span></label><label className="toggle-option"><input type="checkbox" checked={preprocessOptions.normalize_numeric} onChange={event => setPreprocessOptions(current => ({ ...current, normalize_numeric: event.target.checked }))} /><span><strong>Normalize numeric inputs</strong><small>Apply z-score scaling to numeric features.</small></span></label><label className="field"><span>Missing values</span><select value={preprocessOptions.missing_strategy} onChange={event => setPreprocessOptions(current => ({ ...current, missing_strategy: event.target.value }))}><option value="impute">Median / most frequent imputation</option><option value="drop">Drop rows with any missing value</option><option value="keep">Keep missing values</option></select></label></div><div className="form-row">{renderTargetSelect()}<button className="button button-primary" disabled={busy} onClick={runPreprocess}><SlidersHorizontal size={14} />{busy ? 'Applying…' : 'Apply preprocessing'}</button></div></Panel>
    <Panel title="Before and after" eyebrow="SERVER-CALCULATED COUNTS">{preprocessResult ? <div className="compare-stats"><StatsColumn title="Original dataset" stats={preprocessResult.original} /><ChevronRight className="compare-arrow" /><StatsColumn title="Processed working copy" stats={preprocessResult.processed} /></div> : <div className="chart-empty">Apply a preprocessing configuration to calculate the actual before/after counts.</div>}{preprocessResult && <p className="footnote">{preprocessResult.note}</p>}</Panel>
  </>;

  const renderFeatures = () => !dataset ? <EmptyState onUpload={() => document.querySelector('#datasetUpload')?.click()} /> : <>
    <div className="page-intro"><div><div className="eyebrow">COLUMN MAPPING FROM ACTUAL HEADERS</div><h1>Feature Engineering</h1><p>Feature concepts below are matched to uploaded column names; no absent columns are synthesized.</p></div></div>
    <div className="feature-layout"><Panel title="Choose class and inputs" eyebrow="MODEL DESIGN"><div className="form-stack">{renderTargetSelect()}<div><div className="field-title">Predictor attributes</div>{renderFeaturePicker()}</div></div><div className="feature-summary"><span>{selectedFeatures.length} predictors selected</span><span>Target: {target || 'not selected'}</span></div></Panel><Panel title="Research concept matches" eyebrow="NAME-BASED, NOT A CLAIM OF SEMANTIC VALIDATION"><ConceptMatches /></Panel></div>
  </>;

  const renderModel = modelName => {
    const result = trainingResults[modelName];
    const labels = { 'decision-tree': 'Decision Tree', 'random-forest': 'Random Forest', xgboost: 'XGBoost' };
    return <>
      <div className="page-intro"><div><div className="eyebrow">WEKA-STYLE CLASSIFICATION WORKBENCH</div><h1>{labels[modelName]}</h1><p>Train and evaluate against a reproducible holdout split of the uploaded, labeled dataset.</p></div><Badge kind={result ? 'green' : 'neutral'}>{result ? 'Model trained successfully' : 'Not trained'}</Badge></div>
      {!dataset ? <EmptyState onUpload={() => document.querySelector('#datasetUpload')?.click()} /> : <>
        <Panel title="Classifier configuration" eyebrow="TRAIN / TEST SETTINGS" actions={<Badge kind="green">Backend execution</Badge>}>
          <div className="model-config">{renderTargetSelect()}<label className="field"><span>Test set fraction</span><select name="test_size" value={trainConfig.test_size} onChange={setTrainValue}><option value="0.2">20%</option><option value="0.25">25%</option><option value="0.3">30%</option></select></label><label className="field"><span>Maximum tree depth</span><input type="number" min="1" max="40" value={trainConfig.max_depth} name="max_depth" onChange={setTrainValue} /></label>{modelName !== 'decision-tree' && <label className="field"><span>Number of estimators</span><input type="number" min="10" max="2000" step="10" value={trainConfig.n_estimators} name="n_estimators" onChange={setTrainValue} /></label>}{modelName === 'xgboost' && <><label className="field"><span>Learning rate</span><input type="number" min="0.001" max="1" step="0.01" value={trainConfig.learning_rate} name="learning_rate" onChange={setTrainValue} /></label><label className="field"><span>Subsample</span><input type="number" min="0.1" max="1" step="0.1" value={trainConfig.subsample} name="subsample" onChange={setTrainValue} /></label><label className="field"><span>Column sample</span><input type="number" min="0.1" max="1" step="0.1" value={trainConfig.colsample_bytree} name="colsample_bytree" onChange={setTrainValue} /></label></>}</div>
          {modelName === 'xgboost' && <div className="notice notice-info"><Zap size={15} />Research defaults: 200 estimators · 0.05 learning rate · depth 6 · 0.8 subsample · 0.8 column sample · seed 42.</div>}
          <div className="run-row"><span>{selectedFeatures.length} input attributes · target {target || 'not selected'}</span><button className="button button-primary" disabled={busy || !target} onClick={() => runTrain(modelName)}><Play size={13} />{busy ? 'Training…' : `Run ${labels[modelName]}`}</button></div>
        </Panel>
        {result ? <TrainingResults result={result} /> : <div className="results-placeholder"><Network size={19} /><span>Run the classifier to calculate metrics, confusion matrix, report, feature importance and test-row predictions.</span></div>}
      </>}
    </>;
  };

  const renderComparison = () => <>
    <div className="page-intro"><div><div className="eyebrow">LIVE RESULTS ARE KEPT SEPARATE FROM PAPER VALUES</div><h1>Model Comparison</h1><p>Calculated model values appear only after training against the uploaded CSV.</p></div></div>
    <SourceHeading calculated>Trained classifiers</SourceHeading>
    <Table columns={[{ key: 'model', label: 'MODEL' }, { key: 'training_accuracy', label: 'TRAIN ACCURACY', render: percent }, { key: 'accuracy', label: 'TEST ACCURACY', render: percent }, { key: 'precision_macro', label: 'MACRO PRECISION', render: value => fmt(value, 4) }, { key: 'recall_macro', label: 'MACRO RECALL', render: value => fmt(value, 4) }, { key: 'f1_macro', label: 'MACRO F1', render: value => fmt(value, 4) }]} rows={models} empty="Train Decision Tree, Random Forest, or XGBoost to populate this comparison." />
    {models.length > 0 && <div className="panel chart-panel"><div className="chart-area"><ResponsiveContainer width="100%" height="100%"><BarChart data={models.map(item => ({ model: item.model, Accuracy: item.metrics.accuracy * 100, Precision: item.metrics.precision_macro * 100, Recall: item.metrics.recall_macro * 100, F1: item.metrics.f1_macro * 100 }))}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="model" tick={{ fontSize: 9 }} /><YAxis domain={[0, 100]} /><Tooltip content={<ChartTooltip />} /><Legend /><Bar dataKey="Accuracy" fill="#397e62" /><Bar dataKey="Precision" fill="#62849b" /><Bar dataKey="Recall" fill="#d18a56" /><Bar dataKey="F1" fill="#9b7890" /></BarChart></ResponsiveContainer></div></div>}
    <SourceHeading>Reported benchmark</SourceHeading><PaperResultsTable />
  </>;

  const renderPrediction = future => {
    const modelOptions = models.map(item => <option key={item.model} value={item.model === 'Decision Tree' ? 'decision-tree' : item.model === 'Random Forest' ? 'random-forest' : 'xgboost'}>{item.model}</option>);
    return <>
      <div className="page-intro"><div><div className="eyebrow">{future ? 'SCENARIO CLASSIFICATION · NOT YEAR EXTRAPOLATION' : 'INDIVIDUAL CLASSIFICATION'}</div><h1>{future ? 'Future Flood Risk Prediction' : 'Flood Prediction'}</h1><p>{future ? 'A future year is metadata only unless the training features contain a validated time signal. No future values are invented.' : 'Enter values for supported dataset attributes. The selected trained pipeline imputes missing fields and classifies the scenario.'}</p></div></div>
      {!dataset ? <EmptyState onUpload={() => document.querySelector('#datasetUpload')?.click()} /> : models.length === 0 ? <div className="notice notice-warn"><AlertTriangle size={16} />Train a classifier from the dataset before making predictions.</div> : <div className="prediction-layout"><Panel title="Scenario inputs" eyebrow="ONLY UPLOADED FEATURES ARE SHOWN"><div className="form-stack">{future && <div className="future-topline"><label className="field"><span>Future year</span><input type="number" min={new Date().getFullYear() + 1} value={futureYear} onChange={event => setFutureYear(Number(event.target.value))} /></label>{dataset.geographic_columns.state && <label className="field"><span>State (display context)</span><input value={futureState} onChange={event => setFutureState(event.target.value)} placeholder="Optional" /></label>}{dataset.geographic_columns.district && <label className="field"><span>District (display context)</span><input value={futureDistrict} onChange={event => setFutureDistrict(event.target.value)} placeholder="Optional" /></label>}</div>}<label className="field"><span>Trained model</span><select value={activeModel} onChange={event => setActiveModel(event.target.value)}>{modelOptions}</select></label>{renderInputFields()}<button className="button button-primary prediction-submit" disabled={busy} onClick={() => runPrediction(activeModel, future)}>{busy ? 'Calculating…' : future ? 'Predict future-year scenario' : 'Predict flood class'}<ChevronRight size={15} /></button></div></Panel><div className="prediction-result">{prediction ? <PredictionResult prediction={prediction} future={future} /> : <div className="result-await"><BrainCircuit size={25} /><strong>Prediction result</strong><span>Submit scenario inputs to run the trained model. Results are saved to prediction history.</span></div>}<div className="notice notice-warn compact"><ShieldAlert size={14} />Research decision support only. Not an official warning.</div></div></div>}
    </>;
  };

  const renderLstm = () => <>
    <div className="page-intro"><div><div className="eyebrow">CHRONOLOGY VALIDATED BEFORE TRAINING</div><h1>LSTM Forecasting</h1><p>Forecast numeric rainfall only when the uploaded data contains enough genuine chronological observations.</p></div></div>
    {!dataset ? <EmptyState onUpload={() => document.querySelector('#datasetUpload')?.click()} /> : <>
      <div className={`notice ${dataset.time_series.suitable ? 'notice-success' : 'notice-warn'}`}><Activity size={16} /><span>{dataset.time_series.suitable ? dataset.time_series.reason : `Current dataset does not contain sufficient chronological time-series information for genuine LSTM forecasting. ${dataset.time_series.reason}`}</span></div>
      <Panel title="Sequence eligibility" eyebrow="DATASET INSPECTION"><div className="details-grid"><div><small>Candidate time fields</small><strong>{dataset.time_series.time_columns.map(item => `${item.column} (${fmt(item.unique_timestamps)} unique)`).join(', ') || 'None found'}</strong></div><div><small>Numeric rainfall / flow fields</small><strong>{dataset.time_series.rainfall_columns.join(', ') || 'None found'}</strong></div><div><small>Minimum sequence check</small><strong>30 observations · 20 unique timestamps</strong></div><div><small>Training runtime</small><strong>PyTorch LSTM · Adam optimizer</strong></div></div></Panel>
      {dataset.time_series.suitable && <Panel title="LSTM settings" eyebrow="64 UNITS · ADAM · 50 EPOCHS DEFAULT"><div className="model-config"><label className="field"><span>Lookback steps</span><input type="number" min="2" max="60" value={lstmConfig.lookback} onChange={event => setLstmConfig(current => ({ ...current, lookback: Number(event.target.value) }))} /></label><label className="field"><span>Forecast horizon</span><input type="number" min="1" max="30" value={lstmConfig.horizon} onChange={event => setLstmConfig(current => ({ ...current, horizon: Number(event.target.value) }))} /></label><label className="field"><span>Epochs</span><input type="number" min="1" max="300" value={lstmConfig.epochs} onChange={event => setLstmConfig(current => ({ ...current, epochs: Number(event.target.value) }))} /></label><label className="field"><span>Hidden units</span><input type="number" min="4" max="256" value={lstmConfig.units} onChange={event => setLstmConfig(current => ({ ...current, units: Number(event.target.value) }))} /></label></div><button className="button button-primary" disabled={busy} onClick={runLstm}><Play size={13} />{busy ? 'Training LSTM…' : 'Train rainfall LSTM'}</button></Panel>}
      {lstmResult && !lstmResult.error && <LstmResults result={lstmResult} />}
      {lstmResult?.error && <div className="notice notice-warn"><AlertTriangle size={15} />{lstmResult.error}</div>}
    </>}
  </>;

  const renderMap = () => !dataset ? <EmptyState onUpload={() => document.querySelector('#datasetUpload')?.click()} /> : <>
    <div className="page-intro"><div><div className="eyebrow">DATASET-DERIVED GEOGRAPHY ONLY</div><h1>Flood Risk Map</h1><p>Coordinate markers retain the original target labels; no severity ordering is inferred.</p></div>{mapData && <Badge>{mapData.mode === 'coordinates' ? 'Live dataset coordinates' : 'Administrative data'}</Badge>}</div>
    {mapData?.mode === 'coordinates' && mapData.points.some(point => Number.isFinite(Number(point.latitude)) && Number.isFinite(Number(point.longitude))) ? <div className="map-frame"><MapContainer center={centerFor(mapData.points)} zoom={5} scrollWheelZoom className="leaflet-map"><TileLayer attribution="&copy; OpenStreetMap contributors" url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />{mapData.points.filter(point => Number.isFinite(Number(point.latitude)) && Number.isFinite(Number(point.longitude))).map((point, index) => <CircleMarker key={index} center={[Number(point.latitude), Number(point.longitude)]} radius={5} pathOptions={{ color: colorFor(point.label || 'unlabelled'), fillColor: colorFor(point.label || 'unlabelled'), fillOpacity: .7 }}><Popup><strong>{point.label || 'No target label'}</strong><br />{point.latitude}, {point.longitude}</Popup></CircleMarker>)}</MapContainer><div className="map-legend">{dataset.target_classes.map(label => <span key={label}><i style={{ background: colorFor(label) }} />{label}</span>)}</div></div> : mapData?.mode === 'administrative' ? <Panel title="Events by administrative area" eyebrow="NO COORDINATES INFERRED"><Table columns={[{ key: 'location', label: 'STATE / DISTRICT' }, { key: 'events', label: 'EVENT RECORDS' }]} rows={mapData.groups} /></Panel> : <EmptyState title="Geographic visualization unavailable">{mapData?.mode === 'coordinates' ? 'Coordinate columns exist, but no valid coordinate pairs were found in the dataset.' : mapData?.message || 'No usable geographic columns are available.'}</EmptyState>}
    <div className="notice notice-info map-note"><MapPinned size={15} />Research-paper map claims and uploaded-dataset geography are separate. The paper visualization is not substituted for missing dataset coordinates.</div>
  </>;

  const renderWarning = () => <>
    <div className="page-intro"><div><div className="eyebrow">APPLICATION-GENERATED · NOT OFFICIAL</div><h1>Early Warning</h1><p>Recommendations are generated only from an explicit model prediction and a known source label mapping.</p></div></div>
    {prediction ? <PredictionResult prediction={prediction} future={Boolean(prediction.future_year)} /> : <div className="empty-state"><span className="empty-icon"><Bell size={21} /></span><h2>No recent prediction</h2><p>Run a supported scenario through Flood Prediction before displaying an action recommendation.</p><button className="button button-primary" onClick={() => navigate('prediction')}>Open flood prediction</button></div>}
    <div className="notice notice-warn warning-disclaimer"><AlertTriangle size={16} />Labels such as S1/S2 are not automatically mapped to Low/Medium/High/Critical. Follow official government alerts and local disaster-management guidance.</div>
  </>;

  const renderPaperPage = section => {
    const definitions = {
      research: { title: 'Research Results', intro: 'Published metrics transcribed from the supplied research paper. These are not computed from an uploaded dataset.' },
      'case-study': { title: 'Assam Floods 2015', intro: 'Case study values reproduced exactly as reported in the research paper.' },
      ablation: { title: 'Ablation Study', intro: 'Reported effect of removing selected feature groups; values are not recomputed here.' },
      performance: { title: 'System Performance', intro: 'Latency values reported by the paper. Measurement units in the source need clarification.' },
      usability: { title: 'Usability Study', intro: 'Participant metrics reported in the paper (n = 30). These are not live user-study results.' },
    };
    const info = definitions[section];
    return <><div className="page-intro"><div><div className="eyebrow">FIXED PAPER VALUES · SEPARATE FROM LIVE RESULTS</div><h1>{info.title}</h1><p>{info.intro}</p></div><Badge kind="paper">Reported in Research Paper</Badge></div>
      {section === 'research' && <><PaperResultsTable /><Panel title="Training vs validation accuracy" eyebrow="REPORTED IN RESEARCH PAPER"><div className="chart-area"><ResponsiveContainer width="100%" height="100%"><BarChart data={PAPER_TRAINING}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="model" tick={{ fontSize: 9 }} /><YAxis domain={[0, 100]} /><Tooltip content={<ChartTooltip />} /><Legend /><Bar dataKey="training_accuracy" name="Training (%)" fill="#9ab082" /><Bar dataKey="validation_accuracy" name="Validation (%)" fill="#477d63" /></BarChart></ResponsiveContainer></div></Panel></>}
      {section === 'case-study' && <Panel title="Reported and predicted values" eyebrow="ASSAM FLOODS · 2015"><Table columns={[{ key: 'attribute', label: 'ATTRIBUTE' }, { key: 'reported_value', label: 'REPORTED VALUE' }, { key: 'prediction_value', label: 'PREDICTION VALUE' }]} rows={PAPER_CASE} /></Panel>}
      {section === 'ablation' && <Panel title="Feature ablation results" eyebrow="VALIDATION ACCURACY"><Table columns={[{ key: 'configuration', label: 'CONFIGURATION' }, { key: 'accuracy', label: 'ACCURACY (%)' }, { key: 'latency', label: 'LATENCY (AS REPORTED)' }]} rows={PAPER_ABLATION.map(item => ({ ...item, accuracy: fmt(item.accuracy, 2), latency: item.latency.toFixed(6) }))} /><div className="chart-area chart-panel"><ResponsiveContainer width="100%" height="100%"><BarChart data={PAPER_ABLATION}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="configuration" tick={{ fontSize: 8 }} interval={0} angle={-10} textAnchor="end" height={45} /><YAxis domain={[0, 100]} /><Tooltip content={<ChartTooltip />} /><Bar dataKey="accuracy" name="Validation accuracy (%)" fill="#527f68" /></BarChart></ResponsiveContainer></div><div className="notice notice-info">Removing Human Displaced has the largest reported accuracy decrease: 12.66 percentage points.</div></Panel>}
      {section === 'performance' && <Panel title="End-to-end latency measurements" eyebrow="UNITS AS WRITTEN IN THE PAPER"><Table columns={[{ key: 'component', label: 'PIPELINE COMPONENT' }, { key: 'seconds', label: 'REPORTED VALUE' }]} rows={PAPER_LATENCY.map(item => ({ ...item, seconds: item.seconds }))} /><div className="chart-area chart-panel"><ResponsiveContainer width="100%" height="100%"><BarChart data={PAPER_LATENCY}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="component" tick={{ fontSize: 8 }} /><YAxis /><Tooltip content={<ChartTooltip />} /><Bar dataKey="seconds" name="Reported value" fill="#5988a0" /></BarChart></ResponsiveContainer></div><p className="footnote">The paper's table reports 1.125, 0.496, and 1.668, while its prose later gives sub-millisecond values. Units and values are inconsistent in the source; no conversion is assumed.</p></Panel>}
      {section === 'usability' && <Panel title="Participant outcomes" eyebrow="N = 30 PARTICIPANTS"><Table columns={[{ key: 'metric', label: 'METRIC' }, { key: 'mean', label: 'MEAN' }, { key: 'sd', label: 'STANDARD DEVIATION' }, { key: 'unit', label: 'UNIT' }]} rows={PAPER_USABILITY.map(item => ({ ...item, mean: item.metric === 'Net Promoter Score' ? '+42' : item.mean, sd: item.sd ?? '—' }))} /></Panel>}
    </>;
  };

  const renderHistory = () => <>
    <div className="page-intro"><div><div className="eyebrow">PERSISTENT LOCAL SQLITE STORE</div><h1>Prediction History</h1><p>Every live prediction is stored with timestamp, inputs, model, class, probability and mapped action.</p></div><a className="button button-secondary" href="/api/prediction-history.csv"><ArrowDownToLine size={14} />Export CSV</a></div>
    <Panel title="Saved predictions" eyebrow="SEARCH / FILTER / DELETE"><div className="table-controls history-controls"><label className="search-field"><Search size={14} /><input value={historySearch} onChange={event => setHistorySearch(event.target.value)} placeholder="Search label or location" /></label><select value={historyFilter} onChange={event => setHistoryFilter(event.target.value)}><option value="">All models</option><option value="decision-tree">Decision Tree</option><option value="random-forest">Random Forest</option><option value="xgboost">XGBoost</option></select></div><Table columns={[{ key: 'timestamp', label: 'TIMESTAMP', render: value => new Date(value).toLocaleString() }, { key: 'model', label: 'MODEL' }, { key: 'state', label: 'STATE' }, { key: 'district', label: 'DISTRICT' }, { key: 'future_year', label: 'SCENARIO YEAR' }, { key: 'predicted_severity', label: 'PREDICTED' }, { key: 'probability', label: 'PROBABILITY', render: percent }, { key: 'risk_level', label: 'RISK LEVEL' }, { key: 'id', label: 'ACTION', render: (id) => <button className="icon-button delete-button" title="Delete prediction" onClick={() => deleteHistory(id)}><Trash2 size={14} /></button> }]} rows={history} empty="No predictions match this filter." /></Panel>
  </>;

  const renderPage = () => {
    if (page === 'dashboard') return renderDashboard();
    if (page === 'dataset') return renderDataset();
    if (page === 'preprocessing') return renderPreprocessing();
    if (page === 'features') return renderFeatures();
    if (['decision-tree', 'random-forest', 'xgboost'].includes(page)) return renderModel(page);
    if (page === 'lstm') return renderLstm();
    if (page === 'comparison') return renderComparison();
    if (page === 'prediction') return renderPrediction(false);
    if (page === 'future') return renderPrediction(true);
    if (page === 'map') return renderMap();
    if (page === 'warning') return renderWarning();
    if (['research', 'case-study', 'ablation', 'performance', 'usability'].includes(page)) return renderPaperPage(page);
    if (page === 'history') return renderHistory();
    return renderDashboard();
  };

  return <div className="app-shell">
    <aside className={`sidebar ${mobileOpen ? 'sidebar-open' : ''}`}>
      <a href="#dashboard" className="brand" onClick={event => { event.preventDefault(); navigate('dashboard'); }}><span className="brand-mark"><Waves size={17} /></span><span>floodline<span className="brand-period">.</span></span></a>
      <div className="brand-subtitle">AI FLOOD RESEARCH SYSTEM</div>
      <nav aria-label="Research application navigation">{NAV.map(group => <div className="nav-group" key={group.group}><div className="nav-caption">{group.group}</div>{group.items.map(([id, label, Icon]) => <button key={id} className={`nav-link ${page === id ? 'active' : ''}`} onClick={() => navigate(id)}><Icon size={15} strokeWidth={1.7} /><span>{label}</span>{id === 'history' && history.length > 0 && <small>{history.length}</small>}</button>)}</div>)}</nav>
      <div className="sidebar-bottom"><div className="source-chip"><span className={`source-dot ${dataset ? 'online' : ''}`} /><div><strong>{dataset ? dataset.dataset_name : 'No dataset loaded'}</strong><small>{dataset ? `${fmt(dataset.rows)} records · API session` : 'Upload a CSV to begin'}</small></div></div><div className="sidebar-version">RESEARCH PROTOTYPE <span>·</span> 1.0</div></div>
    </aside>
    {mobileOpen && <button aria-label="Close navigation" className="mobile-scrim" onClick={() => setMobileOpen(false)} />}
    <main className="main-content">
      <header className="topbar"><button className="mobile-menu icon-button" onClick={() => setMobileOpen(value => !value)} aria-label="Toggle navigation">{mobileOpen ? <X size={17} /> : <Menu size={17} />}</button><div className="breadcrumbs"><span>Flood risk research</span><b>/</b><strong>{TITLES[page]}</strong></div><div className="topbar-right"><span className={`api-status ${apiConnected ? '' : 'api-offline'}`}><i style={{ background: apiConnected ? '#78a875' : '#bf7858' }} />{apiConnected ? 'API connected' : 'API offline'}</span><label className="button button-quiet top-upload"><Upload size={13} />Upload<input type="file" accept=".csv,text/csv" onChange={event => doUpload(event.target.files?.[0])} /></label></div></header>
      <div className="content-wrap">{renderPage()}</div>
      <footer className="page-footer"><span>FLOODLINE <i>·</i> RESEARCH PROTOTYPE</span><span>Model outputs are not official flood warnings.</span></footer>
    </main>
    <input id="datasetUpload" type="file" accept=".csv,text/csv" hidden onChange={event => doUpload(event.target.files?.[0])} />
    {toast && <div className="toast" role="status">{toast}<button aria-label="Dismiss notification" onClick={() => setToast('')}><X size={13} /></button></div>}
  </div>;
}

function StatusItem({ label, ok, detail }) { return <div className="service-item"><span className={`service-dot ${ok ? 'service-ok' : ''}`} /><div><strong>{label}</strong><small>{detail}</small></div></div>; }
function StatsColumn({ title, stats }) { return <div className="stats-column"><h3>{title}</h3><div><span>Rows</span><strong>{fmt(stats.rows)}</strong></div><div><span>Missing cells</span><strong>{fmt(stats.missing_cells)}</strong></div><div><span>Duplicate rows</span><strong>{fmt(stats.duplicate_rows)}</strong></div>{stats.columns && <div><span>Columns</span><strong>{fmt(stats.columns.length)}</strong></div>}</div>; }
function AttributeProfile({ column }) {
  if (!column) return <div className="chart-empty">Choose an attribute to inspect its statistics.</div>;
  if (column.kind === 'numeric') return <div className="details-grid stats-grid">{Object.entries(column.statistics || {}).map(([key, value]) => <div key={key}><small>{key.replaceAll('_', ' ')}</small><strong>{fmt(value, 4)}</strong></div>)}<div><small>Missing values</small><strong>{fmt(column.missing)}</strong></div><div><small>Unique values</small><strong>{fmt(column.unique)}</strong></div></div>;
  const frequencies = Object.entries(column.frequencies || {}).slice(0, 15);
  return <div className="frequency-list">{frequencies.map(([value, count]) => <div key={value}><span>{value === '<NA>' ? 'Missing' : value}</span><b>{fmt(count)}</b><i style={{ width: `${Math.max(3, count / Math.max(...frequencies.map(item => item[1])) * 100)}%` }} /></div>)}<small>{fmt(column.unique)} unique values · {fmt(column.missing)} missing</small></div>;
}
function ConceptMatches() {
  const [matches, setMatches] = useState(null);
  const [error, setError] = useState('');
  useEffect(() => { api('/api/features').then(setMatches).catch(reason => setError(reason.message)); }, []);
  if (error) return <div className="chart-empty">{error}</div>;
  if (!matches) return <div className="chart-empty">Loading observed column matches…</div>;
  return <div className="concept-list">{Object.entries(matches.concept_matches).map(([concept, columns]) => <div key={concept}><span>{concept.replaceAll('_', ' ')}</span><strong>{columns.length ? columns.join(', ') : 'Not available from the current dataset'}</strong></div>)}</div>;
}
function PaperResultsTable() {
  return <Panel title="Severity classification results" eyebrow="TABLE 2 · FIXED RESEARCH PAPER VALUES" actions={<Badge kind="paper">Reported in Research Paper</Badge>}><Table columns={[{ key: 'model', label: 'MODEL' }, { key: 'accuracy', label: 'ACCURACY', render: value => `${fmt(value, 2)}%` }, { key: 'precision', label: 'PRECISION', render: value => fmt(value, 4) }, { key: 'recall', label: 'RECALL', render: value => fmt(value, 4) }, { key: 'f1', label: 'F1-SCORE', render: value => fmt(value, 4) }]} rows={PAPER_RESULTS} /></Panel>;
}
function TrainingResults({ result }) {
  const matrix = result.confusion_matrix;
  const classRows = Object.entries(result.classification_report).filter(([name, values]) => values && typeof values === 'object' && 'precision' in values).map(([name, values]) => ({ name, ...values }));
  const metrics = [
    ['Training accuracy', percent(result.metrics.training_accuracy)], ['Test accuracy', percent(result.metrics.accuracy)],
    ['Macro precision', fmt(result.metrics.precision_macro, 4)], ['Macro recall', fmt(result.metrics.recall_macro, 4)],
    ['Macro F1-score', fmt(result.metrics.f1_macro, 4)], ['Correctly classified', `${fmt(result.metrics.correctly_classified)} / ${fmt(result.split.test_rows)}`],
    ['Training time', `${fmt(result.timing.training_seconds, 4)} s`], ['Prediction time', `${fmt(result.timing.prediction_seconds, 5)} s`],
  ];
  const exportResult = () => {
    const csv = ['row_index,actual,predicted,correct,confidence', ...result.predictions.map(row => [row.row_index, row.actual, row.predicted, row.correct, row.confidence].join(','))].join('\n');
    const link = document.createElement('a'); link.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' })); link.download = `${result.model.toLowerCase().replaceAll(' ', '-')}-predictions.csv`; link.click(); URL.revokeObjectURL(link.href);
  };
  const exportTree = () => { const link = document.createElement('a'); link.href = URL.createObjectURL(new Blob([result.tree_text || 'Tree text unavailable for this estimator.'], { type: 'text/plain' })); link.download = 'decision-tree.txt'; link.click(); URL.revokeObjectURL(link.href); };
  return <div className="results-stack"><div className="results-title"><SourceHeading calculated>Classifier results</SourceHeading><div className="result-meta"><span><small>Classifier</small><strong>{result.model}</strong></span><span><small>Dataset</small><strong>{result.dataset}</strong></span><span><small>Instances / attributes</small><strong>{fmt(result.instances)} / {fmt(result.attributes)}</strong></span><span><small>Class</small><strong>{result.target}</strong></span><span><small>Split</small><strong>{fmt(result.split.train_rows)} train / {fmt(result.split.test_rows)} test · {result.split.method}</strong></span></div></div>
    <div className="metrics-result-grid">{metrics.map(([label, value]) => <div key={label}><small>{label}</small><strong>{value}</strong></div>)}</div>
    <div className="two-result-grid"><Panel title="Confusion matrix" eyebrow="ROWS: ACTUAL · COLUMNS: PREDICTED"><ConfusionMatrix matrix={matrix} /></Panel><Panel title="Per-class report" eyebrow="TEST SPLIT · ACTUAL METRICS"><Table columns={[{ key: 'name', label: 'CLASS' }, { key: 'precision', label: 'PRECISION', render: value => fmt(value, 3) }, { key: 'recall', label: 'RECALL', render: value => fmt(value, 3) }, { key: 'f1-score', label: 'F1', render: value => fmt(value, 3) }, { key: 'support', label: 'SUPPORT', render: value => fmt(value) }]} rows={classRows} /></Panel></div>
    {result.feature_importance?.length > 0 && <Panel title="Feature importance" eyebrow="TRANSFORMED MODEL FEATURES"><div className="chart-area chart-tall"><ResponsiveContainer width="100%" height="100%"><BarChart data={result.feature_importance.slice(0, 15).reverse()} layout="vertical" margin={{ left: 90, right: 20 }}><CartesianGrid strokeDasharray="3 3" horizontal={false} /><XAxis type="number" /><YAxis type="category" dataKey="feature" width={100} tick={{ fontSize: 9 }} /><Tooltip content={<ChartTooltip />} /><Bar dataKey="importance" fill="#4b8469" radius={[0, 3, 3, 0]} /></BarChart></ResponsiveContainer></div></Panel>}
    {result.tree_text && <Panel title="Decision tree representation" eyebrow="EXPORTED FROM THE TRAINED CLASSIFIER" actions={<button className="button button-secondary button-small" onClick={exportTree}><ArrowDownToLine size={13} />Download tree</button>}><pre className="tree-output">{result.tree_text}</pre></Panel>}
    <Panel title="Test-set predictions" eyebrow="CALCULATED ROW-BY-ROW" actions={<button className="button button-secondary button-small" onClick={exportResult}><ArrowDownToLine size={13} />Download results</button>}><Table columns={[{ key: 'row_index', label: 'ROW INDEX' }, { key: 'actual', label: 'ACTUAL' }, { key: 'predicted', label: 'PREDICTED' }, { key: 'correct', label: 'MATCH', render: value => <Badge kind={value ? 'green' : 'red'}>{value ? 'Correct' : 'Incorrect'}</Badge> }, { key: 'confidence', label: 'MAX CLASS PROBABILITY', render: percent }]} rows={result.predictions.slice(0, 100)} empty="No test predictions." /></Panel>
    </div>;
}
function ConfusionMatrix({ matrix }) {
  if (!matrix?.labels?.length) return <p className="chart-empty">Confusion matrix unavailable.</p>;
  return <div className="matrix-wrap"><table className="matrix"><thead><tr><th>Actual ↓ / Predicted →</th>{matrix.labels.map(label => <th key={label}>{label}</th>)}</tr></thead><tbody>{matrix.values.map((row, index) => <tr key={matrix.labels[index]}><th>{matrix.labels[index]}</th>{row.map((value, col) => <td key={col} style={{ background: index === col ? `rgba(57,126,98,${Math.min(.2 + value / Math.max(1, Math.max(...matrix.values.flat())) * .65, .85)})` : undefined }}>{fmt(value)}</td>)}</tr>)}</tbody></table></div>;
}
function PredictionResult({ prediction, future }) {
  const mapped = ['Low', 'Medium', 'High', 'Critical'].includes(prediction.predicted_severity);
  const action = { Low: 'Normal Monitoring', Medium: 'Advisory', High: 'Warning', Critical: 'Emergency Warning' }[prediction.predicted_severity] || 'No mapped action';
  return <div className={`prediction-card ${mapped ? `risk-${prediction.predicted_severity.toLowerCase()}` : ''}`}><div className="prediction-card-top"><span>{future ? `FUTURE SCENARIO · ${prediction.future_year}` : 'MODEL OUTPUT'}</span><Badge kind={mapped ? 'green' : 'neutral'}>{prediction.risk_level}</Badge></div><div className="prediction-answer"><small>PREDICTED SOURCE CLASS</small><strong>{prediction.predicted_severity}</strong></div><div className="prediction-stats"><span><small>Model</small><b>{prediction.model}</b></span><span><small>Probability</small><b>{percent(prediction.probability)}</b></span><span><small>Recommended action</small><b>{prediction.risk_action || action}</b></span><span><small>Time</small><b>{prediction.timestamp ? new Date(prediction.timestamp).toLocaleString() : '—'}</b></span></div>{prediction.methodology && <p className="footnote">{prediction.methodology}</p>}<p className="warning-copy">Research-generated decision support only. Not an official government warning.</p></div>;
}
function LstmResults({ result }) {
  const losses = result.loss_history.map((row, index) => ({ epoch: index + 1, training: row.training_loss, validation: row.validation_loss }));
  const points = result.validation.map((row, index) => ({ index: index + 1, actual: row.actual, predicted: row.predicted }));
  return <div className="two-result-grid"><Panel title="Training curve" eyebrow="ACTUAL PYTORCH LOSSES"><div className="chart-area"><ResponsiveContainer width="100%" height="100%"><LineChart data={losses}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="epoch" /><YAxis /><Tooltip content={<ChartTooltip />} /><Legend /><Line type="monotone" dataKey="training" stroke="#367b5c" dot={false} /><Line type="monotone" dataKey="validation" stroke="#d4844f" dot={false} /></LineChart></ResponsiveContainer></div></Panel><Panel title="Chronological test and forecast" eyebrow={`${result.rainfall_column} · ${result.time_column}`}><div className="chart-area"><ResponsiveContainer width="100%" height="100%"><LineChart data={points}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="index" /><YAxis /><Tooltip content={<ChartTooltip />} /><Legend /><Line type="monotone" dataKey="actual" stroke="#547f9b" dot={false} /><Line type="monotone" dataKey="predicted" stroke="#d4844f" dot={false} /></LineChart></ResponsiveContainer></div><Table columns={[{ key: 'step', label: 'FORECAST STEP' }, { key: 'predicted_rainfall', label: 'PREDICTED RAINFALL' }]} rows={result.forecast} /><p className="footnote">{result.warning}</p></Panel></div>;
}
function centerFor(points = []) {
  const valid = points.filter(point => Number.isFinite(Number(point.latitude)) && Number.isFinite(Number(point.longitude)));
  if (!valid.length) return [22.5, 79];
  return [valid.reduce((sum, item) => sum + Number(item.latitude), 0) / valid.length, valid.reduce((sum, item) => sum + Number(item.longitude), 0) / valid.length];
}

export default App;
