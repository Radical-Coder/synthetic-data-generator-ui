import { useDeferredValue, useRef, useState, useTransition } from 'react'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts'
import DataTable from './components/DataTable'
import Sidebar from './components/Sidebar'
import { BENCHMARK_OPTIONS, TAB_ITEMS, TYPE_STYLES } from './data/constants'
import {
  buildBoxPlotData,
  buildDetectionSummary,
  buildColumnOverrideInfo,
  buildFrequencyData,
  buildHistogramData,
  datetimeComparisonRows,
  detectColumnInfo,
  generateSyntheticRows,
  getCategoricalValues,
  getNumericValues,
  numericComparisonRows,
  parseReferenceFile,
  parseUploadedFile,
  rowsToCsv,
  summarizeSourceColumns,
  topFrequencies,
  triggerDownload
} from './utils/dataUtils'

const DEFAULT_CONFIG = {
  numRows: 1000,
  randomSeed: 42,
  maxComponents: 3,
  discreteThreshold: 0.05,
  useReferenceTables: false
}

const BENCHMARK_DEFAULTS = {
  1000: true,
  10000: true,
  50000: true,
  100000: false,
  500000: false,
  1000000: false
}

const TYPE_OVERRIDE_OPTIONS = [
  'continuous_numeric',
  'discrete_numeric',
  'categorical',
  'boolean',
  'datetime',
  'email',
  'phone'
]

function wait(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms)
  })
}

function formatStat(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '—'
  }

  return value.toFixed(2)
}

function formatKilobytes(fileSizeKb) {
  if (typeof fileSizeKb !== 'number' || Number.isNaN(fileSizeKb)) {
    return '—'
  }

  return `${fileSizeKb.toFixed(1)} KB`
}

function clampNumber(value, min, max) {
  return Math.min(max, Math.max(min, value))
}

function buildTimestamp() {
  const now = new Date()
  const year = now.getFullYear()
  const month = String(now.getMonth() + 1).padStart(2, '0')
  const day = String(now.getDate()).padStart(2, '0')
  const hours = String(now.getHours()).padStart(2, '0')
  const minutes = String(now.getMinutes()).padStart(2, '0')
  const seconds = String(now.getSeconds()).padStart(2, '0')

  return `${year}${month}${day}_${hours}${minutes}${seconds}`
}

function mean(values) {
  if (!values.length) {
    return 0
  }

  return values.reduce((sum, value) => sum + value, 0) / values.length
}

function sampleStdDev(values) {
  if (values.length <= 1) {
    return 0
  }

  const avg = mean(values)
  const variance = values.reduce((sum, value) => sum + (value - avg) ** 2, 0) / (values.length - 1)
  return Math.sqrt(variance)
}

function formatPercent(value, digits = 1) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '—'
  }

  return `${value.toFixed(digits)}%`
}

function relativeDelta(realValue, syntheticValue) {
  if (
    typeof realValue !== 'number' ||
    Number.isNaN(realValue) ||
    typeof syntheticValue !== 'number' ||
    Number.isNaN(syntheticValue)
  ) {
    return NaN
  }

  if (realValue === 0) {
    return syntheticValue === 0 ? 0 : 100
  }

  return (Math.abs(realValue - syntheticValue) / Math.abs(realValue)) * 100
}

function toneForDelta(delta) {
  if (typeof delta !== 'number' || Number.isNaN(delta)) {
    return { label: 'Open', tone: 'muted' }
  }

  if (delta <= 5) {
    return { label: 'Tight', tone: 'good' }
  }

  if (delta <= 15) {
    return { label: 'Watch', tone: 'watch' }
  }

  return { label: 'Wide', tone: 'high' }
}

function buildFrequencyLookup(values) {
  return values.reduce((map, value) => {
    const key = String(value)
    map.set(key, (map.get(key) ?? 0) + 1)
    return map
  }, new Map())
}

function distributionDrift(realValues, syntheticValues) {
  if (!realValues.length || !syntheticValues.length) {
    return NaN
  }

  const realCounts = buildFrequencyLookup(realValues)
  const syntheticCounts = buildFrequencyLookup(syntheticValues)
  const keys = new Set([...realCounts.keys(), ...syntheticCounts.keys()])

  let drift = 0
  keys.forEach((key) => {
    const realProbability = (realCounts.get(key) ?? 0) / realValues.length
    const syntheticProbability = (syntheticCounts.get(key) ?? 0) / syntheticValues.length
    drift += Math.abs(realProbability - syntheticProbability)
  })

  return (drift / 2) * 100
}

function formatDateTime(value) {
  if (!value) {
    return '—'
  }

  const parsed = value instanceof Date ? value : new Date(value)
  if (Number.isNaN(parsed.getTime())) {
    return '—'
  }

  return parsed.toLocaleString()
}

function formatDurationLabel(milliseconds) {
  if (typeof milliseconds !== 'number' || Number.isNaN(milliseconds) || milliseconds <= 0) {
    return '—'
  }

  const hours = milliseconds / (1000 * 60 * 60)
  const days = hours / 24

  if (days >= 60) {
    return `${(days / 30).toFixed(1)} months`
  }

  if (days >= 2) {
    return `${days.toFixed(1)} days`
  }

  if (hours >= 1) {
    return `${hours.toFixed(1)} hours`
  }

  return `${Math.round(milliseconds / (1000 * 60))} min`
}

function truncateAxisLabel(value, limit = 12) {
  const text = String(value)
  return text.length > limit ? `${text.slice(0, limit)}…` : text
}

function HelpTooltip({ text, label = 'Help', className = '', bubbleClassName = '' }) {
  return (
    <span className={`help-tooltip ${className}`.trim()}>
      <button
        type="button"
        className="help-tooltip-trigger"
        aria-label={label}
        title={text}
      >
        ?
      </button>
      <span className={`help-tooltip-bubble ${bubbleClassName}`.trim()} role="tooltip">
        {text}
      </span>
    </span>
  )
}

function FieldLabel({ text, helpText, className = '' }) {
  return (
    <span className={`field-label-row ${className}`.trim()}>
      <span>{text}</span>
      {helpText ? <HelpTooltip text={helpText} label={`Help for ${text}`} /> : null}
    </span>
  )
}

function StreamlitMultiSelect({
  label,
  helpText,
  options,
  selected,
  open,
  onToggleOpen,
  onToggleOption,
  onClearAll
}) {
  return (
    <div className="streamlit-multiselect">
      <div className="streamlit-multiselect-label-row">
        <label className="streamlit-multiselect-label">{label}</label>
        {helpText ? <HelpTooltip text={helpText} label={`Help for ${label}`} /> : null}
      </div>
      <button
        type="button"
        className={`streamlit-multiselect-trigger ${open ? 'is-open' : ''}`}
        onClick={onToggleOpen}
      >
        <div className="streamlit-multiselect-values">
          {selected.length ? (
            selected.map((value) => (
              <span key={value} className="streamlit-chip">
                {value}
                <button
                  type="button"
                  className="streamlit-chip-close"
                  aria-label={`Remove ${value}`}
                  onClick={(event) => {
                    event.stopPropagation()
                    onToggleOption(value)
                  }}
                >
                  ×
                </button>
              </span>
            ))
          ) : (
            <span className="streamlit-multiselect-placeholder">Choose columns for detailed comparison</span>
          )}
        </div>
        {selected.length ? (
          <button
            type="button"
            className="streamlit-multiselect-clear"
            aria-label="Clear selected columns"
            onClick={(event) => {
              event.stopPropagation()
              onClearAll()
            }}
          >
            ×
          </button>
        ) : null}
        <span className="streamlit-multiselect-caret" aria-hidden="true">
          ▾
        </span>
      </button>
      {open ? (
        <div className="streamlit-multiselect-menu">
          {options.map((option) => (
            <label key={option} className="streamlit-multiselect-option">
              <input
                type="checkbox"
                checked={selected.includes(option)}
                onChange={() => onToggleOption(option)}
              />
              <span>{option}</span>
            </label>
          ))}
        </div>
      ) : null}
    </div>
  )
}

function BoxPlotComparison({ data }) {
  if (!data.length) {
    return <div className="empty-state">Not enough numeric values to build a box plot.</div>
  }

  const width = 760
  const height = 260
  const left = 112
  const right = 42
  const top = 38
  const plotWidth = width - left - right
  const axisY = 218
  const allValues = data.flatMap((series) => [
    series.min,
    series.max,
    series.lowerWhisker,
    series.upperWhisker,
    ...series.outliers
  ])
  const rawMin = Math.min(...allValues)
  const rawMax = Math.max(...allValues)
  const padding = rawMax === rawMin ? 1 : (rawMax - rawMin) * 0.08
  const domainMin = rawMin - padding
  const domainMax = rawMax + padding
  const domainSpan = domainMax - domainMin || 1
  const scaleX = (value) => left + ((value - domainMin) / domainSpan) * plotWidth
  const ticks = Array.from({ length: 5 }, (_, index) => domainMin + (domainSpan * index) / 4)

  return (
    <div className="box-plot-card">
      <svg className="box-plot-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Real and synthetic box plot with outlier dots">
        <line className="box-plot-axis" x1={left} x2={width - right} y1={axisY} y2={axisY} />
        {ticks.map((tick) => {
          const x = scaleX(tick)
          return (
            <g key={tick}>
              <line className="box-plot-grid" x1={x} x2={x} y1={top - 12} y2={axisY} />
              <line className="box-plot-axis-tick" x1={x} x2={x} y1={axisY} y2={axisY + 6} />
              <text className="box-plot-axis-label" x={x} y={axisY + 24} textAnchor="middle">
                {formatStat(tick)}
              </text>
            </g>
          )
        })}

        {data.map((series, index) => {
          const y = top + index * 82 + 32
          const boxTop = y - 17
          const boxHeight = 34
          const outliersToShow = series.outliers

          return (
            <g key={series.label}>
              <text className="box-plot-series-label" x={0} y={y + 5}>
                {series.label}
              </text>
              <line
                className={`box-plot-whisker is-${index === 0 ? 'real' : 'synthetic'}`}
                x1={scaleX(series.lowerWhisker)}
                x2={scaleX(series.upperWhisker)}
                y1={y}
                y2={y}
              />
              <line
                className="box-plot-cap"
                x1={scaleX(series.lowerWhisker)}
                x2={scaleX(series.lowerWhisker)}
                y1={y - 15}
                y2={y + 15}
              />
              <line
                className="box-plot-cap"
                x1={scaleX(series.upperWhisker)}
                x2={scaleX(series.upperWhisker)}
                y1={y - 15}
                y2={y + 15}
              />
              <rect
                className={`box-plot-box is-${index === 0 ? 'real' : 'synthetic'}`}
                x={scaleX(series.q1)}
                y={boxTop}
                width={Math.max(2, scaleX(series.q3) - scaleX(series.q1))}
                height={boxHeight}
                rx={5}
              />
              <line
                className="box-plot-median"
                x1={scaleX(series.median)}
                x2={scaleX(series.median)}
                y1={boxTop - 4}
                y2={boxTop + boxHeight + 4}
              />
              {outliersToShow.map((value, outlierIndex) => (
                <circle
                  key={`${series.label}-${value}-${outlierIndex}`}
                  className={`box-plot-outlier is-${index === 0 ? 'real' : 'synthetic'}`}
                  cx={scaleX(value)}
                  cy={y + ((outlierIndex % 5) - 2) * 3}
                  r={3.8}
                />
              ))}
              {series.outlierCount > outliersToShow.length ? (
                <text className="box-plot-outlier-note" x={width - right} y={y - 23} textAnchor="end">
                  showing {outliersToShow.length}/{series.outlierCount} outliers
                </text>
              ) : null}
            </g>
          )
        })}
      </svg>

      <div className="box-plot-summary-grid">
        {data.map((series) => (
          <div key={`${series.label}-summary`} className="box-plot-summary-item">
            <strong>{series.label}</strong>
            <span>
              median {formatStat(series.median)} · IQR {formatStat(series.q1)}-{formatStat(series.q3)} · outliers{' '}
              {series.outlierCount}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function App() {
  const fileInputRef = useRef(null)
  const [activeTab, setActiveTab] = useState('upload')
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)
  const [originalRows, setOriginalRows] = useState([])
  const [syntheticRows, setSyntheticRows] = useState([])
  const [sourceColumnInfo, setSourceColumnInfo] = useState({})
  const [columnInfo, setColumnInfo] = useState({})
  const [uploadMeta, setUploadMeta] = useState(null)
  const [loadError, setLoadError] = useState('')
  const [referenceTables, setReferenceTables] = useState({})
  const [selectedReferenceColumns, setSelectedReferenceColumns] = useState([])
  const [generationComplete, setGenerationComplete] = useState(false)
  const [selectedColumns, setSelectedColumns] = useState([])
  const [statusMessage, setStatusMessage] = useState('')
  const [statusTone, setStatusTone] = useState('info')
  const [benchmarkResults, setBenchmarkResults] = useState([])
  const [config, setConfig] = useState(DEFAULT_CONFIG)
  const [benchmarkSelection, setBenchmarkSelection] = useState(BENCHMARK_DEFAULTS)
  const [referencePickerValue, setReferencePickerValue] = useState('')
  const [isQualitySelectorOpen, setIsQualitySelectorOpen] = useState(false)
  const [isPending, startTransition] = useTransition()

  const deferredSelectedColumns = useDeferredValue(selectedColumns)
  const hasData = originalRows.length > 0
  const activeColumnInfo = generationComplete ? columnInfo : sourceColumnInfo
  const referenceableColumns = Object.entries(sourceColumnInfo)
    .filter(([, details]) => details.type === 'categorical')
    .map(([column]) => column)
  const selectedBenchmarkSizes = Object.entries(benchmarkSelection)
    .filter(([, enabled]) => enabled)
    .map(([size]) => Number(size))
  const showHeroStatus = statusTone === 'error' && Boolean(statusMessage)

  function resetResults() {
    setSyntheticRows([])
    setColumnInfo({})
    setGenerationComplete(false)
    setBenchmarkResults([])
  }

  async function handleDatasetUpload(event) {
    const file = event.target.files?.[0]
    if (!file) {
      return
    }

    setStatusTone('info')
    setStatusMessage('Loading dataset...')
    setLoadError('')

    try {
      const parsed = await parseUploadedFile(file)
      const detected = detectColumnInfo(parsed.rows, config.discreteThreshold)

      startTransition(() => {
        setOriginalRows(parsed.rows)
        setUploadMeta({
          fileName: file.name,
          rows: parsed.rows.length,
          columns: parsed.columns.length,
          fileSizeKb: parsed.fileSizeKb,
          nestedColumns: parsed.nestedColumns
        })
        setSourceColumnInfo(detected)
        setSelectedColumns(parsed.columns.slice(0, 3))
        setConfig((current) => ({
          ...current,
          numRows: Math.min(1000, parsed.rows.length || 1000)
        }))
        setReferenceTables({})
        setSelectedReferenceColumns([])
        setReferencePickerValue('')
        resetResults()
      })

      setStatusTone('info')
      setStatusMessage('')
    } catch (error) {
      setLoadError(error.message)
      setStatusTone('error')
      setStatusMessage('Failed to load the file.')
    } finally {
      event.target.value = ''
    }
  }

  async function handleReferenceUpload(column, event) {
    const file = event.target.files?.[0]
    if (!file) {
      return
    }

    try {
      const values = await parseReferenceFile(file)
      setReferenceTables((current) => ({
        ...current,
        [column]: values
      }))
    } catch (error) {
      setLoadError(`Could not read reference file for ${column}: ${error.message}`)
    } finally {
      event.target.value = ''
    }
  }

  function updateDiscreteThreshold(nextValue) {
    const numericValue = Number(nextValue)
    setConfig((current) => ({
      ...current,
      discreteThreshold: numericValue
    }))

    if (hasData) {
      setSourceColumnInfo(detectColumnInfo(originalRows, numericValue))
      resetResults()
    }
  }

  function updateConfigNumber(field, nextValue, min, max) {
    const numericValue = Number(nextValue)
    if (Number.isNaN(numericValue)) {
      return
    }

    setConfig((current) => ({
      ...current,
      [field]: clampNumber(numericValue, min, max)
    }))
  }

  function nudgeConfigNumber(field, delta, min, max) {
    setConfig((current) => ({
      ...current,
      [field]: clampNumber((Number(current[field]) || 0) + delta, min, max)
    }))
  }

  async function handleGenerate() {
    if (!hasData) {
      return
    }

    setStatusTone('info')
    setStatusMessage('Step 1/3: Analyzing column types...')
    await wait(180)

    const reviewedColumnInfo = Object.keys(sourceColumnInfo).length
      ? sourceColumnInfo
      : detectColumnInfo(originalRows, config.discreteThreshold)

    setStatusMessage('Step 2/3: Generating synthetic data...')
    await wait(180)

    const generated = generateSyntheticRows({
      rows: originalRows,
      columnInfo: reviewedColumnInfo,
      referenceTables: config.useReferenceTables ? referenceTables : {},
      numRows: config.numRows,
      seed: config.randomSeed
    })

    setStatusMessage('Step 3/3: Validating quality...')
    await wait(180)

    startTransition(() => {
      setColumnInfo(reviewedColumnInfo)
      setSyntheticRows(generated)
      setGenerationComplete(true)
      setSelectedColumns(Object.keys(reviewedColumnInfo).slice(0, 3))
    })

    setStatusTone('success')
    setStatusMessage(`Generation complete. ${generated.length.toLocaleString()} rows ready.`)
  }

  function clearReferenceTables() {
    setReferenceTables({})
    setSelectedReferenceColumns([])
    setReferencePickerValue('')
  }

  function clearUploadedData() {
    setOriginalRows([])
    setSyntheticRows([])
    setSourceColumnInfo({})
    setColumnInfo({})
    setUploadMeta(null)
    setLoadError('')
    setReferenceTables({})
    setSelectedReferenceColumns([])
    setGenerationComplete(false)
    setSelectedColumns([])
    setBenchmarkResults([])
    setStatusTone('info')
    setStatusMessage('')
    setReferencePickerValue('')
    setIsQualitySelectorOpen(false)
  }

  function openFilePicker() {
    fileInputRef.current?.click()
  }

  function toggleColumnSelection(column) {
    setSelectedColumns((current) =>
      current.includes(column)
        ? current.filter((item) => item !== column)
        : [...current, column]
    )
  }

  function addReferenceColumn(column) {
    if (!column) {
      return
    }

    setSelectedReferenceColumns((current) => (current.includes(column) ? current : [...current, column]))
    setReferencePickerValue('')
  }

  function handleColumnTypeOverride(column, nextType) {
    const overrideInfo = buildColumnOverrideInfo(originalRows, column, nextType, config.discreteThreshold)

    startTransition(() => {
      setSourceColumnInfo((current) => ({
        ...current,
        [column]: overrideInfo
      }))
      resetResults()
    })
  }

  function resetColumnType(column) {
    const detected = detectColumnInfo(originalRows, config.discreteThreshold)

    startTransition(() => {
      setSourceColumnInfo((current) => ({
        ...current,
        [column]: detected[column]
      }))
      resetResults()
    })
  }

  function resetAllColumnTypes() {
    startTransition(() => {
      setSourceColumnInfo(detectColumnInfo(originalRows, config.discreteThreshold))
      resetResults()
    })
  }

  function toggleBenchmark(size) {
    setBenchmarkSelection((current) => ({
      ...current,
      [size]: !current[size]
    }))
  }

  async function runBenchmarks() {
    if (!hasData || !selectedBenchmarkSizes.length) {
      setStatusTone('warning')
      setStatusMessage('Select at least one benchmark size first.')
      return
    }

    const detected = generationComplete
      ? columnInfo
      : Object.keys(sourceColumnInfo).length
        ? sourceColumnInfo
        : detectColumnInfo(originalRows, config.discreteThreshold)
    const nextResults = []

    for (let index = 0; index < selectedBenchmarkSizes.length; index += 1) {
      const size = selectedBenchmarkSizes[index]
      setStatusTone('info')
      setStatusMessage(`Benchmarking ${size.toLocaleString()} rows... (${index + 1}/${selectedBenchmarkSizes.length})`)
      await wait(120)

      const startedAt = performance.now()
      generateSyntheticRows({
        rows: originalRows,
        columnInfo: detected,
        referenceTables: config.useReferenceTables ? referenceTables : {},
        numRows: size,
        seed: config.randomSeed + index
      })
      const elapsedMs = performance.now() - startedAt
      const elapsedSeconds = elapsedMs / 1000
      const rowsPerSecond = Math.max(1, Math.round(size / (elapsedSeconds || 1)))

      nextResults.push({
        rowsGenerated: size,
        timeSeconds: Number(elapsedSeconds.toFixed(2)),
        rowsPerSecond
      })
    }

    startTransition(() => {
      setBenchmarkResults(nextResults)
      if (!generationComplete) {
        setColumnInfo(detected)
      }
      setActiveTab('benchmarks')
    })

    setStatusTone('success')
    setStatusMessage('Benchmark complete.')
  }

  function downloadSyntheticCsv() {
    const timestamp = buildTimestamp()
    triggerDownload(
      `synthetic_data_${timestamp}.csv`,
      rowsToCsv(syntheticRows),
      'text/csv'
    )
  }

  function downloadColumnInfo() {
    const timestamp = buildTimestamp()
    triggerDownload(
      `column_info_${timestamp}.json`,
      JSON.stringify(columnInfo, null, 2),
      'application/json'
    )
  }

  const detectionRows = buildDetectionSummary(columnInfo, config.useReferenceTables ? referenceTables : {})
  const sourceDetectionRows = buildDetectionSummary(sourceColumnInfo, config.useReferenceTables ? referenceTables : {})
  const numericComparison = numericComparisonRows(originalRows, syntheticRows, columnInfo)
  const datetimeComparison = datetimeComparisonRows(originalRows, syntheticRows, columnInfo)
  const syntheticSizeMb = syntheticRows.length
    ? rowsToCsv(syntheticRows).length / 1024 / 1024
    : 0
  const sourceColumnSummary = summarizeSourceColumns(originalRows)
  const benchmarkChartData = benchmarkResults.filter(
    (result) => typeof result.timeSeconds === 'number' && !Number.isNaN(result.timeSeconds)
  )
  const fastestBenchmark =
    benchmarkChartData.length > 0
      ? benchmarkChartData.reduce((fastest, current) =>
          current.timeSeconds < fastest.timeSeconds ? current : fastest
        )
      : null
  const slowestBenchmark =
    benchmarkChartData.length > 0
      ? benchmarkChartData.reduce((slowest, current) =>
          current.timeSeconds > slowest.timeSeconds ? current : slowest
        )
      : null
  const averageBenchmarkRps = benchmarkChartData.length
    ? Math.round(
        benchmarkChartData.reduce((sum, current) => sum + current.rowsPerSecond, 0) /
          benchmarkChartData.length
      )
    : 0

  const numericComparisonColumns = Object.entries(columnInfo)
    .filter(([, details]) => details.type === 'continuous_numeric' || details.type === 'discrete_numeric')
    .map(([column, details]) => {
      const realValues = getNumericValues(originalRows, column)
      const syntheticValues = getNumericValues(syntheticRows, column)
      const realMean = mean(realValues)
      const syntheticMean = mean(syntheticValues)
      const realStd = sampleStdDev(realValues)
      const syntheticStd = sampleStdDev(syntheticValues)
      const meanDiffPct = relativeDelta(realMean, syntheticMean)
      const stdDiffPct = relativeDelta(realStd, syntheticStd)

      return {
        column,
        details,
        realMean,
        syntheticMean,
        realStd,
        syntheticStd,
        realMin: realValues.length ? Math.min(...realValues) : NaN,
        realMax: realValues.length ? Math.max(...realValues) : NaN,
        syntheticMin: syntheticValues.length ? Math.min(...syntheticValues) : NaN,
        syntheticMax: syntheticValues.length ? Math.max(...syntheticValues) : NaN,
        meanDiffPct,
        stdDiffPct,
        meanTone: toneForDelta(meanDiffPct),
        stdTone: toneForDelta(stdDiffPct),
        histogramData: buildHistogramData(realValues, syntheticValues, 10),
        boxPlotData: buildBoxPlotData(realValues, syntheticValues)
      }
    })

  const categoricalComparisonColumns = Object.entries(columnInfo)
    .filter(([, details]) => ['categorical', 'boolean', 'email', 'phone'].includes(details.type))
    .map(([column, details]) => {
      const realValues = getCategoricalValues(originalRows, column)
      const syntheticValues = getCategoricalValues(syntheticRows, column)
      const frequencyData = buildFrequencyData(realValues, syntheticValues, 8)
      const realTop = topFrequencies(realValues, 3)
      const syntheticTop = topFrequencies(syntheticValues, 3)
      const realSet = new Set(realValues.map(String))
      const syntheticSet = new Set(syntheticValues.map(String))
      const missingFromSynthetic = Array.from(realSet).filter((value) => !syntheticSet.has(value)).length
      const driftPct = distributionDrift(realValues, syntheticValues)

      return {
        column,
        details,
        realUnique: realSet.size,
        syntheticUnique: syntheticSet.size,
        missingFromSynthetic,
        driftPct,
        driftTone: toneForDelta(driftPct),
        frequencyData,
        summaryRows: frequencyData.map((row) => ({
          Category: row.category,
          Real: realValues.length ? formatPercent((row.real / realValues.length) * 100, 1) : '0.0%',
          Synthetic: syntheticValues.length
            ? formatPercent((row.synthetic / syntheticValues.length) * 100, 1)
            : '0.0%'
        })),
        realTop,
        syntheticTop
      }
    })

  const datetimeComparisonColumns = Object.entries(columnInfo)
    .filter(([, details]) => details.type === 'datetime')
    .map(([column, details]) => {
      const realTimes = originalRows
        .map((row) => Date.parse(row[column]))
        .filter((value) => !Number.isNaN(value))
      const syntheticTimes = syntheticRows
        .map((row) => Date.parse(row[column]))
        .filter((value) => !Number.isNaN(value))
      const realMin = realTimes.length ? Math.min(...realTimes) : NaN
      const realMax = realTimes.length ? Math.max(...realTimes) : NaN
      const syntheticMin = syntheticTimes.length ? Math.min(...syntheticTimes) : NaN
      const syntheticMax = syntheticTimes.length ? Math.max(...syntheticTimes) : NaN
      const realSpan = Number.isNaN(realMin) || Number.isNaN(realMax) ? NaN : realMax - realMin
      const syntheticSpan =
        Number.isNaN(syntheticMin) || Number.isNaN(syntheticMax) ? NaN : syntheticMax - syntheticMin
      const spanDiffPct = relativeDelta(realSpan, syntheticSpan)

      return {
        column,
        details,
        realMin,
        realMax,
        syntheticMin,
        syntheticMax,
        realSpan,
        syntheticSpan,
        spanDiffPct,
        spanTone: toneForDelta(spanDiffPct)
      }
    })

  const averageNumericMeanDiff = numericComparisonColumns.length
    ? mean(numericComparisonColumns.map((item) => item.meanDiffPct))
    : NaN
  const averageCategoricalDrift = categoricalComparisonColumns.length
    ? mean(categoricalComparisonColumns.map((item) => item.driftPct))
    : NaN
  const averageDatetimeSpanDiff = datetimeComparisonColumns.length
    ? mean(datetimeComparisonColumns.map((item) => item.spanDiffPct))
    : NaN

  return (
    <div className={`app-shell ${isSidebarOpen ? '' : 'is-sidebar-collapsed'}`}>
      <button
        type="button"
        className="sidebar-open-button"
        aria-label="Open sidebar"
        aria-controls="app-sidebar"
        aria-expanded={isSidebarOpen}
        onClick={() => setIsSidebarOpen(true)}
      >
        Show sidebar
      </button>

      <Sidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        hasData={hasData}
        generationComplete={generationComplete}
        originalRows={originalRows}
        syntheticRows={syntheticRows}
      />

      <main className="app-main">
        <header className="hero">
          <h1>Auto-Detecting Synthetic Data Generator</h1>
          <div className="info-banner">
            <strong>Welcome!</strong> This tool automatically generates high-quality synthetic data that preserves
            the statistical properties of your original dataset. No coding required!
          </div>
          {showHeroStatus ? (
            <div className={`message-banner is-${statusTone}`}>{statusMessage}</div>
          ) : null}
        </header>

        <nav className="tab-strip" aria-label="Synthetic data tabs">
          {TAB_ITEMS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              className={`tab-button ${activeTab === tab.id ? 'is-active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        {activeTab === 'upload' ? (
          <section className="content-panel">
            <div className="section-block upload-step">
              <h2>Step 1: Upload Your Dataset</h2>
              <p className="section-copy">Choose a CSV or JSON file</p>
              <div className="uploader-frame">
                <HelpTooltip
                  text="Upload a CSV or flat JSON file. The tool will automatically detect column types."
                  label="Upload help"
                  className="uploader-help"
                />
                <div
                  className="uploader-dropzone"
                  onClick={openFilePicker}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault()
                      openFilePicker()
                    }
                  }}
                  role="button"
                  tabIndex={0}
                >
                  <div className="uploader-main">
                    <div className="upload-cloud" aria-hidden="true">
                      <svg viewBox="0 0 48 48" className="upload-cloud-svg">
                        <path
                          d="M15 35h18a8 8 0 0 0 .7-16 12 12 0 0 0-23.2-3.1A9 9 0 0 0 15 35Z"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2.8"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                        <path
                          d="M24 28V17m0 0-5 5m5-5 5 5"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2.8"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                    </div>
                    <div className="uploader-copy">
                      <div className="uploader-title">Drag and drop file here</div>
                      <div className="uploader-subtitle">Limit 200MB per file • CSV, JSON</div>
                    </div>
                  </div>

                  <button
                    type="button"
                    className="browse-files-button"
                    onClick={(event) => {
                      event.stopPropagation()
                      openFilePicker()
                    }}
                  >
                    Browse files
                  </button>
                </div>
                <input
                  ref={fileInputRef}
                  className="hidden-file-input"
                  type="file"
                  accept=".csv,.json"
                  onChange={handleDatasetUpload}
                />
              </div>
              {uploadMeta ? (
                <div className="uploaded-file-row">
                  <div className="uploaded-file-main">
                    <div className="uploaded-file-icon" aria-hidden="true">
                      <svg viewBox="0 0 24 24" className="uploaded-file-svg">
                        <path
                          d="M7 3h7l5 5v13H7z"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="1.7"
                          strokeLinejoin="round"
                        />
                        <path
                          d="M14 3v5h5"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="1.7"
                          strokeLinejoin="round"
                        />
                      </svg>
                    </div>
                    <div className="uploaded-file-copy">
                      <span className="uploaded-file-name">{uploadMeta.fileName}</span>
                      <span className="uploaded-file-size">{formatKilobytes(uploadMeta.fileSizeKb)}</span>
                    </div>
                  </div>
                  <button type="button" className="uploaded-file-remove" onClick={clearUploadedData} aria-label="Remove uploaded file">
                    ×
                  </button>
                </div>
              ) : null}
              {loadError ? <div className="message-banner is-error">{loadError}</div> : null}
              {uploadMeta?.nestedColumns?.length ? (
                <div className="message-banner is-warning">
                  Dropped nested columns: {uploadMeta.nestedColumns.join(', ')}
                </div>
              ) : null}
            </div>

            {hasData ? (
              <>
                {!generationComplete ? (
                  <div className="message-banner is-success upload-success-banner">File uploaded successfully!</div>
                ) : null}

                <div className="upload-metrics-row">
                  <div className="upload-metric">
                    <div className="upload-metric-label">Total Rows</div>
                    <div className="upload-metric-value">{uploadMeta.rows.toLocaleString()}</div>
                  </div>
                  <div className="upload-metric">
                    <div className="upload-metric-label">Total Columns</div>
                    <div className="upload-metric-value">{uploadMeta.columns}</div>
                  </div>
                  <div className="upload-metric">
                    <div className="upload-metric-label">File Size</div>
                    <div className="upload-metric-value">{formatKilobytes(uploadMeta.fileSizeKb)}</div>
                  </div>
                </div>

                <div className="section-block">
                  <h2>Preview of seed/sample data</h2>
                  <DataTable rows={originalRows} showIndex compact dataframe emptyValue="None" />
                </div>

                <div className="section-block">
                  <h2>Step 2: Configure Generation Parameters</h2>
                  <div className="form-grid two-column">
                    <label className="field field-stepper">
                      <FieldLabel
                        text="Number of rows to generate"
                        helpText="How many synthetic rows to generate"
                      />
                      <div className="stepper-input">
                        <input
                          type="number"
                          min="100"
                          max="1000000"
                          step="100"
                          value={config.numRows}
                          onChange={(event) => updateConfigNumber('numRows', event.target.value, 100, 1000000)}
                        />
                        <div className="stepper-controls">
                          <button
                            type="button"
                            className="stepper-button"
                            aria-label="Decrease number of rows"
                            onClick={() => nudgeConfigNumber('numRows', -100, 100, 1000000)}
                          >
                            −
                          </button>
                          <button
                            type="button"
                            className="stepper-button"
                            aria-label="Increase number of rows"
                            onClick={() => nudgeConfigNumber('numRows', 100, 100, 1000000)}
                          >
                            +
                          </button>
                        </div>
                      </div>
                    </label>

                    <label className="field field-stepper">
                      <FieldLabel
                        text="Random Seed (for reproducibility)"
                        helpText="Use same seed to get identical results"
                      />
                      <div className="stepper-input">
                        <input
                          type="number"
                          min="0"
                          max="9999"
                          value={config.randomSeed}
                          onChange={(event) => updateConfigNumber('randomSeed', event.target.value, 0, 9999)}
                        />
                        <div className="stepper-controls">
                          <button
                            type="button"
                            className="stepper-button"
                            aria-label="Decrease random seed"
                            onClick={() => nudgeConfigNumber('randomSeed', -1, 0, 9999)}
                          >
                            −
                          </button>
                          <button
                            type="button"
                            className="stepper-button"
                            aria-label="Increase random seed"
                            onClick={() => nudgeConfigNumber('randomSeed', 1, 0, 9999)}
                          >
                            +
                          </button>
                        </div>
                      </div>
                    </label>
                  </div>

                  <details className="details-panel" open>
                    <summary>Advanced Options</summary>
                    <div className="details-content">
                      <label className="field range-field">
                        <FieldLabel
                          text="Maximum GMM Components"
                          helpText="Maximum Gaussian components for continuous data modeling"
                          className="range-field-label"
                        />
                        <div className="range-value-display">{config.maxComponents}</div>
                        <input
                          type="range"
                          min="1"
                          max="10"
                          value={config.maxComponents}
                          onChange={(event) =>
                            setConfig((current) => ({
                              ...current,
                              maxComponents: Number(event.target.value)
                            }))
                          }
                        />
                        <div className="range-endpoints">
                          <span>1</span>
                          <span>10</span>
                        </div>
                      </label>

                      <label className="field range-field">
                        <FieldLabel
                          text="Discrete Detection Threshold"
                          helpText="Unique value ratio below which numeric columns are treated as discrete"
                          className="range-field-label"
                        />
                        <div className="range-value-display">{config.discreteThreshold.toFixed(2)}</div>
                        <input
                          type="range"
                          min="0.01"
                          max="0.20"
                          step="0.01"
                          value={config.discreteThreshold}
                          onChange={(event) => updateDiscreteThreshold(event.target.value)}
                        />
                        <div className="range-endpoints">
                          <span>0.01</span>
                          <span>0.20</span>
                        </div>
                      </label>

                      <label className="checkbox-field">
                        <input
                          type="checkbox"
                          checked={config.useReferenceTables}
                          onChange={(event) =>
                            setConfig((current) => ({
                              ...current,
                              useReferenceTables: event.target.checked
                            }))
                          }
                        />
                        <FieldLabel
                          text="Use Reference Tables"
                          helpText="Provide custom lists of valid values for categorical columns"
                        />
                      </label>

                      {config.useReferenceTables ? (
                        <div className="reference-shell">
                          <h3 className="reference-title">Upload Reference Tables for Categorical Columns</h3>
                          <p className="reference-caption">
                            Each reference table should be a CSV file with a single column of valid values.
                          </p>
                          {referenceableColumns.length ? (
                            <label className="field">
                              <FieldLabel
                                text="Select columns to add reference tables for:"
                                helpText="Only categorical columns are shown"
                              />
                              <select
                                value={referencePickerValue}
                                onChange={(event) => {
                                  const nextColumn = event.target.value
                                  setReferencePickerValue(nextColumn)
                                  addReferenceColumn(nextColumn)
                                }}
                              >
                                <option value="">Choose an option</option>
                                {referenceableColumns.map((column) => (
                                  <option key={column} value={column}>
                                    {column}
                                  </option>
                                ))}
                              </select>
                            </label>
                          ) : (
                            <div className="empty-state">Upload a dataset first to expose categorical columns.</div>
                          )}

                          {selectedReferenceColumns.map((column) => (
                            <div key={column} className="reference-row">
                              <label className="field compact">
                                <span>Reference CSV for {column}</span>
                                <input
                                  type="file"
                                  accept=".csv"
                                  onChange={(event) => handleReferenceUpload(column, event)}
                                />
                              </label>
                              {referenceTables[column]?.length ? (
                                <div className="status-tile is-success">
                                  {column} — {referenceTables[column].length.toLocaleString()} values loaded from reference table
                                </div>
                              ) : null}
                            </div>
                          ))}

                          {Object.keys(referenceTables).length ? (
                            <div className="reference-active-list">
                              <strong>Active Reference Tables:</strong>
                              {Object.entries(referenceTables).map(([column, values]) => (
                                <div key={column} className="status-tile is-info">
                                  {column}: {values.length.toLocaleString()} reference values loaded
                                </div>
                              ))}
                              <div className="inline-actions">
                                <button type="button" className="secondary-button" onClick={clearReferenceTables}>
                                  Clear All Reference Tables
                                </button>
                              </div>
                            </div>
                          ) : null}
                        </div>
                      ) : null}
                    </div>
                  </details>
                </div>

                <div className="section-block">
                  <div className="section-header-row">
                    <div>
                      <h2>Review &amp; Override Detected Data Types</h2>
                      <p className="section-copy">
                        Confirm the detected column types before generation. Overrides change how the synthetic data
                        is generated for that column.
                      </p>
                    </div>
                    <button type="button" className="secondary-button" onClick={resetAllColumnTypes}>
                      Reset all types
                    </button>
                  </div>

                  <div className="type-override-list">
                    {Object.entries(sourceColumnInfo).map(([column, details]) => {
                      const style = TYPE_STYLES[details?.type] ?? TYPE_STYLES.unknown
                      const originalStyle = TYPE_STYLES[details?.originalDetectedType] ?? null

                      return (
                        <div key={column} className="type-override-row">
                          <div className="type-override-main">
                            <div className="type-override-column">{column}</div>
                            <div className="type-override-meta">
                              Current: {style.emoji} {style.label}
                              {details?.overridden && originalStyle ? (
                                <span> • auto was {originalStyle.emoji} {originalStyle.label}</span>
                              ) : null}
                            </div>
                          </div>

                          <label className="field type-override-select">
                            <span className="sr-only">Override type for {column}</span>
                            <select
                              value={details?.type ?? 'categorical'}
                              onChange={(event) => handleColumnTypeOverride(column, event.target.value)}
                            >
                              {TYPE_OVERRIDE_OPTIONS.map((type) => {
                                const optionStyle = TYPE_STYLES[type] ?? TYPE_STYLES.unknown
                                return (
                                  <option key={type} value={type}>
                                    {optionStyle.emoji} {optionStyle.label}
                                  </option>
                                )
                              })}
                            </select>
                          </label>

                          <button
                            type="button"
                            className="table-clear-button"
                            disabled={!details?.overridden && !details?.overrideRequested}
                            onClick={() => resetColumnType(column)}
                          >
                            Reset
                          </button>
                        </div>
                      )
                    })}
                  </div>

                  <details className="details-panel compact-details">
                    <summary>Detected data type summary</summary>
                    <div className="details-content">
                      <DataTable rows={sourceDetectionRows} maxRows={sourceDetectionRows.length || 10} showIndex dataframe />
                    </div>
                  </details>
                </div>

                <div className="section-block">
                  <h2>Step 3: Generate Synthetic Data</h2>
                  {Object.keys(referenceTables).length && config.useReferenceTables ? (
                    <div className="message-banner is-success">
                      Reference tables active for: {Object.keys(referenceTables).join(', ')}
                    </div>
                  ) : null}
                  <button
                    type="button"
                    className="primary-button"
                    onClick={handleGenerate}
                    disabled={isPending}
                  >
                    {isPending ? 'Working...' : 'Generate Synthetic Data'}
                  </button>
                </div>

                {generationComplete ? (
                  <>
                    <div className="message-banner is-success">
                      <strong>Generation Complete!</strong>
                      <div>
                        Successfully generated {syntheticRows.length.toLocaleString()} rows with{' '}
                        {Object.keys(columnInfo).length} columns.
                      </div>
                      <div>Navigate to other tabs to view quality dashboard and download results.</div>
                    </div>

                    <div className="section-block">
                      <h2>Column Detection Summary</h2>
                      <DataTable rows={detectionRows} maxRows={detectionRows.length || 10} showIndex dataframe />
                    </div>
                  </>
                ) : null}
              </>
            ) : (
              <div className="message-banner is-warning">Please upload a CSV or JSON file to begin.</div>
            )}
          </section>
        ) : null}

        {activeTab === 'quality' ? (
          <section className="content-panel">
            <div className="section-block">
              <h2>Quality Validation Dashboard</h2>
              {generationComplete ? (
                <>
                  <div className="quality-section">
                    <h3>Summary Statistics</h3>
                    <div className="quality-metrics-row">
                      <div className="quality-metric">
                        <div className="quality-metric-label">Original Rows</div>
                        <div className="quality-metric-value">{originalRows.length.toLocaleString()}</div>
                      </div>
                      <div className="quality-metric">
                        <div className="quality-metric-label">Synthetic Rows</div>
                        <div className="quality-metric-value">{syntheticRows.length.toLocaleString()}</div>
                      </div>
                      <div className="quality-metric">
                        <div className="quality-metric-label">Numeric Columns</div>
                        <div className="quality-metric-value">{sourceColumnSummary.numeric}</div>
                      </div>
                      <div className="quality-metric">
                        <div className="quality-metric-label">Categorical Columns</div>
                        <div className="quality-metric-value">{sourceColumnSummary.categorical}</div>
                      </div>
                    </div>
                  </div>

                  {numericComparisonColumns.length ? (
                    <div className="quality-section">
                      <h3>Numeric Outlier Box Plots</h3>
                      <p className="section-copy">
                        Quick view of numeric spread, medians, whiskers, and outlier dots for every detected numeric
                        column.
                      </p>
                      <div className="comparison-card-grid comparison-card-grid-numeric">
                        {numericComparisonColumns.map((item) => (
                          <article key={`quality-box-${item.column}`} className="comparison-card">
                            <div className="comparison-card-head">
                              <div>
                                <h4>{item.column}</h4>
                                <p className="comparison-card-copy">Real vs synthetic numeric outliers</p>
                              </div>
                              <span className={`comparison-badge is-${item.meanTone.tone}`}>
                                Mean {item.meanTone.label}
                              </span>
                            </div>
                            <BoxPlotComparison data={item.boxPlotData} />
                          </article>
                        ))}
                      </div>
                    </div>
                  ) : null}

                  <div className="quality-section">
                    <h3>Select Columns to Visualize</h3>
                    <StreamlitMultiSelect
                      label="Choose columns for detailed comparison"
                      helpText="Select which columns you want to see detailed visualizations for"
                      options={Object.keys(columnInfo)}
                      selected={selectedColumns}
                      open={isQualitySelectorOpen}
                      onToggleOpen={() => setIsQualitySelectorOpen((current) => !current)}
                      onToggleOption={toggleColumnSelection}
                      onClearAll={() => setSelectedColumns([])}
                    />
                  </div>

                  {!deferredSelectedColumns.length ? (
                    <div className="message-banner is-warning">Please select at least one column to visualize.</div>
                  ) : null}

                  {deferredSelectedColumns.map((column) => {
                    const details = columnInfo[column]
                    const style = TYPE_STYLES[details?.type] ?? TYPE_STYLES.unknown
                    const numericReal = getNumericValues(originalRows, column)
                    const numericSynthetic = getNumericValues(syntheticRows, column)
                    const categoricalReal = getCategoricalValues(originalRows, column)
                    const categoricalSynthetic = getCategoricalValues(syntheticRows, column)
                    const realTop = topFrequencies(categoricalReal)
                    const syntheticTop = topFrequencies(categoricalSynthetic)
                    const realMean = mean(numericReal)
                    const syntheticMean = mean(numericSynthetic)
                    const realStd = sampleStdDev(numericReal)
                    const syntheticStd = sampleStdDev(numericSynthetic)

                    return (
                      <div key={column} className="quality-column-panel">
                        <h3>{column}</h3>
                        <div className="quality-type-line">
                          <strong>Type:</strong> {style.emoji} {style.label}
                        </div>

                        {details.type === 'continuous_numeric' || details.type === 'discrete_numeric' ? (
                          <>
                            <div className="quality-distribution-grid">
                              <div className="quality-text-block">
                                <h4>Real Data Stats</h4>
                                <ul className="quality-stat-list">
                                  <li>Mean: {formatStat(realMean)}</li>
                                  <li>Std: {formatStat(realStd)}</li>
                                  <li>Min: {formatStat(Math.min(...numericReal))}</li>
                                  <li>Max: {formatStat(Math.max(...numericReal))}</li>
                                </ul>
                              </div>
                              <div className="quality-text-block">
                                <h4>Synthetic Data Stats</h4>
                                <ul className="quality-stat-list">
                                  <li>Mean: {formatStat(syntheticMean)}</li>
                                  <li>Std: {formatStat(syntheticStd)}</li>
                                  <li>Min: {formatStat(Math.min(...numericSynthetic))}</li>
                                  <li>Max: {formatStat(Math.max(...numericSynthetic))}</li>
                                </ul>
                              </div>
                            </div>

                            <div className="quality-chart-block">
                              <h4>Pattern Similarity</h4>
                              <ResponsiveContainer width="100%" height={320}>
                                <LineChart data={buildHistogramData(numericReal, numericSynthetic)}>
                                  <CartesianGrid strokeDasharray="3 3" />
                                  <XAxis dataKey="label" hide />
                                  <YAxis />
                                  <Tooltip />
                                  <Legend />
                                  <Line
                                    type="monotone"
                                    dataKey="real"
                                    stroke="#1f77b4"
                                    strokeWidth={3}
                                    dot={false}
                                    name="Real Data"
                                  />
                                  <Line
                                    type="monotone"
                                    dataKey="synthetic"
                                    stroke="#d62728"
                                    strokeWidth={3}
                                    dot={false}
                                    name="Synthetic Data"
                                  />
                                </LineChart>
                              </ResponsiveContainer>
                            </div>

                            <div className="quality-chart-block">
                              <h4>Outlier Box Plot</h4>
                              <BoxPlotComparison data={buildBoxPlotData(numericReal, numericSynthetic)} />
                            </div>
                          </>
                        ) : null}

                        {['categorical', 'boolean', 'email', 'phone'].includes(details.type) ? (
                          <>
                            <div className="quality-distribution-grid">
                              <div className="quality-text-block">
                                <h4>Real Data Distribution</h4>
                                <ul className="quality-stat-list">
                                  {realTop.map((item) => (
                                    <li key={`${column}-real-${item.value}`}>{item.value}: {(item.probability * 100).toFixed(1)}%</li>
                                  ))}
                                </ul>
                              </div>
                              <div className="quality-text-block">
                                <h4>Synthetic Data Distribution</h4>
                                <ul className="quality-stat-list">
                                  {syntheticTop.map((item) => (
                                    <li key={`${column}-synthetic-${item.value}`}>{item.value}: {(item.probability * 100).toFixed(1)}%</li>
                                  ))}
                                </ul>
                              </div>
                            </div>

                            <div className="quality-chart-block">
                              <h4>{column} - Pattern Similarity</h4>
                              <ResponsiveContainer width="100%" height={290}>
                                <LineChart
                                  data={buildFrequencyData(categoricalReal, categoricalSynthetic)}
                                  margin={{ top: 8, right: 48, left: 0, bottom: 26 }}
                                >
                                  <CartesianGrid strokeDasharray="3 3" />
                                  <XAxis
                                    dataKey="category"
                                    angle={-40}
                                    textAnchor="end"
                                    height={96}
                                    interval={0}
                                    tick={{ fontSize: 10 }}
                                  />
                                  <YAxis />
                                  <Tooltip />
                                  <Legend
                                    layout="vertical"
                                    align="right"
                                    verticalAlign="middle"
                                    wrapperStyle={{ fontSize: '12px', paddingLeft: '12px' }}
                                  />
                                  <Line
                                    type="monotone"
                                    dataKey="real"
                                    stroke="#1f77b4"
                                    strokeWidth={3}
                                    dot={{ r: 3 }}
                                    name="Real Data"
                                  />
                                  <Line
                                    type="monotone"
                                    dataKey="synthetic"
                                    stroke="#d62728"
                                    strokeWidth={3}
                                    dot={{ r: 3 }}
                                    name="Synthetic Data"
                                  />
                                </LineChart>
                              </ResponsiveContainer>
                            </div>
                          </>
                        ) : null}

                        {details.type === 'datetime' ? (
                          <div className="message-banner is-info">
                            DateTime column. Range preserved from {new Date(details.min).toLocaleString()} to{' '}
                            {new Date(details.max).toLocaleString()}.
                          </div>
                        ) : null}

                        <hr className="quality-divider" />
                      </div>
                    )
                  })}
                </>
              ) : (
                <div className="message-banner is-warning">
                  Please generate synthetic data first (Tab 1: Upload &amp; Generate).
                </div>
              )}
            </div>
          </section>
        ) : null}

        {activeTab === 'comparison' ? (
          <section className="content-panel">
            <div className="section-block">
              {generationComplete ? (
                <>
                  <div className="comparison-hero">
                    <div className="comparison-hero-copy">
                      <p className="comparison-kicker">Real vs synthetic</p>
                      <h2>Detailed Statistical Comparison</h2>
                      <p className="section-copy">
                        Inspect where the generated dataset stays tight, where drift opens up, and how well numeric,
                        categorical, and temporal signals carry over from the source data.
                      </p>
                    </div>

                    <div className="comparison-summary-grid">
                      <div className="comparison-summary-card">
                        <div className="comparison-summary-label">Rows compared</div>
                        <div className="comparison-summary-value">
                          {originalRows.length.toLocaleString()} / {syntheticRows.length.toLocaleString()}
                        </div>
                        <div className="comparison-summary-detail">original vs synthetic</div>
                      </div>
                      <div className="comparison-summary-card">
                        <div className="comparison-summary-label">Numeric columns</div>
                        <div className="comparison-summary-value">{numericComparisonColumns.length}</div>
                        <div className="comparison-summary-detail">
                          avg mean drift {formatPercent(averageNumericMeanDiff)}
                        </div>
                      </div>
                      <div className="comparison-summary-card">
                        <div className="comparison-summary-label">Categorical columns</div>
                        <div className="comparison-summary-value">{categoricalComparisonColumns.length}</div>
                        <div className="comparison-summary-detail">
                          avg distribution drift {formatPercent(averageCategoricalDrift)}
                        </div>
                      </div>
                      <div className="comparison-summary-card">
                        <div className="comparison-summary-label">DateTime columns</div>
                        <div className="comparison-summary-value">{datetimeComparisonColumns.length}</div>
                        <div className="comparison-summary-detail">
                          avg span drift {formatPercent(averageDatetimeSpanDiff)}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="comparison-panel">
                    <div className="comparison-panel-head">
                      <div>
                        <h3>Numeric Columns</h3>
                        <p className="section-copy">Track average shift, spread drift, and distribution shape.</p>
                      </div>
                      <div className="comparison-panel-count">{numericComparisonColumns.length} columns</div>
                    </div>

                    {numericComparisonColumns.length ? (
                      <>
                        <DataTable
                          rows={numericComparison}
                          maxRows={numericComparison.length || 10}
                          emptyMessage="No numeric columns detected."
                          dataframe
                        />

                        <div className="comparison-card-grid comparison-card-grid-numeric">
                          {numericComparisonColumns.map((item) => {
                            const style = TYPE_STYLES[item.details.type] ?? TYPE_STYLES.unknown

                            return (
                              <article key={item.column} className="comparison-card">
                                <div className="comparison-card-head">
                                  <div>
                                    <h4>{item.column}</h4>
                                    <p className="comparison-card-copy">
                                      {style.emoji} {style.label}
                                    </p>
                                  </div>
                                  <span className={`comparison-badge is-${item.meanTone.tone}`}>
                                    {item.meanTone.label}
                                  </span>
                                </div>

                                <div className="comparison-pill-row">
                                  <span className="comparison-pill">Mean Δ {formatPercent(item.meanDiffPct)}</span>
                                  <span className="comparison-pill">Std Δ {formatPercent(item.stdDiffPct)}</span>
                                  <span className={`comparison-badge is-${item.stdTone.tone}`}>
                                    Spread {item.stdTone.label}
                                  </span>
                                </div>

                                <div className="comparison-detail-grid">
                                  <div className="comparison-detail-card">
                                    <div className="comparison-detail-label">Real</div>
                                    <ul className="comparison-detail-list">
                                      <li>Mean {formatStat(item.realMean)}</li>
                                      <li>Std {formatStat(item.realStd)}</li>
                                      <li>
                                        Range {formatStat(item.realMin)} to {formatStat(item.realMax)}
                                      </li>
                                    </ul>
                                  </div>
                                  <div className="comparison-detail-card">
                                    <div className="comparison-detail-label">Synthetic</div>
                                    <ul className="comparison-detail-list">
                                      <li>Mean {formatStat(item.syntheticMean)}</li>
                                      <li>Std {formatStat(item.syntheticStd)}</li>
                                      <li>
                                        Range {formatStat(item.syntheticMin)} to {formatStat(item.syntheticMax)}
                                      </li>
                                    </ul>
                                  </div>
                                </div>

                                <div className="comparison-chart-shell">
                                  <h5>Pattern similarity</h5>
                                  {item.histogramData.length ? (
                                    <ResponsiveContainer width="100%" height={240}>
                                      <LineChart data={item.histogramData}>
                                        <CartesianGrid strokeDasharray="3 3" />
                                        <XAxis dataKey="label" hide />
                                        <YAxis />
                                        <Tooltip />
                                        <Legend />
                                        <Line
                                          type="monotone"
                                          dataKey="real"
                                          stroke="#1f77b4"
                                          strokeWidth={3}
                                          dot={false}
                                          name="Real Data"
                                        />
                                        <Line
                                          type="monotone"
                                          dataKey="synthetic"
                                          stroke="#d62728"
                                          strokeWidth={3}
                                          dot={false}
                                          name="Synthetic Data"
                                        />
                                      </LineChart>
                                    </ResponsiveContainer>
                                  ) : (
                                    <div className="empty-state">Not enough numeric values to compare.</div>
                                  )}
                                </div>

                                <div className="comparison-chart-shell">
                                  <h5>Outlier box plot</h5>
                                  <BoxPlotComparison data={item.boxPlotData} />
                                </div>
                              </article>
                            )
                          })}
                        </div>
                      </>
                    ) : (
                      <div className="empty-state">No numeric columns detected.</div>
                    )}
                  </div>

                  <div className="comparison-panel">
                    <div className="comparison-panel-head">
                      <div>
                        <h3>Categorical Columns</h3>
                        <p className="section-copy">
                          Compare label mix, top-category balance, and coverage gaps across generated output.
                        </p>
                      </div>
                      <div className="comparison-panel-count">{categoricalComparisonColumns.length} columns</div>
                    </div>

                    {categoricalComparisonColumns.length ? (
                      <div className="comparison-card-grid">
                        {categoricalComparisonColumns.map((item) => {
                          const style = TYPE_STYLES[item.details.type] ?? TYPE_STYLES.unknown

                          return (
                            <article key={item.column} className="comparison-card">
                              <div className="comparison-card-head">
                                <div>
                                  <h4>{item.column}</h4>
                                  <p className="comparison-card-copy">
                                    {style.emoji} {style.label}
                                  </p>
                                </div>
                                <span className={`comparison-badge is-${item.driftTone.tone}`}>
                                  {item.driftTone.label}
                                </span>
                              </div>

                              <div className="comparison-pill-row">
                                <span className="comparison-pill">Drift {formatPercent(item.driftPct)}</span>
                                <span className="comparison-pill">Real unique {item.realUnique}</span>
                                <span className="comparison-pill">Synthetic unique {item.syntheticUnique}</span>
                                <span className="comparison-pill">
                                  Missing in synthetic {item.missingFromSynthetic}
                                </span>
                              </div>

                              <div className="comparison-detail-grid">
                                <div className="comparison-detail-card">
                                  <div className="comparison-detail-label">Real top values</div>
                                  <ul className="comparison-detail-list">
                                    {item.realTop.map((entry) => (
                                      <li key={`${item.column}-real-${entry.value}`}>
                                        {entry.value}: {formatPercent(entry.probability * 100)}
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                                <div className="comparison-detail-card">
                                  <div className="comparison-detail-label">Synthetic top values</div>
                                  <ul className="comparison-detail-list">
                                    {item.syntheticTop.map((entry) => (
                                      <li key={`${item.column}-synthetic-${entry.value}`}>
                                        {entry.value}: {formatPercent(entry.probability * 100)}
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                              </div>

                              <div className="comparison-chart-shell">
                                <h5>Pattern similarity</h5>
                                {item.frequencyData.length ? (
                                  <ResponsiveContainer width="100%" height={260}>
                                    <LineChart
                                      data={item.frequencyData}
                                      margin={{ top: 8, right: 16, left: 0, bottom: 58 }}
                                    >
                                      <CartesianGrid strokeDasharray="3 3" />
                                      <XAxis
                                        dataKey="category"
                                        angle={-28}
                                        textAnchor="end"
                                        height={72}
                                        interval={0}
                                        tick={{ fontSize: 10 }}
                                        tickFormatter={truncateAxisLabel}
                                      />
                                      <YAxis />
                                      <Tooltip />
                                      <Legend />
                                      <Line
                                        type="monotone"
                                        dataKey="real"
                                        stroke="#1f77b4"
                                        strokeWidth={3}
                                        dot={{ r: 3 }}
                                        name="Real Data"
                                      />
                                      <Line
                                        type="monotone"
                                        dataKey="synthetic"
                                        stroke="#d62728"
                                        strokeWidth={3}
                                        dot={{ r: 3 }}
                                        name="Synthetic Data"
                                      />
                                    </LineChart>
                                  </ResponsiveContainer>
                                ) : (
                                  <div className="empty-state">No category frequencies available to compare.</div>
                                )}
                              </div>

                              <DataTable
                                rows={item.summaryRows}
                                maxRows={item.summaryRows.length || 6}
                                compact
                                emptyMessage="No category rows to compare."
                              />
                            </article>
                          )
                        })}
                      </div>
                    ) : (
                      <div className="empty-state">No categorical columns detected.</div>
                    )}
                  </div>

                  <div className="comparison-panel">
                    <div className="comparison-panel-head">
                      <div>
                        <h3>DateTime Columns</h3>
                        <p className="section-copy">
                          Check whether the generated set preserves the same temporal window as the source data.
                        </p>
                      </div>
                      <div className="comparison-panel-count">{datetimeComparisonColumns.length} columns</div>
                    </div>

                    {datetimeComparisonColumns.length ? (
                      <>
                        <div className="comparison-card-grid comparison-card-grid-datetime">
                          {datetimeComparisonColumns.map((item) => (
                            <article key={item.column} className="comparison-card">
                              <div className="comparison-card-head">
                                <div>
                                  <h4>{item.column}</h4>
                                  <p className="comparison-card-copy">🟠 datetime</p>
                                </div>
                                <span className={`comparison-badge is-${item.spanTone.tone}`}>
                                  {item.spanTone.label}
                                </span>
                              </div>

                              <div className="comparison-pill-row">
                                <span className="comparison-pill">Span Δ {formatPercent(item.spanDiffPct)}</span>
                                <span className="comparison-pill">Real {formatDurationLabel(item.realSpan)}</span>
                                <span className="comparison-pill">
                                  Synthetic {formatDurationLabel(item.syntheticSpan)}
                                </span>
                              </div>

                              <div className="comparison-detail-grid">
                                <div className="comparison-detail-card">
                                  <div className="comparison-detail-label">Real range</div>
                                  <ul className="comparison-detail-list">
                                    <li>Start {formatDateTime(item.realMin)}</li>
                                    <li>End {formatDateTime(item.realMax)}</li>
                                  </ul>
                                </div>
                                <div className="comparison-detail-card">
                                  <div className="comparison-detail-label">Synthetic range</div>
                                  <ul className="comparison-detail-list">
                                    <li>Start {formatDateTime(item.syntheticMin)}</li>
                                    <li>End {formatDateTime(item.syntheticMax)}</li>
                                  </ul>
                                </div>
                              </div>
                            </article>
                          ))}
                        </div>

                        <DataTable
                          rows={datetimeComparison}
                          maxRows={datetimeComparison.length || 10}
                          emptyMessage="No DateTime columns detected."
                          dataframe
                        />
                      </>
                    ) : (
                      <div className="empty-state">No DateTime columns detected.</div>
                    )}
                  </div>
                </>
              ) : (
                <div className="message-banner is-warning">
                  Please generate synthetic data first (Tab 1: Upload &amp; Generate).
                </div>
              )}
            </div>
          </section>
        ) : null}

        {activeTab === 'download' ? (
          <section className="content-panel">
            <div className="section-block">
              <h2>Download Your Results</h2>
              {generationComplete ? (
                <>
                  <div className="download-section">
                    <h3>Download Options</h3>
                  </div>

                  <div className="download-section">
                    <div className="download-preview-label">Synthetic Data Preview:</div>
                    <DataTable rows={syntheticRows} maxRows={20} dataframe showIndex />
                  </div>

                  <div className="button-grid">
                    <button type="button" className="primary-button" onClick={downloadSyntheticCsv}>
                      Download Synthetic Data (CSV)
                    </button>
                    <button type="button" className="secondary-button" onClick={downloadColumnInfo}>
                      {'\uD83D\uDCCB'} Download Column Info (JSON)
                    </button>
                  </div>

                  <div className="download-section">
                    <h3>File Information</h3>
                    <div className="upload-metrics-row">
                      <div className="upload-metric">
                        <div className="upload-metric-label">Rows</div>
                        <div className="upload-metric-value">{syntheticRows.length.toLocaleString()}</div>
                      </div>
                      <div className="upload-metric">
                        <div className="upload-metric-label">Columns</div>
                        <div className="upload-metric-value">{syntheticRows[0] ? Object.keys(syntheticRows[0]).length : 0}</div>
                      </div>
                      <div className="upload-metric">
                        <div className="upload-metric-label">Estimated Size</div>
                        <div className="upload-metric-value">{syntheticSizeMb.toFixed(2)} MB</div>
                      </div>
                    </div>
                  </div>

                  <div className="info-banner">
                    <strong>Tip:</strong> The synthetic data has the same structure as your original data and can be
                    used directly in your applications, testing environments, or shared with partners without privacy
                    concerns.
                  </div>
                </>
              ) : (
                <div className="message-banner is-warning">
                  Please generate synthetic data first (Tab 1: Upload &amp; Generate).
                </div>
              )}
            </div>
          </section>
        ) : null}

        {activeTab === 'benchmarks' ? (
          <section className="content-panel">
            <div className="section-block">
              <h2>Performance Benchmarks</h2>
              <p className="section-copy">
                Test how fast the tool generates synthetic data at different row counts.
              </p>

              {hasData ? (
                <>
                  <div className="benchmark-section">
                    <h3>Select Row Sizes to Benchmark</h3>
                    <div className="benchmark-checkbox-grid">
                      <div className="benchmark-checkbox-column">
                        {BENCHMARK_OPTIONS.slice(0, 3).map((size) => (
                          <label key={size} className="benchmark-checkbox">
                            <input
                              type="checkbox"
                              checked={benchmarkSelection[size]}
                              onChange={() => toggleBenchmark(size)}
                            />
                            <span>{size.toLocaleString()} rows</span>
                          </label>
                        ))}
                      </div>
                      <div className="benchmark-checkbox-column">
                        {BENCHMARK_OPTIONS.slice(3).map((size) => (
                          <label key={size} className="benchmark-checkbox">
                            <input
                              type="checkbox"
                              checked={benchmarkSelection[size]}
                              onChange={() => toggleBenchmark(size)}
                            />
                            <span>{size.toLocaleString()} rows</span>
                          </label>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="inline-actions">
                    <button type="button" className="primary-button" onClick={runBenchmarks} disabled={isPending}>
                      Run Benchmark
                    </button>
                  </div>

                  {statusMessage ? (
                    <div className="benchmark-status-group">
                      <div className="benchmark-progress-bar" aria-hidden="true" />
                      <div className={`message-banner is-${statusTone === 'warning' ? 'warning' : statusTone}`}>
                        {statusMessage}
                      </div>
                    </div>
                  ) : null}

                  {benchmarkResults.length ? (
                    <>
                      <div className="benchmark-section">
                        <h3>Results</h3>
                        <DataTable
                          rows={benchmarkResults.map((result) => ({
                            'Rows Generated': result.rowsGenerated.toLocaleString(),
                            'Time (seconds)': result.timeSeconds,
                            'Rows per Second': result.rowsPerSecond.toLocaleString()
                          }))}
                          maxRows={benchmarkResults.length}
                          dataframe
                          showIndex
                        />
                      </div>

                      <div className="benchmark-section">
                        <h3>Generation Time vs Row Count</h3>
                        <ResponsiveContainer width="100%" height={400}>
                          <LineChart data={benchmarkChartData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis
                              dataKey="rowsGenerated"
                              tickFormatter={(value) => Number(value).toLocaleString()}
                              label={{ value: 'Rows Generated', position: 'insideBottom', offset: -6 }}
                            />
                            <YAxis
                              label={{ value: 'Time (seconds)', angle: -90, position: 'insideLeft' }}
                            />
                            <Tooltip formatter={(value) => `${value}s`} />
                            <Line
                              type="monotone"
                              dataKey="timeSeconds"
                              stroke="#1f77b4"
                              strokeWidth={2}
                              dot={{ r: 4 }}
                              name="Generation Time"
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>

                      {benchmarkChartData.length ? (
                        <div className="benchmark-section">
                          <h3>Summary</h3>
                          <div className="benchmark-summary-row">
                            <div className="benchmark-summary-metric">
                              <div className="benchmark-summary-label">Fastest Run</div>
                              <div className="benchmark-summary-value">{fastestBenchmark.timeSeconds}s</div>
                              <div className="benchmark-summary-delta">↑ {fastestBenchmark.rowsGenerated.toLocaleString()}</div>
                            </div>
                            <div className="benchmark-summary-metric">
                              <div className="benchmark-summary-label">Slowest Run</div>
                              <div className="benchmark-summary-value">{slowestBenchmark.timeSeconds}s</div>
                              <div className="benchmark-summary-delta">↑ {slowestBenchmark.rowsGenerated.toLocaleString()}</div>
                            </div>
                            <div className="benchmark-summary-metric">
                              <div className="benchmark-summary-label">Avg Rows/Second</div>
                              <div className="benchmark-summary-value">{averageBenchmarkRps.toLocaleString()}</div>
                            </div>
                          </div>
                        </div>
                      ) : null}
                    </>
                  ) : null}
                </>
              ) : (
                <div className="message-banner is-warning">
                  Please upload a dataset first in Upload &amp; Generate.
                </div>
              )}
            </div>
          </section>
        ) : null}

        <footer className="app-footer">
          <strong>Auto-Detecting Synthetic Data Generator</strong>
          <span>Developed by Robel</span>
          <span>© 2026 | Version 1.1</span>
        </footer>
      </main>
    </div>
  )
}
