import Papa from 'papaparse'

const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/i
const PHONE_PATTERN = /^\+?[\d\s().-]{7,}$/
const DATE_ONLY_PATTERN = /^\d{4}-\d{2}-\d{2}$/
const BOOLEAN_TOKENS = new Set(['true', 'false', 'yes', 'no', '1', '0'])

function isPlainObject(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isNestedValue(value) {
  return Array.isArray(value) || isPlainObject(value)
}

function toText(value) {
  if (value === null || value === undefined) {
    return ''
  }

  return String(value).trim()
}

function parseMaybeNumber(value) {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }

  const text = toText(value).replaceAll(',', '')
  if (!text || !/^[-+]?\d*\.?\d+$/.test(text)) {
    return null
  }

  const numeric = Number(text)
  return Number.isFinite(numeric) ? numeric : null
}

function parseMaybeDate(value) {
  if (value instanceof Date && !Number.isNaN(value.getTime())) {
    return value
  }

  const text = toText(value)
  if (!text) {
    return null
  }

  const timestamp = Date.parse(text)
  return Number.isNaN(timestamp) ? null : new Date(timestamp)
}

function toBooleanToken(value) {
  const text = toText(value).toLowerCase()
  if (!BOOLEAN_TOKENS.has(text)) {
    return null
  }

  return text === 'true' || text === 'yes' || text === '1'
}

function average(values) {
  if (!values.length) {
    return 0
  }

  return values.reduce((sum, value) => sum + value, 0) / values.length
}

function stddev(values, mean) {
  if (values.length <= 1) {
    return 0
  }

  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length
  return Math.sqrt(variance)
}

function quantile(values, q) {
  if (!values.length) {
    return 0
  }

  const sorted = [...values].sort((a, b) => a - b)
  const position = (sorted.length - 1) * q
  const lower = Math.floor(position)
  const upper = Math.ceil(position)

  if (lower === upper) {
    return sorted[lower]
  }

  const weight = position - lower
  return sorted[lower] * (1 - weight) + sorted[upper] * weight
}

function countDecimals(values) {
  let maxDecimals = 0

  values.forEach((value) => {
    const parts = String(value).split('.')
    if (parts[1]) {
      maxDecimals = Math.max(maxDecimals, parts[1].length)
    }
  })

  return Math.min(maxDecimals, 4)
}

function countFrequencies(values) {
  const counts = new Map()

  values.forEach((value) => {
    const key = typeof value === 'string' ? value : String(value)
    counts.set(key, (counts.get(key) ?? 0) + 1)
  })

  return Array.from(counts.entries())
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([value, count]) => ({ value, count, probability: count / values.length }))
}

function toRecordRows(raw) {
  if (Array.isArray(raw)) {
    return raw
  }

  if (!isPlainObject(raw)) {
    throw new Error('Unsupported JSON format. Please provide a flat JSON file.')
  }

  const entries = Object.entries(raw)
  const firstValue = entries[0]?.[1]

  if (isPlainObject(firstValue)) {
    const rowKeys = Array.from(
      new Set(entries.flatMap(([, value]) => (isPlainObject(value) ? Object.keys(value) : [])))
    )

    return rowKeys.map((rowKey) => {
      const row = {}
      entries.forEach(([column, mapping]) => {
        row[column] = isPlainObject(mapping) ? mapping[rowKey] : mapping
      })
      return row
    })
  }

  return [raw]
}

function normalizeRows(rows) {
  const objectRows = rows.filter(Boolean).map((row) => (isPlainObject(row) ? row : {}))
  const columns = Array.from(new Set(objectRows.flatMap((row) => Object.keys(row))))

  return objectRows.map((row) => {
    const normalized = {}
    columns.forEach((column) => {
      normalized[column] = row[column] ?? null
    })
    return normalized
  })
}

function normalizeCsvHeaderName(rawHeader, index) {
  if (rawHeader === null || rawHeader === undefined) {
    return `Unnamed: ${index}`
  }

  const headerText = String(rawHeader)
  return headerText.trim() === '' ? `Unnamed: ${index}` : headerText
}

function buildPandasLikeHeaders(headerRow, columnCount) {
  const seen = new Map()

  return Array.from({ length: columnCount }, (_, index) => {
    const baseName = normalizeCsvHeaderName(headerRow[index], index)
    const duplicateCount = seen.get(baseName) ?? 0
    seen.set(baseName, duplicateCount + 1)

    return duplicateCount === 0 ? baseName : `${baseName}.${duplicateCount}`
  })
}

function csvMatrixToRecords(matrix) {
  if (!matrix.length) {
    return { rows: [], columns: [] }
  }

  const columnCount = Math.max(...matrix.map((row) => (Array.isArray(row) ? row.length : 0)))
  const headerRow = Array.isArray(matrix[0]) ? matrix[0] : []
  const columns = buildPandasLikeHeaders(headerRow, columnCount)
  const rows = matrix.slice(1).map((row) =>
    Object.fromEntries(
      columns.map((column, index) => {
        const value = Array.isArray(row) ? row[index] : undefined
        return [column, value === '' || value === undefined ? null : value]
      })
    )
  )

  return { rows, columns }
}

function seededRandom(seed) {
  let state = seed >>> 0

  return () => {
    state += 0x6d2b79f5
    let t = state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function gaussian(random) {
  let u = 0
  let v = 0

  while (u === 0) u = random()
  while (v === 0) v = random()

  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

function chooseWeighted(random, frequencies) {
  const target = random()
  let running = 0

  for (const item of frequencies) {
    running += item.probability
    if (target <= running) {
      return item.value
    }
  }

  return frequencies[frequencies.length - 1]?.value ?? ''
}

function formatDate(date, dateOnly) {
  if (dateOnly) {
    return date.toISOString().slice(0, 10)
  }

  return date.toISOString().slice(0, 19).replace('T', ' ')
}

function mutatePhone(random, template) {
  return template.replace(/\d/g, () => Math.floor(random() * 10).toString())
}

export async function parseUploadedFile(file) {
  const extension = file.name.split('.').pop()?.toLowerCase()
  let rows = []
  let columns = []

  if (extension === 'csv') {
    const text = await file.text()
    const parsed = Papa.parse(text, {
      header: false,
      skipEmptyLines: true
    })

    if (parsed.errors.length) {
      throw new Error(parsed.errors[0].message)
    }

    const csvResult = csvMatrixToRecords(parsed.data)
    rows = csvResult.rows
    columns = csvResult.columns
  } else if (extension === 'json') {
    const text = await file.text()
    rows = toRecordRows(JSON.parse(text))
  } else {
    throw new Error('Unsupported file type. Please upload a CSV or JSON file.')
  }

  const normalized = normalizeRows(rows)
  const normalizedColumns = columns.length ? columns : normalized[0] ? Object.keys(normalized[0]) : []

  if (!normalizedColumns.length) {
    throw new Error('The uploaded file did not contain any columns.')
  }

  const nestedColumns = normalizedColumns.filter((column) =>
    normalized.some((row) => isNestedValue(row[column]))
  )

  const flatRows = normalized.map((row) =>
    Object.fromEntries(Object.entries(row).filter(([column]) => !nestedColumns.includes(column)))
  )
  const flatColumns = normalizedColumns.filter((column) => !nestedColumns.includes(column))

  return {
    rows: flatRows,
    columns: flatColumns,
    nestedColumns,
    fileSizeKb: file.size / 1024
  }
}

export async function parseReferenceFile(file) {
  const text = await file.text()
  const parsed = Papa.parse(text, {
    skipEmptyLines: true
  })

  if (parsed.errors.length) {
    throw new Error(parsed.errors[0].message)
  }

  const values = parsed.data
    .flat()
    .map((value) => toText(value))
    .filter(Boolean)

  return Array.from(new Set(values))
}

export function detectColumnInfo(rows, discreteThreshold = 0.05) {
  const columns = rows[0] ? Object.keys(rows[0]) : []
  const info = {}

  columns.forEach((column) => {
    const nonEmpty = rows.map((row) => row[column]).filter((value) => toText(value) !== '')
    const textValues = nonEmpty.map((value) => toText(value))
    const numericValues = nonEmpty.map(parseMaybeNumber).filter((value) => value !== null)
    const booleanValues = nonEmpty.map(toBooleanToken).filter((value) => value !== null)
    const dateValues = nonEmpty.map(parseMaybeDate).filter(Boolean)
    const uniqueTextCount = new Set(textValues).size
    const isEmail = textValues.length > 0 && textValues.every((value) => EMAIL_PATTERN.test(value))
    const isPhone = textValues.length > 0 && textValues.every((value) => PHONE_PATTERN.test(value))
    const isBoolean =
      nonEmpty.length > 0 &&
      (booleanValues.length === nonEmpty.length ||
        (numericValues.length !== nonEmpty.length && uniqueTextCount === 2))
    const isDate = dateValues.length === nonEmpty.length && nonEmpty.length > 0
    const isNumeric = numericValues.length === nonEmpty.length && nonEmpty.length > 0

    if (!nonEmpty.length) {
      info[column] = {
        type: 'empty',
        method: 'skip'
      }
      return
    }

    if (isEmail) {
      const frequencies = countFrequencies(textValues)
      const domains = countFrequencies(textValues.map((value) => value.split('@')[1] ?? 'example.com'))
      const fragments = textValues
        .map((value) => value.split('@')[0] ?? '')
        .filter(Boolean)
        .slice(0, 200)

      info[column] = {
        type: 'email',
        method: 'email_generation',
        frequencies,
        domains,
        fragments
      }
      return
    }

    if (isPhone) {
      info[column] = {
        type: 'phone',
        method: 'phone_generation',
        frequencies: countFrequencies(textValues),
        patterns: textValues.slice(0, 100)
      }
      return
    }

    if (isBoolean) {
      info[column] = {
        type: 'boolean',
        method: 'categorical_sampling',
        frequencies: countFrequencies(booleanValues.length === nonEmpty.length ? booleanValues : textValues)
      }
      return
    }

    if (isDate) {
      const timestamps = dateValues.map((value) => value.getTime())
      info[column] = {
        type: 'datetime',
        method: 'datetime_generation',
        min: Math.min(...timestamps),
        max: Math.max(...timestamps),
        dateOnly: textValues.every((value) => DATE_ONLY_PATTERN.test(value))
      }
      return
    }

    if (isNumeric) {
      const uniqueRatio = new Set(numericValues).size / Math.max(1, numericValues.length)
      if (uniqueRatio <= discreteThreshold) {
        info[column] = {
          type: 'discrete_numeric',
          method: 'categorical_sampling',
          frequencies: countFrequencies(numericValues),
          stats: {
            min: Math.min(...numericValues),
            max: Math.max(...numericValues)
          }
        }
        return
      }

      const mean = average(numericValues)
      const deviation = stddev(numericValues, mean)

      info[column] = {
        type: 'continuous_numeric',
        method: 'gmm',
        stats: {
          mean,
          std: deviation,
          min: Math.min(...numericValues),
          max: Math.max(...numericValues),
          q1: quantile(numericValues, 0.25),
          median: quantile(numericValues, 0.5),
          q3: quantile(numericValues, 0.75),
          decimals: countDecimals(numericValues)
        }
      }
      return
    }

    info[column] = {
      type: 'categorical',
      method: 'categorical_sampling',
      frequencies: countFrequencies(textValues)
    }
  })

  return info
}

export function buildDetectionSummary(columnInfo, referenceTables = {}) {
  return Object.entries(columnInfo).map(([column, details]) => ({
    Column: column,
    'Detected Type': details.type,
    'Generation Method': referenceTables[column]?.length
      ? `reference_table_sampling (${referenceTables[column].length.toLocaleString()} values)`
      : details.method
  }))
}

export function generateSyntheticRows({
  rows,
  columnInfo,
  referenceTables = {},
  numRows,
  seed
}) {
  const random = seededRandom(seed)
  const columns = rows[0] ? Object.keys(rows[0]) : []
  const generatedRows = []

  for (let index = 0; index < numRows; index += 1) {
    const nextRow = {}

    columns.forEach((column) => {
      const details = columnInfo[column]
      const referenceValues = referenceTables[column]

      if (referenceValues?.length) {
        nextRow[column] = referenceValues[Math.floor(random() * referenceValues.length)]
        return
      }

      if (!details) {
        nextRow[column] = ''
        return
      }

      switch (details.type) {
        case 'continuous_numeric': {
          const sample = details.stats.mean + gaussian(random) * (details.stats.std || 1)
          const bounded = Math.max(details.stats.min, Math.min(details.stats.max, sample))
          nextRow[column] = Number(bounded.toFixed(details.stats.decimals))
          break
        }
        case 'discrete_numeric': {
          nextRow[column] = Number(chooseWeighted(random, details.frequencies))
          break
        }
        case 'boolean': {
          nextRow[column] = chooseWeighted(random, details.frequencies)
          break
        }
        case 'categorical': {
          nextRow[column] = chooseWeighted(random, details.frequencies)
          break
        }
        case 'datetime': {
          const timestamp = details.min + random() * (details.max - details.min || 1)
          nextRow[column] = formatDate(new Date(timestamp), details.dateOnly)
          break
        }
        case 'email': {
          const domain = chooseWeighted(random, details.domains) || 'example.com'
          const fragment =
            details.fragments[Math.floor(random() * details.fragments.length)] ?? `user${index + 1}`
          const suffix = Math.floor(random() * 10000)
          nextRow[column] = `${fragment.replace(/[^a-z0-9]/gi, '').toLowerCase() || 'user'}${suffix}@${domain}`
          break
        }
        case 'phone': {
          const template = details.patterns[Math.floor(random() * details.patterns.length)] ?? '(555) 123-4567'
          nextRow[column] = mutatePhone(random, template)
          break
        }
        default:
          nextRow[column] = ''
      }
    })

    generatedRows.push(nextRow)
  }

  return generatedRows
}

function getValues(rows, column, mapper = (value) => value) {
  return rows
    .map((row) => row[column])
    .filter((value) => toText(value) !== '')
    .map(mapper)
    .filter((value) => value !== null && value !== undefined && !Number.isNaN(value))
}

export function getNumericValues(rows, column) {
  return getValues(rows, column, parseMaybeNumber)
}

export function getCategoricalValues(rows, column) {
  return getValues(rows, column, (value) => String(value))
}

export function buildHistogramData(realValues, syntheticValues, bins = 12) {
  if (!realValues.length || !syntheticValues.length) {
    return []
  }

  const min = Math.min(...realValues, ...syntheticValues)
  const max = Math.max(...realValues, ...syntheticValues)
  const span = max - min || 1
  const step = span / bins
  const data = Array.from({ length: bins }, (_, index) => ({
    label: `${(min + step * index).toFixed(1)}-${(min + step * (index + 1)).toFixed(1)}`,
    real: 0,
    synthetic: 0
  }))

  realValues.forEach((value) => {
    const index = Math.min(bins - 1, Math.floor((value - min) / step))
    data[index].real += 1
  })

  syntheticValues.forEach((value) => {
    const index = Math.min(bins - 1, Math.floor((value - min) / step))
    data[index].synthetic += 1
  })

  return data
}

export function buildBoxSummary(realValues, syntheticValues) {
  if (!realValues.length || !syntheticValues.length) {
    return []
  }

  return [
    { label: 'Min', real: Math.min(...realValues), synthetic: Math.min(...syntheticValues) },
    { label: 'Q1', real: quantile(realValues, 0.25), synthetic: quantile(syntheticValues, 0.25) },
    { label: 'Median', real: quantile(realValues, 0.5), synthetic: quantile(syntheticValues, 0.5) },
    { label: 'Q3', real: quantile(realValues, 0.75), synthetic: quantile(syntheticValues, 0.75) },
    { label: 'Max', real: Math.max(...realValues), synthetic: Math.max(...syntheticValues) }
  ]
}

export function buildFrequencyData(realValues, syntheticValues, limit = 12) {
  const combined = Array.from(
    new Set([
      ...countFrequencies(realValues)
        .slice(0, limit)
        .map((item) => item.value),
      ...countFrequencies(syntheticValues)
        .slice(0, limit)
        .map((item) => item.value)
    ])
  )

  return combined.map((category) => ({
    category,
    real: realValues.filter((value) => String(value) === category).length,
    synthetic: syntheticValues.filter((value) => String(value) === category).length
  }))
}

export function topFrequencies(values, limit = 5) {
  return countFrequencies(values).slice(0, limit)
}

export function numericComparisonRows(originalRows, syntheticRows, columnInfo) {
  return Object.entries(columnInfo)
    .filter(([, details]) => details.type === 'continuous_numeric' || details.type === 'discrete_numeric')
    .map(([column]) => {
      const realValues = getNumericValues(originalRows, column)
      const syntheticValues = getNumericValues(syntheticRows, column)
      const realMean = average(realValues)
      const syntheticMean = average(syntheticValues)
      const meanDiff = Math.abs(realMean - syntheticMean)
      const diffPct = realMean !== 0 ? (meanDiff / Math.abs(realMean)) * 100 : 0

      return {
        Column: column,
        'Real Mean': realMean.toFixed(2),
        'Synthetic Mean': syntheticMean.toFixed(2),
        'Mean Diff %': `${diffPct.toFixed(1)}%`,
        'Real Std': stddev(realValues, realMean).toFixed(2),
        'Synthetic Std': stddev(syntheticValues, syntheticMean).toFixed(2)
      }
    })
}

export function datetimeComparisonRows(originalRows, syntheticRows, columnInfo) {
  return Object.entries(columnInfo)
    .filter(([, details]) => details.type === 'datetime')
    .map(([column]) => {
      const realDates = getValues(originalRows, column, parseMaybeDate)
      const syntheticDates = getValues(syntheticRows, column, parseMaybeDate)
      return {
        Column: column,
        'Real Min': realDates[0] ? formatDate(new Date(Math.min(...realDates.map((value) => value.getTime()))), false) : '—',
        'Real Max': realDates[0] ? formatDate(new Date(Math.max(...realDates.map((value) => value.getTime()))), false) : '—',
        'Synthetic Min': syntheticDates[0]
          ? formatDate(new Date(Math.min(...syntheticDates.map((value) => value.getTime()))), false)
          : '—',
        'Synthetic Max': syntheticDates[0]
          ? formatDate(new Date(Math.max(...syntheticDates.map((value) => value.getTime()))), false)
          : '—'
      }
    })
}

export function summarizeColumns(columnInfo) {
  const values = Object.values(columnInfo)

  return {
    numeric: values.filter(
      (item) => item.type === 'continuous_numeric' || item.type === 'discrete_numeric'
    ).length,
    categorical: values.filter((item) =>
      ['categorical', 'boolean', 'email', 'phone'].includes(item.type)
    ).length
  }
}

export function summarizeSourceColumns(rows) {
  const columns = rows[0] ? Object.keys(rows[0]) : []

  return columns.reduce(
    (summary, column) => {
      const rawValues = rows.map((row) => row[column])
      const nonEmpty = rawValues.filter((value) => toText(value) !== '')
      const numericValues = nonEmpty.map(parseMaybeNumber).filter((value) => value !== null)

      if (!nonEmpty.length || numericValues.length === nonEmpty.length) {
        summary.numeric += 1
      } else {
        summary.categorical += 1
      }

      return summary
    },
    { numeric: 0, categorical: 0 }
  )
}

export function rowsToCsv(rows) {
  return Papa.unparse(rows)
}

export function triggerDownload(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.append(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}
