import { useDeferredValue, useMemo, useState } from 'react'

const EMPTY_ROWS = []

function formatCell(value, emptyValue) {
  if (value === null || value === undefined || value === '') {
    return emptyValue
  }

  if (typeof value === 'number') {
    return Number.isInteger(value) ? value.toString() : value.toFixed(2)
  }

  if (typeof value === 'boolean') {
    return value ? 'true' : 'false'
  }

  return String(value)
}

function isEmptyCell(value) {
  return value === null || value === undefined || value === ''
}

function getNumericValue(value) {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : null
  }

  if (typeof value !== 'string') {
    return null
  }

  const normalized = value.trim().replaceAll(',', '')

  if (!normalized) {
    return null
  }

  const numberValue = Number(normalized)
  return Number.isFinite(numberValue) ? numberValue : null
}

function getDateValue(value) {
  if (value instanceof Date) {
    return Number.isFinite(value.getTime()) ? value.getTime() : null
  }

  if (typeof value !== 'string' || !/[/-]|\d{4}/.test(value)) {
    return null
  }

  const dateValue = Date.parse(value)
  return Number.isFinite(dateValue) ? dateValue : null
}

function compareCellValues(left, right) {
  const leftNumber = getNumericValue(left)
  const rightNumber = getNumericValue(right)

  if (leftNumber !== null && rightNumber !== null) {
    return leftNumber - rightNumber
  }

  const leftDate = getDateValue(left)
  const rightDate = getDateValue(right)

  if (leftDate !== null && rightDate !== null) {
    return leftDate - rightDate
  }

  if (typeof left === 'boolean' && typeof right === 'boolean') {
    return Number(left) - Number(right)
  }

  return String(left).localeCompare(String(right), undefined, {
    numeric: true,
    sensitivity: 'base'
  })
}

function getSortLabel(sortState, column) {
  if (sortState.column !== column) {
    return 'Sort column'
  }

  if (sortState.direction === 'asc') {
    return 'Sorted ascending. Click for descending.'
  }

  if (sortState.direction === 'desc') {
    return 'Sorted descending. Click to clear sort.'
  }

  return 'Sort column'
}

function getAriaSort(sortState, column) {
  if (sortState.column !== column) {
    return 'none'
  }

  return sortState.direction === 'asc' ? 'ascending' : 'descending'
}

export default function DataTable({
  rows,
  maxRows = 10,
  emptyMessage = 'No rows to display.',
  showIndex = false,
  compact = false,
  dataframe = false,
  emptyValue = '—'
}) {
  const [searchQuery, setSearchQuery] = useState('')
  const [sortState, setSortState] = useState({ column: null, direction: null })
  const deferredSearchQuery = useDeferredValue(searchQuery)
  const safeRows = Array.isArray(rows) ? rows : EMPTY_ROWS
  const columns = useMemo(() => {
    const firstRow = safeRows.find((row) => row && typeof row === 'object')
    return firstRow ? Object.keys(firstRow) : []
  }, [safeRows])

  const normalizedSearchQuery = deferredSearchQuery.trim().toLowerCase()
  const filteredRows = useMemo(() => {
    if (!normalizedSearchQuery) {
      return safeRows
    }

    return safeRows.filter((row) =>
      columns.some((column) => formatCell(row[column], emptyValue).toLowerCase().includes(normalizedSearchQuery))
    )
  }, [columns, emptyValue, normalizedSearchQuery, safeRows])

  const sortedRows = useMemo(() => {
    if (!sortState.column || !sortState.direction) {
      return filteredRows
    }

    return filteredRows
      .map((row, originalIndex) => ({ row, originalIndex }))
      .sort((leftItem, rightItem) => {
        const leftValue = leftItem.row[sortState.column]
        const rightValue = rightItem.row[sortState.column]
        const leftEmpty = isEmptyCell(leftValue)
        const rightEmpty = isEmptyCell(rightValue)

        if (leftEmpty && rightEmpty) {
          return leftItem.originalIndex - rightItem.originalIndex
        }

        if (leftEmpty) {
          return 1
        }

        if (rightEmpty) {
          return -1
        }

        const result = compareCellValues(leftValue, rightValue)

        if (result === 0) {
          return leftItem.originalIndex - rightItem.originalIndex
        }

        return sortState.direction === 'asc' ? result : -result
      })
      .map((item) => item.row)
  }, [filteredRows, sortState])

  const visibleRows = sortedRows.slice(0, maxRows)
  const hasActiveFilter = Boolean(searchQuery.trim())
  const isFiltered = normalizedSearchQuery.length > 0

  function handleSort(column) {
    setSortState((currentSort) => {
      if (currentSort.column !== column) {
        return { column, direction: 'asc' }
      }

      if (currentSort.direction === 'asc') {
        return { column, direction: 'desc' }
      }

      return { column: null, direction: null }
    })
  }

  if (!safeRows.length || !columns.length) {
    return <div className="empty-state">{emptyMessage}</div>
  }

  const resultSummary = isFiltered
    ? `Showing ${visibleRows.length.toLocaleString()} of ${filteredRows.length.toLocaleString()} matching rows`
    : `Showing ${visibleRows.length.toLocaleString()} of ${safeRows.length.toLocaleString()} rows`

  return (
    <div className="table-block">
      <div className="table-toolbar">
        <label className="table-search">
          <span>Search table</span>
          <input
            type="search"
            value={searchQuery}
            placeholder="Filter rows..."
            onChange={(event) => setSearchQuery(event.target.value)}
          />
        </label>
        <div className="table-toolbar-meta">
          <span>{resultSummary}</span>
          {hasActiveFilter ? (
            <button type="button" className="table-clear-button" onClick={() => setSearchQuery('')}>
              Clear
            </button>
          ) : null}
        </div>
      </div>

      {visibleRows.length ? (
        <div className="table-shell">
          <table className={`data-table ${compact ? 'is-compact' : ''} ${dataframe ? 'is-dataframe' : ''}`}>
            <thead>
              <tr>
                {showIndex ? <th className="index-column" aria-label="row index" /> : null}
                {columns.map((column) => (
                  <th key={column} aria-sort={getAriaSort(sortState, column)}>
                    <button
                      type="button"
                      className={`table-sort-button ${sortState.column === column ? 'is-active' : ''}`}
                      aria-label={`${getSortLabel(sortState, column)}: ${column}`}
                      onClick={() => handleSort(column)}
                    >
                      <span>{column}</span>
                      <span className="table-sort-indicator" aria-hidden="true">
                        {sortState.column === column ? (sortState.direction === 'asc' ? '↑' : '↓') : '↕'}
                      </span>
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {visibleRows.map((row, index) => (
                <tr key={`${index}-${columns[0]}-${formatCell(row[columns[0]], emptyValue)}`}>
                  {showIndex ? <td className="index-column">{index}</td> : null}
                  {columns.map((column) => (
                    <td key={`${index}-${column}`}>{formatCell(row[column], emptyValue)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="empty-state">
          No rows match {searchQuery.trim() ? <strong>"{searchQuery.trim()}"</strong> : 'the current filter'}.
        </div>
      )}
    </div>
  )
}
