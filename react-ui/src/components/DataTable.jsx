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

export default function DataTable({
  rows,
  maxRows = 10,
  emptyMessage = 'No rows to display.',
  showIndex = false,
  compact = false,
  dataframe = false,
  emptyValue = '—'
}) {
  const visibleRows = rows.slice(0, maxRows)
  const columns = visibleRows[0] ? Object.keys(visibleRows[0]) : []

  if (!visibleRows.length || !columns.length) {
    return <div className="empty-state">{emptyMessage}</div>
  }

  return (
    <div className="table-shell">
      <table className={`data-table ${compact ? 'is-compact' : ''} ${dataframe ? 'is-dataframe' : ''}`}>
        <thead>
          <tr>
            {showIndex ? <th className="index-column" aria-label="row index" /> : null}
            {columns.map((column) => (
              <th key={column}>{column}</th>
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
  )
}
