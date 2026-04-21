export default function Sidebar({ hasData, generationComplete, originalRows, syntheticRows }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-close" aria-hidden="true">
        ×
      </div>

      <section className="sidebar-section">
        <h2>About</h2>
        <p className="sidebar-intro">This tool uses advanced machine learning to:</p>
        <ul>
          <li>Automatically detect data types</li>
          <li>Preserve statistical distributions</li>
          <li>Generate realistic synthetic data</li>
          <li>Validate quality automatically</li>
        </ul>
      </section>

      <section className="sidebar-section">
        <h2>Supported Data Types</h2>
        <ul>
          <li>
            <strong>Numeric:</strong> Continuous &amp; Discrete
          </li>
          <li>
            <strong>Categorical:</strong> Text categories
          </li>
          <li>
            <strong>Boolean:</strong> True/False values
          </li>
          <li>
            <strong>DateTime:</strong> Dates &amp; timestamps
          </li>
          <li>
            <strong>Email:</strong> Fake email generation
          </li>
          <li>
            <strong>Phone:</strong> Fake phone number generation
          </li>
        </ul>
      </section>

      <section className="sidebar-section">
        <h2>Current Status</h2>
        <div className={`status-tile ${hasData ? 'is-success' : 'is-info'}`}>
          {hasData ? `Data Loaded: ${originalRows.length.toLocaleString()} rows` : 'No data loaded'}
        </div>
        <div className={`status-tile ${generationComplete ? 'is-success' : 'is-info'}`}>
          {generationComplete
            ? `Synthetic Data Generated: ${syntheticRows.length.toLocaleString()} rows`
            : 'Not generated yet'}
        </div>
      </section>
    </aside>
  )
}
