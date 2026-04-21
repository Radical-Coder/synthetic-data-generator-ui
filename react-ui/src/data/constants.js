export const TAB_ITEMS = [
  { id: 'upload', label: 'Upload & Generate' },
  { id: 'quality', label: 'Quality Dashboard' },
  { id: 'comparison', label: 'Statistical Comparison' },
  { id: 'download', label: 'Download Results' },
  { id: 'benchmarks', label: 'Performance Benchmarks' }
]

export const BENCHMARK_OPTIONS = [
  1000,
  10000,
  50000,
  100000,
  500000,
  1000000
]

export const TYPE_STYLES = {
  continuous_numeric: { emoji: '🔵', label: 'continuous_numeric' },
  discrete_numeric: { emoji: '🟢', label: 'discrete_numeric' },
  categorical: { emoji: '🟡', label: 'categorical' },
  boolean: { emoji: '🟣', label: 'boolean' },
  datetime: { emoji: '🟠', label: 'datetime' },
  email: { emoji: '📧', label: 'email' },
  phone: { emoji: '📱', label: 'phone' },
  unknown: { emoji: '⚪', label: 'unknown' }
}
