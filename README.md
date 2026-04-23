# Auto-Detecting Synthetic Data Generator
Developed by Robel | © 2026 | Version 1.0

---

## Files
- `synthetic_data_app.py` — main application
- `AUTO_DETECTING_SYNTHETIC_DATA_GENERATOR_ENHANCED.py` — core engine
- `requirements.txt` — dependencies
- `react-ui/` — React/Vite frontend draft and deployment handoff artifacts

---

## How to Run Streamlit

**1. Install dependencies:**
```
pip install -r requirements.txt
```

**2. Run the app:**
```
streamlit run synthetic_data_app.py
```

**3. Open your browser:**
```
http://localhost:8501
```

---

## Requirements
- Python 3.11 or higher

---

## How to Run React UI

**1. Install frontend dependencies:**
```
cd react-ui
npm ci
```

**2. Run locally:**
```
npm run dev
```

**3. Open your browser:**
```
http://localhost:8502
```

**4. Build for handoff/deployment:**
```
npm run build
```

---

## React UI Container Handoff

Build the frontend container from `react-ui/`:
```
docker build -t synthetic-data-generator-react-ui ./react-ui
```

Run it locally on port `8502`:
```
docker run --rm -p 8502:80 synthetic-data-generator-react-ui
```

The container serves the static Vite build with nginx. Backend/Python integration is not wired into this frontend draft yet; the current React UI uses client-side demo generation logic.
