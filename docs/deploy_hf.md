# Deploying CropIQ to Hugging Face Spaces

This Space runs **Streamlit only**. The FastAPI service is local-only — HF Spaces
runs a single process per Space, so the dashboard calls `PredictionEngine` in-process
(no HTTP hop) via `app.py` at the repo root.

---

## 1. Pre-flight checks (local)

Run `make all` from a clean checkout. Then verify the artifacts that HF needs are
tracked in git, not gitignored:

```bash
git check-ignore -v models/best_model.pkl              # should NOT be ignored
git check-ignore -v data/processed/features.parquet    # should NOT be ignored
git ls-files models/best_model.pkl                     # should print the path
git ls-files data/processed/features.parquet           # should print the path
git ls-files app.py                                    # root app.py present
```

The `.gitignore` allows these two artifacts explicitly (CLAUDE.md gotcha #31). If
either is missing, `make all` did not finish; do not push.

Smoke-test the entry point exactly the way HF will:

```bash
python -c "import app"                                 # must import cleanly
.venv/bin/streamlit run app.py                         # opens on http://localhost:8501
```

Click through IA → 2024 → Story County and confirm the SHAP drivers render.

## 2. Create the Space

1. Go to <https://huggingface.co/new-space>.
2. **Owner:** your HF username.
3. **Name:** `cropiq`.
4. **License:** MIT.
5. **SDK:** Streamlit.
6. **Hardware:** CPU basic (free).
7. **Visibility:** Public.

## 3. Add the Space as a git remote

HF will show the exact URL after creation. Then:

```bash
git remote add space https://huggingface.co/spaces/<username>/cropiq
```

## 4. Push

```bash
git push space main
```

The build runs in 3–5 minutes. Watch the log at
`https://huggingface.co/spaces/<username>/cropiq/logs`.

## 5. Smoke test the live URL

1. Open `https://huggingface.co/spaces/<username>/cropiq`.
2. Sidebar: pick IA → 2024 → Story County.
3. Expect: choropleth map of Iowa, county detail card with prediction + 80% CI,
   three SHAP drivers as a horizontal bar chart.

## Troubleshooting

| Symptom                                | Probable cause / fix |
|----------------------------------------|----------------------|
| Build fails on `pip install`           | Version pinning conflict; do not change `requirements.txt` (CLAUDE.md §3 — tested compatible). |
| `app.py` import error in build log     | Test `python -c "import app"` locally first. |
| Streamlit cannot find `models/best_model.pkl` at runtime | Run `git ls-files models/best_model.pkl` locally — if empty, the `.gitignore` exception is broken. |
| Streamlit choropleth is blank          | The plotly geojson source URL is unreachable from inside the Space. The dashboard caches a filtered subset to `data/processed/counties_geo.json` — confirm that file is tracked. |
| First load is slow                     | Normal — the engine + SHAP TreeExplainer are loaded once via `@st.cache_resource`. Subsequent requests are sub-second. |
| Streamlit version drift                | Set `sdk_version: 1.40.1` in the README YAML frontmatter so HF pins the runtime to match `requirements.txt`. |

## CropIQ-specific notes

- No secrets are needed at runtime. The NASS / NOAA fetches happen offline during
  `make data` and are baked into the committed `features.parquet`.
- HF Spaces does not honor `.env`. If a future version of CropIQ needs runtime
  secrets, configure them in **Space Settings → Variables and secrets**.
- Pushing to `space main` redeploys. Local `.venv`, `mlruns/`, `data/raw/`, and the
  non-deployed model variants stay gitignored.
