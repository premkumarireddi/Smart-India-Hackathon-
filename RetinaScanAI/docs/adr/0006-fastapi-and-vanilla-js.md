# ADR 0006: FastAPI backend, dependency-free vanilla-JS frontend

## Status
Accepted

## Context
Need a backend framework to serve the model, and a frontend so a human can
actually try the screening flow (upload -> quality gate -> result +
Grad-CAM) without hitting the API with curl.

## Decision
**Backend: FastAPI**, not Flask. FastAPI gives us, for free: automatic
request/response validation via Pydantic (`app/schemas.py`), interactive
OpenAPI docs at `/docs` (useful for a demo — a judge can try the API
without the frontend at all), and native `async def` support for the file
upload path. `python-multipart` handles the multipart form upload.

**Frontend: plain HTML/CSS/vanilla JS**, no React/Vite/build step. This is
a single-page upload-and-view-results UI with no client-side routing, no
complex state management, and no component reuse need — exactly the case
where a build toolchain adds setup risk (npm install failures, version
drift) without adding real value. `frontend/index.html` can be opened
directly or served by any static file server; `app.js` talks to the
backend via `fetch()` against a configurable `API_BASE`.

## Consequences
- Positive: `pip install -r requirements.txt && uvicorn app.main:app` is
  the entire backend setup; the frontend needs no `npm install` at all —
  just open `index.html` or run any static server pointed at `frontend/`.
- Negative: if the UI grows more complex (multi-page, auth, richer client
  state), a framework migration (React/Vue) is the natural next step —
  noted in the README roadmap, not a blocker today.
