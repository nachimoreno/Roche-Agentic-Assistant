# Deployment — Hugging Face Spaces + Neon Postgres

A shareable hosted demo for Roche scientists. **Total cost: €0/month.**

## Architecture

| Piece | Role | Tier |
|---|---|---|
| **HF Space (Docker)** | FastAPI + embeddings + ChromaDB (ephemeral, rebuilt from Drive on boot) | Free CPU Basic — 2 vCPU / 16 GB RAM |
| **Neon** | Postgres: users, sessions, chat history, feedback (via `DATABASE_URL`) | Free — 0.5 GB, scale-to-zero |
| **Groq** | LLM (Llama 3.3 70B) | Free API tier |

The app needs ~1 GB RAM at boot (ONNX embedding model), which is why HF's 16 GB free tier
fits where Render's 512 MB free tier OOMs.

## Already done (code is deploy-ready)

- `psycopg[binary]` driver added; `make_engine` sets `pool_pre_ping` + `pool_recycle`
  (transparent reconnect when Neon scales to zero) and auto-upgrades a bare
  `postgres://` scheme to psycopg3.
- `database_url` reads from `DATABASE_URL` **or** `DB_CONNECTION_STRING`.
- Drive service-account key accepted as **inline JSON** (an env secret), not just a file path.
- `Dockerfile` + `.dockerignore` (UID-1000 user, `libgomp1`, binds `0.0.0.0:7860`).
- **Migration smoke test passed** against Neon: `create_all` produces clean DDL
  (native `UUID` PKs, `TIMESTAMP`), UUIDv7 + datetime round-trip verified.

## Step 4 — create and configure the Space

1. **huggingface.co → New Space → SDK: Docker (blank) → CPU basic (free).**
   Set visibility to **Private** if you don't want the source public (the demo
   link still works either way).

2. **Add Space frontmatter to `README.md`** (HF reads Space config from here):
   ```yaml
   ---
   title: Roche Scientist Assistant
   emoji: 🧪
   colorFrom: blue
   colorTo: indigo
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```

3. **Set Secrets** (Space → Settings → *Variables and secrets*).

   Secrets (encrypted):
   | Key | Value |
   |---|---|
   | `GROQ_API_KEY` | your Groq key |
   | `DB_CONNECTION_STRING` | the Neon URL (`postgresql://...` — driver upgraded automatically) |
   | `GOOGLE_SERVICE_ACCOUNT_JSON` | the **full JSON content** of the service-account key (paste the file's contents, not a path) |
   | `SESSION_SECRET` | a strong random value — `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
   | `ADMIN_EMAILS` | `nachi@student.ie.edu` (comma-separated for more) |

   Variables (non-secret):
   | Key | Value |
   |---|---|
   | `DRIVE_FOLDER_ID` | `1HyEJ9L_YFpzCUFnakvuvzGrVB8rftmjJ` |
   | `SESSION_HTTPS_ONLY` | `true` (HF serves HTTPS → cookie gets the Secure flag) |

   `HOST=0.0.0.0` and `PORT=7860` are already baked into the Dockerfile — no need to set them.

   > **Document uploads need write access.** Read-only ingestion works with the
   > folder shared to the service account as **Viewer**, but the in-app "add to
   > knowledge base" upload needs the folder shared as **Editor** (and, because a
   > service account has no personal Drive storage quota, the folder should live
   > in a **Shared Drive**). When the account is Viewer-only the app still runs
   > normally — the attach control simply greys out with a reason. To enable
   > uploads, share the `DRIVE_FOLDER_ID` folder with the service-account email
   > as Editor.

4. **Push the repo to the Space remote:**
   ```bash
   git remote add space https://huggingface.co/spaces/<user>/<space-name>
   git push space main
   ```
   `.env`, `secrets/`, `.chroma/`, and `data/app.db` are gitignored and `.dockerignore`d —
   they will not be pushed or baked into the image.

5. **First build/boot** (watch the Space logs): ~a few minutes to `pip install`, then the
   first boot downloads the embedding model, ingests from Drive into ephemeral Chroma, and
   one-time-seeds the demo analytics into Neon.

## Verify

- Open `https://<user>-<space-name>.hf.space` → register → chat.
- Log in as an `ADMIN_EMAILS` user → `/admin` shows the analytics dashboard.
- **Restart the Space**, log back in → chat history persists ⇒ Neon wiring confirmed.

## Operational notes

- **Free Space sleeps after ~48 h idle**; first visit wakes it — model reload + Drive
  re-ingest takes tens of seconds (only the relational data is persistent; Chroma is rebuilt).
- **Neon scales to zero after 5 min idle**; `pool_pre_ping` wakes it transparently (~1–2 s
  on the first query). No keep-alive needed (unlike Supabase's 7-day hard pause).
- To eliminate the Chroma cold-rebuild later, HF persistent storage is $5/mo — but then you
  wouldn't need the external DB. Free HF + Neon is the cheaper combination.
