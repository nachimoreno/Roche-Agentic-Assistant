# Roche Sans (brand typeface)

The UI is wired to use **Roche Sans** with a graceful fallback to Inter
(see the `@font-face` block in `src/static/index.html`). The font files
themselves are **not committed** — Roche Sans is licensed and must come
from the Roche brand portal.

## How to enable it

Drop the licensed files into this folder using **exactly these names**
(the `@font-face` rules reference them). `.woff2` is preferred for the
web; `.ttf` is an accepted fallback — provide whichever you have.

| Weight | Style   | Filename (woff2 preferred)   |
|-------:|---------|------------------------------|
| 300    | Light   | `RocheSans-Light.woff2`      |
| 400    | Regular | `RocheSans-Regular.woff2`    |
| 500    | Medium  | `RocheSans-Medium.woff2`     |
| 600    | SemiBold| `RocheSans-SemiBold.woff2`   |
| 700    | Bold    | `RocheSans-Bold.woff2`       |

If you only have `.ttf`/`.otf`, either:
- rename them to `RocheSans-<Weight>.ttf` (the rules already list a
  `.ttf` fallback), **or**
- convert to `.woff2` (e.g. https://cloudconvert.com/ttf-to-woff2) for
  smaller, faster downloads.

You don't need all five weights — any you omit simply fall back to the
nearest available weight (and ultimately to Inter). Regular (400),
Medium (500) and Bold (700) cover most of the UI.

## Verifying

After adding the files and restarting the server, open the app and
check DevTools → Network → filter "Font": you should see
`RocheSans-*.woff2` loading with status 200. The body text and the
"Roche" wordmark will switch from Inter to Roche Sans automatically.
