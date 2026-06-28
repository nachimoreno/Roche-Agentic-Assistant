/* Roche Fritz — service worker.
 *
 * Goal: make the app installable and resilient on tablet Wi-Fi, without ever
 * serving stale dynamic data. Strategy:
 *   - App shell (HTML, icons, manifest) is precached on install.
 *   - Navigations are NETWORK-FIRST: always try the server so a new deploy is
 *     picked up immediately; fall back to the cached shell only when offline.
 *   - Same-origin static assets are CACHE-FIRST for instant loads.
 *   - /api/* and all non-GET requests are never cached — auth + chat must
 *     always hit the live server.
 * Bump CACHE to invalidate everything on the next activate.
 */
const CACHE = 'roche-shell-v1';
const SHELL = [
  '/',
  '/static/index.html',
  '/static/manifest.webmanifest',
  '/static/icon.svg',
  '/static/icon-192.png',
  '/static/icon-512.png',
  '/static/icon-512-maskable.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE)
      .then((cache) => cache.addAll(SHELL))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;                 // never cache POST/PUT/etc.

  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;  // let cross-origin (font CDN) pass through
  if (url.pathname.startsWith('/api/')) return;     // dynamic + auth — always live
  if (url.pathname === '/sw.js') return;

  // Navigations: network-first, fall back to the cached shell when offline.
  if (req.mode === 'navigate') {
    event.respondWith(
      fetch(req)
        .then((res) => {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put('/', copy));
          return res;
        })
        .catch(() => caches.match('/').then((r) => r || caches.match('/static/index.html')))
    );
    return;
  }

  // Static assets: cache-first, then fill the cache on a miss.
  event.respondWith(
    caches.match(req).then((hit) => {
      if (hit) return hit;
      return fetch(req).then((res) => {
        if (res && res.ok) {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(req, copy));
        }
        return res;
      });
    })
  );
});
