const CACHE_NAME = "egu-session-matcher-v4-interdisciplinary";
const IS_LOCAL_DEVELOPMENT = ["localhost", "127.0.0.1", "::1"].includes(
  self.location.hostname,
);

self.addEventListener("install", (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    Promise.all([
      self.clients.claim(),
      caches.keys().then((keys) =>
        Promise.all(keys.filter((key) => key.startsWith("egu-session-matcher-") && key !== CACHE_NAME)
          .map((key) => caches.delete(key))),
      ),
    ]),
  );
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  const url = new URL(request.url);
  if (request.method !== "GET" || url.origin !== self.location.origin) return;
  // Vite's client, source modules, and Web Workers must never come from the
  // offline cache during local development. Cache-first delivery here can mix
  // two interface versions and also preserve an obsolete HMR token.
  if (IS_LOCAL_DEVELOPMENT) {
    event.respondWith(fetch(request));
    return;
  }
  if (url.pathname.includes("/catalogue/")) {
    event.respondWith(
      caches.open(CACHE_NAME).then(async (cache) => {
        try {
          const response = await fetch(request);
          if (response.ok) await cache.put(request, response.clone());
          return response;
        } catch (error) {
          const cached = await cache.match(request);
          if (cached) return cached;
          throw error;
        }
      }),
    );
    return;
  }
  event.respondWith(
    caches.open(CACHE_NAME).then(async (cache) => {
      const cached = await cache.match(request);
      if (cached) return cached;
      const response = await fetch(request);
      if (response.ok) await cache.put(request, response.clone());
      return response;
    }),
  );
});
