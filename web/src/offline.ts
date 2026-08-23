let registration: Promise<ServiceWorkerRegistration | undefined> | undefined;

export function enableOfflineCache(): Promise<ServiceWorkerRegistration | undefined> {
  if (!("serviceWorker" in navigator)) return Promise.resolve(undefined);
  const appBase = new URL(import.meta.env.BASE_URL, window.location.origin);
  registration ??= navigator.serviceWorker
    .register(new URL("sw.js", appBase), { scope: appBase.pathname })
    .then(() => navigator.serviceWorker.ready)
    .catch((error) => {
      console.warn("Offline static-asset cache is unavailable.", error);
      return undefined;
    });
  return registration;
}
