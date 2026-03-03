const CACHE_NAME = 'vaultcam-v1';

self.addEventListener('install', (event) => {
    self.skipWaiting();
});

self.addEventListener('activate', (event) => {
    event.waitUntil(clients.claim());
});

self.addEventListener('fetch', (event) => {
    if (event.request.mode === 'navigate') {
        event.respondWith(
            fetch(event.request).catch(() => {
                return new Response(
                    '<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Offline — VaultCam</title><style>body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;display:flex;align-items:center;justify-content:center;min-height:100vh;background:#F2F2F7;color:#1C1C1E;text-align:center;padding:20px}h1{font-size:24px;margin-bottom:8px}p{color:#8E8E93}</style></head><body><div><h1>You\'re Offline</h1><p>VaultCam needs an internet connection. Please check your connection and try again.</p></div></body></html>',
                    { headers: { 'Content-Type': 'text/html' } }
                );
            })
        );
    }
});
