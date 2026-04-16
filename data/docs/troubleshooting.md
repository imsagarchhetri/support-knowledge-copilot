<!-- doc_type: troubleshooting -->
<!-- last_updated: 2026-03-15 -->
<!-- access_level: internal -->

# Troubleshooting

## App won't load (error APP-500)
Error APP-500 indicates a server-side issue. First check the status page. If all
systems are green, clear your browser cache and cookies, then hard-refresh. If
the error persists, try an incognito window to rule out extensions.

## Sync is stuck
If data is not syncing, confirm you are online and signed in. Force a sync from
Settings > Sync > Sync now. Large workspaces can take several minutes on first
sync. A stuck sync usually clears after sign-out and sign-in.

## API rate limits (error API-429)
The API allows 600 requests per minute per token. Error API-429 means you have
exceeded the limit; back off and retry with exponential backoff. Enterprise
plans can request higher limits through their support contact.

## Slow performance
Slow performance is often caused by very large boards. Archive old items, reduce
the number of open tabs, and disable unused integrations. Performance issues
that persist across devices should be reported with a HAR file.
