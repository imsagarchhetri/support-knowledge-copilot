<!-- doc_type: api -->
<!-- last_updated: 2026-03-20 -->
<!-- access_level: public -->

# API Documentation

## Authentication
The API uses bearer tokens. Create a token under Settings > Developer. Pass it as
`Authorization: Bearer <token>`. Tokens inherit the permissions of the creating
user. Rotate tokens every 90 days.

## Pagination
List endpoints return up to 50 items per page. Use the `cursor` query parameter
from the response's `next_cursor` field to fetch the next page. Do not assume
offset-based pagination; always follow the cursor.

## Webhooks
Register webhooks under Settings > Developer > Webhooks. We send a POST with an
HMAC signature in the `X-Signature` header. Verify the signature using your
webhook secret. We retry failed deliveries up to 5 times with backoff.

## Errors
Errors return a JSON body with `code` and `message`. Common codes: API-429 (rate
limit), API-401 (bad token), API-422 (validation error). Always check the `code`
field rather than parsing the message string.
