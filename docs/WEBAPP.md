# The Bosco web app (plan, 2026-09-24): SUPERSEDED

> **Superseded 2026-09-25.** The web app is Halteres, a separate repo
> (`~/projects/arclight/halteres`), and it owns sign-in, UI, storage and limits. This
> repo owns only the model and the service Halteres calls. This file is kept as
> history.

The public front end at bosco.systems, on the Kimsufi (Xeon E-2274G,
4 cores, 32 GB, no GPU). It is the only public consumer of the API
(`API.md`). The UI design is Nick's, in Claude Design: two-panel chat plus a
slim brain panel.

## Decided (Nick, 2026-09-24)

- **Login only, through atproto.** No anonymous use. No accounts of our own;
  our own registration only if atproto ever turns out not to be enough.
- **Identity only:** OAuth with the base `atproto` scope, so we learn the DID
  and handle and nothing else. We cannot post, and we cannot read anyone's data.
  (atproto OAuth spec and client guide: docs.bsky.app/docs/advanced-guides/oauth-client.)
- **The API is private to the server.** The web app is its only public
  consumer. Direct use needs a **dev token issued by Nick**: revocable,
  stored hashed, with its own caps.

## Shape

- **Caddy**: HTTPS for bosco.systems.
- **Web app** (TypeScript, the official atproto OAuth client library, as a
  confidential client): login, sessions, per-DID rate limits, one queue with
  a maximum length, input caps, the UI. Publishes its client metadata at
  `https://bosco.systems/oauth/client-metadata.json`.
- **Bosco service** (Python, listening only on the box): runs the frozen
  published version, returns answers and slimmed replay traces.

## What is stored

- **Server:** DID, session, usage counters, minimal request logs. No
  conversation content is kept.
- **Browser:** conversations, in local storage.
- **Nothing any visitor does reaches Bosco's training.**

## Open

- Saved question sets: **never in the user's atproto repo.** Repository
  contents are "entirely public and verifiable" (atproto.com/specs/repository),
  so saving there would publish them. If the feature comes, they live in the
  browser or on our server, keyed by DID. Login itself writes nothing to
  anyone's repo.
- The per-DID caps (questions per minute and per day), set once the server
  speed is measured.
- A DID blocklist, if abuse needs it.
