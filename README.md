# Roche Scientist Assistant

A conversational AI assistant for Roche scientists — one place to ask about protocols, onboarding, equipment, samples, and to report issues, instead of hunting across dozens of internal apps.

## Why

Roche scientists have many specialized in-house tools but no single entry point. They often don't know which app to use, don't read email, and work standing up while moving between devices in a lab. The result is wasted time, miscommunication with IT, and slow onboarding.

## What it does

- **Answers questions** grounded in Roche documentation (onboarding, access requests, cleaning procedures, sample stock, instrument booking, equipment use).
- **Points scientists to the right internal app** when it can't act directly.
- **Creates ServiceNow incidents** from the conversation (device issues, virtual session problems, tool access).
- **Collects feedback** for IT, distinguishing questions from feedback and detecting sentiment (frustrated, confused, happy, angry).
- **Speaks the user's language** — English, German, Italian, French — translating from English source docs on the fly.
- **Follows the scientist across devices**, resuming the conversation wherever they log in.

## Stack (planned)

- **LLM agent + RAG** over a document corpus
- **Google Drive** as the primary knowledge source (SharePoint-ready for the future)
- **ServiceNow API** for incident creation (demo account)
- **Public web search** as a selective fallback for general technical facts
- Session store keyed to scientist identity for cross-device continuity

## Scope

**In scope:** Q&A, document retrieval, ServiceNow incident creation, feedback + sentiment, multilingual responses, cross-device sessions, macro-level feedback analytics.

**Out of scope:** Full ServiceNow workflow (routing, assignment, email follow-up), direct integration with every Roche internal app, real confidential Roche data.

## Documents

- [Architecture, Features &amp; Requirements](Architecture_Features_Requirements.md) — full spec derived from the kickoff meeting

## Success looks like

Scientists save time, onboard faster, hit fewer dead ends, and IT gets real signal on which docs and tools are failing them.
