# Roche Scientist Assistant — Architecture, Features & Requirements

Derived from [Meeting summary 1.pdf](Meeting%20summary%201.pdf).

---

## 1. Project Overview

### 1.1 Purpose
Build an **AI Agent / chatbot / assistant** for Roche scientists that consolidates access to protocols, documents, onboarding information, and day-to-day operational knowledge in a single, conversational interface.

### 1.2 Problem Statement
- Roche scientists currently have **no single internal application** where they can ask all work-related questions.
- Roche has many highly specialized in-house applications, but scientists often don't know **which app to use** or **where to find** information.
- There is **miscommunication between scientists and IT** — sometimes scientists believe they are using the correct tool but actually are not (real example: a Basel-developed in-house tool that thousands of scientists were supposed to use, but a month later many had not really adopted it, and feedback was being submitted against the wrong app).
- It is not only a technical issue — it is a **communication and user experience** problem.
- Scientists, in general, **do not read emails**, so email-based information distribution is insufficient.

### 1.3 Target Users
- Roche scientists (potentially **several thousand worldwide**).
- New scientists going through onboarding.
- Scientists in Germany and Switzerland (multilingual needs).

### 1.4 Shape of the Solution
- **Not fully defined yet** — could be an AI Agent, a chatbot, or another assistant form.
- Roche is open to whatever solution benefits scientists most.
- Should ideally behave as an **agent**: proactive, able to check documents, suggest next steps, summarize scientific documents, and guide the scientist through a process.

---

## 2. Core Features

### 2.1 Conversational Knowledge Access
The assistant must answer scientist questions instantly. Representative queries include:
- "Where can I request access?"
- "How should I clean my equipment?"
- "What is the correct procedure for cleaning lab devices, testing equipment, or sample-related equipment?"
- "How can I check the stock of samples?"
- "Do we have enough samples available?"

**Requirements:**
- Answers **must be reliable** — these are scientists and laboratory procedures.
- Answers **must be grounded** in the correct documentation or official procedures.
- Where the assistant cannot perform an action directly, it must **guide the scientist** to the correct internal Roche application and explain the steps to follow.

### 2.2 Document Search & Retrieval (Google Drive)
- Primary knowledge source: **Google Drive** (Roche's current document storage).
- The system must **search through documents** and return the correct answer.
- Must handle the known Google Drive problem of **multiple versions of the same document** — ideally always using the **most up-to-date version**.
- Future-proofing: Roche may later migrate to **Microsoft / SharePoint** — design should allow this evolution.

**Document domains in scope:**
- Onboarding materials
- Access requests
- How to reach different internal apps
- Document upload procedures
- Incident creation / issue reporting
- Problems with devices or virtual sessions
- Booking instruments
- Checking sample stocks
- Cleaning lab devices
- Decontamination procedures
- Logistics
- Equipment usage

### 2.3 Web / Internet Search (Optional but Desired)
- Ability to fetch information from **reliable public internet sources** when appropriate.
- Example use case: a scientist asks how to clean an HP device — the assistant could look up the **safe alcohol percentage** for cleaning that device.
- Should only be used when public information is genuinely useful and reliable.

### 2.4 ServiceNow Integration — Incident Creation
- Roche uses **ServiceNow** as their ticketing tool.
- Scope for this project: **create incidents from the chatbot** using information collected from the user.
- **Out of scope:**
  - Internal assignment process
  - Department management / routing
  - Follow-up emails to users
  - Full ServiceNow workflow
- A **ServiceNow demo account + API** can be used for testing.
- Trigger scenarios for incident creation:
  - Problem with a device
  - Problem with a virtual session
  - Access to a tool

**Reference flow (for context, not implementation):** user fills a form → describes incident → submits → system assigns to department → department resolves → emails user. We replace only the "form / submit" step with the chatbot.

### 2.5 Feedback Channel for IT
The chatbot is also a **feedback collection mechanism** for IT teams.

**Requirements:**
- Allow scientists to **report issues, confusion, or suggestions** about digital solutions.
- Allow feedback on **specific points** in a guide or document that the scientist did not understand.
- Allow scientists to **ask for additional information** or flag that a process is confusing.
- The feedback process must be **as easy as possible** — scientists are busy, often standing, and dislike bureaucracy.

### 2.6 Question vs. Feedback Classification
- The system must **differentiate** between:
  - A normal **information-seeking question** → answer it
  - **Feedback** about a tool, process, or document → route to feedback pipeline

### 2.7 Sentiment / Emotion Detection
- When the user provides feedback, the system must detect **sentiment / emotion**, including at minimum:
  - Frustrated
  - Happy
  - Angry
  - Confused
- Emotion signal feeds the macro-level feedback analytics for IT and documentation teams.

### 2.8 Multilingual Support
- **Core languages:** English, German, Italian, French.
  - German and English are confirmed essential.
  - Italian and French to be confirmed by Roche.
- **Language detection:** identify what language the user is writing in.
- **Mirror response language:** always answer in the **same language** the user used.
- **On-the-fly translation:** if the underlying document is in English and the user asks in German, translate the answer to German instantly.
- Must support translating information **into English** when needed.

### 2.9 Proactive Agent Behaviors
The assistant should not only respond — it should also:
- **Check documents** for relevance
- **Suggest next steps** in a procedure
- **Summarize scientific documents**
- **Guide the scientist** through multi-step processes (onboarding, access requests, document uploads, etc.)

### 2.10 Guidance vs. Direct Action
- The assistant is **not expected to perform actions inside all Roche internal applications** at this stage.
- For most operational tasks (checking stock, booking instruments, etc.), the chatbot should **explain which internal app to use** and **how to get there**.
- Exception: **ServiceNow incident creation** is an actual action the chatbot performs.

### 2.11 Session Continuity Across Devices
- Scientists use **multiple devices** during the day.
- The chatbot must be **linked to the user's session**, so the conversation can be **resumed seamlessly** when they switch devices.

### 2.12 Macro-Level Feedback Analytics
For IT and documentation teams, the system must surface insights such as:
- Are scientists understanding the tools and documents?
- What general issues are being reported?
- Which parts of the documentation may need improvement?
- Aggregated sentiment trends.

---

## 3. Non-Functional Requirements

### 3.1 Reliability
- High reliability is critical — answers drive lab procedures.
- Always serve answers from the **most current version** of source documents.

### 3.2 Scalability
- Must support **several thousand scientists worldwide**.

### 3.3 Simplicity & UX
- Tool must be **simple, direct, proactive, and easy to use**.
- Designed for scientists who:
  - Are often **standing** in the lab
  - Are **moving between devices and locations**
  - Are **focused on science**, not bureaucracy
  - **Do not read emails**
- The user experience is explicitly called out as critical.

### 3.4 Trustworthiness
- Answers must be grounded in approved documentation.
- Public web answers must come from reliable sources.

### 3.5 Privacy / Confidentiality
- Roche cannot share internal systems or all real documents.
- Use **non-confidential example documents** Roche may provide, supplemented by **mock documents** built by students.
- Mock documents are acceptable (e.g., a cleaning-products doc can be built from public internet information).

---

## 4. Architecture Overview

### 4.1 High-Level Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         Scientist UI                            │
│   (cross-device, session-aware, mobile/desktop friendly)        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Conversation Layer                         │
│  - Language detection                                           │
│  - Question vs. Feedback classifier                             │
│  - Sentiment / emotion detector                                 │
│  - Session state (cross-device continuity)                      │
└──────┬────────────────────────────┬─────────────────────────────┘
       │                            │
       ▼                            ▼
┌──────────────┐         ┌──────────────────────┐
│  Agent Core  │◀───────▶│   Tool / Action Hub  │
│  (LLM + RAG) │         │                      │
└──────┬───────┘         │  - ServiceNow API    │
       │                 │  - Web search        │
       │                 │  - Feedback writer   │
       │                 └──────────────────────┘
       ▼
┌──────────────────────────────────────────────┐
│           Knowledge Sources                  │
│  - Google Drive (primary)                    │
│  - (Future) SharePoint                       │
│  - Reliable public web (selective)           │
└──────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────┐
│        Feedback & Analytics Pipeline         │
│  - Macro-level dashboards for IT             │
│  - Sentiment trends                          │
│  - Documentation gap detection               │
└──────────────────────────────────────────────┘
```

### 4.2 Component Responsibilities

**Scientist UI**
- Accessible after the scientist unlocks their **virtual login hardware device** (small device used daily, kept next to the computer).
- Must work across multiple devices used in a single day.
- Optimized for lab usage (quick, low-friction).

**Conversation Layer**
- Detects user language → tags session.
- Classifies each turn as **question** or **feedback**.
- Runs **sentiment / emotion analysis** on feedback turns.
- Maintains **session state** keyed to the scientist's identity (so the same conversation continues across devices).

**Agent Core (LLM + Retrieval-Augmented Generation)**
- Orchestrates tool use.
- Performs document retrieval over the Google Drive corpus.
- Generates grounded, translated answers.
- Drives proactive behaviors (suggesting next steps, summarizing, guiding through procedures).

**Tool / Action Hub**
- **ServiceNow connector** — creates incidents via API.
- **Web search connector** — selective public-internet lookups.
- **Feedback writer** — persists structured feedback (text + sentiment + context) for analytics.

**Knowledge Sources**
- **Google Drive** — primary corpus; needs version-aware retrieval to avoid stale duplicates.
- **SharePoint** — future-state migration target; architecture should not be tightly coupled to Drive.
- **Public web** — fallback for general technical facts (e.g., HP device cleaning concentration).

**Feedback & Analytics Pipeline**
- Aggregates feedback events.
- Surfaces macro-level patterns for IT and documentation owners.
- Highlights confusing documents / processes.

### 4.3 Key Architectural Concerns

| Concern | Approach |
|---|---|
| Document version drift in Drive | Version-aware indexing; prefer latest version; deduplicate near-duplicates. |
| Multilingual answers | Detect language → retrieve from English corpus → translate response into user's language. |
| Cross-device session | Server-side session store keyed to the scientist's identity, not the device. |
| Source switch (Drive → SharePoint) | Abstract the knowledge-source interface so connectors are swappable. |
| Action vs. guidance | Default to guidance (point to the right internal app); only act when a sanctioned tool exists (ServiceNow API). |
| Reliability of answers | RAG with citations to source documents; do not invent procedures. |
| Mock vs. real docs | Connector treats mock and real docs identically; corpus contents can swap without code changes. |

---

## 5. Functional Requirements (Checklist)

### 5.1 Knowledge & Q&A
- [ ] Answer scientist questions about onboarding, access, procedures, equipment, samples.
- [ ] Ground every answer in the source documentation.
- [ ] Cite or point to the source document where relevant.
- [ ] When the answer lives in an internal app, **direct the user to that app** and the steps to take.
- [ ] Optionally fetch from reliable public web sources when appropriate.

### 5.2 ServiceNow
- [ ] Collect required incident information conversationally.
- [ ] Create an incident via the ServiceNow API.
- [ ] Confirm creation back to the scientist (incident reference).
- [ ] Do NOT handle assignment, routing, or follow-up email.

### 5.3 Feedback
- [ ] Classify incoming message as **question** or **feedback**.
- [ ] Detect sentiment/emotion (frustrated, happy, angry, confused, …).
- [ ] Persist feedback with context (which document/process it refers to).
- [ ] Allow inline feedback on a specific document section or guide step.
- [ ] Expose macro-level analytics to IT / documentation owners.

### 5.4 Languages
- [ ] Detect user language automatically.
- [ ] Support EN, DE, IT, FR (DE + EN mandatory; IT + FR to be confirmed).
- [ ] Respond in the user's language even when source docs are in English.
- [ ] Translate content into English on demand.

### 5.5 Sessions
- [ ] Persist conversation per scientist identity.
- [ ] Resume conversation seamlessly when the scientist moves to another device.

### 5.6 Document Management
- [ ] Index Google Drive documents.
- [ ] Detect and prefer the **latest version** of any document.
- [ ] Architecture allows future swap to SharePoint.

### 5.7 Agent / Proactivity
- [ ] Suggest next steps in multi-step processes.
- [ ] Summarize scientific documents on request.
- [ ] Guide users through onboarding and procedural flows.

---

## 6. Out of Scope (For This Project)

- Full ServiceNow workflow (assignment, routing, email follow-up).
- Direct integration with **all** Roche in-house applications — the chatbot only **guides** the user to them (except for ServiceNow incident creation).
- Direct actions like booking instruments or checking sample stocks programmatically — these remain in the respective internal apps.
- Use of Roche's internal systems for development (not permitted) — use demo / mock equivalents.
- Handling real confidential Roche data in mock documents.

---

## 7. Data & Document Strategy

### 7.1 Document Sources
- **Real (non-confidential) docs**: Roche may share examples from Google Drive.
- **Mock docs**: students build these to simulate the type and structure of real Roche documents.
  - Example explicitly approved: a cleaning-equipment doc built from public internet info.

### 7.2 Document Topics to Simulate
- Onboarding & access requests
- Navigating to internal apps
- Uploading documents
- Incident creation / reporting issues
- Device & virtual session problems
- Booking instruments
- Checking sample stocks
- Cleaning lab devices
- Decontamination
- Logistics
- Correct equipment usage

### 7.3 Two Main Documentation Themes
1. **Onboarding & bureaucracy** — for new scientists getting set up.
2. **Day-to-day scientific procedures** — operational guidance, some of which leads to a specific internal app.

---

## 8. User Experience Principles

- **Proactive over reactive** — anticipate, suggest, summarize.
- **Low friction** — minimal clicks/taps; designed for someone standing in a lab.
- **Direct** — short, actionable answers; avoid wall-of-text bureaucracy.
- **Multi-device aware** — pick up where they left off.
- **Email-free** — assume scientists don't read email; the chatbot is the channel.
- **Trustworthy** — visibly grounded in real procedures.

---

## 9. Success Criteria (Roche-Defined)

A successful project will:
- Save scientists time.
- Improve efficiency in day-to-day work.
- Improve onboarding.
- Reduce miscommunication between scientists and IT.
- Improve user experience.
- Enable more effective issue reporting.
- Give IT and documentation teams **useful macro-level insights** to improve processes at scale.
- Be **handy and appealing** for scientists' day-to-day work, fitting their profile, complex routines, and laboratory environment.

---

## 10. Open Questions / To Confirm With Roche

- Final confirmed list of supported languages (IT and FR pending).
- Which non-confidential example documents Roche will actually share.
- Exact ServiceNow demo account / API access details.
- Whether SharePoint migration timing affects prototype scope.
- Specific scope of "macro-level feedback" dashboards (audience, KPIs).
- Exact form factor of the assistant (chatbot UI vs. agent embedded elsewhere).
