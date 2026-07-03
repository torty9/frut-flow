# How Wispr Flow Works

## 1. What It Is / TL;DR

Wispr Flow is a cloud-based AI voice dictation tool that turns spoken language into polished, formatted text and inserts it directly into whatever application you're focused on. The core daily interaction is a system-wide push-to-talk hotkey: you hold a modifier key in any text field, speak, release, and a few seconds later cleaned-up text (filler words removed, punctuation and capitalization added) appears at your cursor.

Key facts confirmed in this research:

- **It is cloud-only.** There is no offline or on-device mode at any pricing tier — transcription "always occurs on the cloud" per Wispr's own documentation ([Wispr Flow Data Controls](https://wisprflow.ai/data-controls)).
- **It runs on four platforms** — macOS, Windows, iOS (iPhone), and Android — which is one of its clearest differentiators ([Supported devices](https://docs.wisprflow.ai/articles/1036674442-supported-devices-and-system-requirements)).
- **It is subscription-priced** around a free tier with weekly word caps and a Pro tier at $15/mo (or $12/mo annual) as of June 2026 ([Pricing](https://wisprflow.ai/pricing)).
- **It is built by Wispr**, a 2021 startup founded by two Stanford AI Lab researchers, and is comparatively well-funded for its category ([Wikipedia](https://en.wikipedia.org/wiki/Wispr)).

A recurring theme: **Wispr is candid about *where* processing happens (the cloud) but vague about the *technical internals* of its model pipeline** — it describes a pipeline rather than naming the specific speech-recognition model.

---

## 2. How It Works Under the Hood

### 2.1 The processing pipeline: cloud-only

The single most important architectural fact — confirmed at the highest confidence level across primary and independent sources — is that **all transcription and AI formatting happen on remote cloud servers; nothing is processed on-device, and there is no offline mode.**

Wispr's own pages state this unambiguously. The Data Controls page reads: *"Transcription always occurs on the cloud. This is the best way for us to provide accurate, low latency transcription."* The privacy page echoes it. The Security and Compliance FAQ goes further, stating that Wispr Flow *"runs entirely in the cloud and is delivered as multi-tenant SaaS, hosted with a major US cloud provider,"* and that *"the backend must decrypt audio to perform transcription"* — explicitly confirming server-side processing ([Data Controls](https://wisprflow.ai/data-controls); [Privacy](https://wisprflow.ai/privacy); [Security and compliance FAQ](https://docs.wisprflow.ai/articles/3467817258-security-and-compliance-faq)).

This is corroborated by the product's failure behavior: when offline, the desktop app shows "No internet connection," mobile shows a greyed-out Flow Bubble, and the Android system requirements page states verbatim *"Internet connection: Offline transcription is not available."* Competitor-comparison articles consistently market rival apps *on the basis* that they offer on-device processing Wispr lacks — reinforcing the claim. **Verdict: confirmed.**

### 2.2 Latency and the "paragraph-by-paragraph" model

Dictation is **not real-time, word-by-word**. The model is paragraph-by-paragraph, with reported end-to-end latency of roughly **1–2 seconds** for most users. Reviewers note a cloud round-trip reported to run on **AWS us-east-1 via Baseten**, with a published **sub-700ms p99 latency for the AI post-processing layer**.

Baseten's own customer case study (a primary subprocessor source) states Wispr's *"entire pipeline, from speech recognition models to Llama-based transcript enhancement, runs end-to-end in under 700 milliseconds"* on Baseten, and that *"Baseten runs Flow's Llama inference on AWS workload planes"* ([Baseten case study](https://www.baseten.co/resources/customers/wispr-flow/)). Note the ~1–2s figure (end-to-end UX) and the sub-700ms figure (the AI post-processing layer specifically) describe different things.

### 2.3 The model stack — where Wispr is vague

This is the area where the company is **least transparent**. Wispr **describes a pipeline rather than naming the specific ASR model**: an ASR/speech-recognition stage followed by a fine-tuned **Llama-based** LLM cleanup stage. Audio is reportedly routed through third-party subprocessors — names cited across analyses include **Baseten and Soniox** (speech recognition) and **OpenAI, Anthropic, Cerebras, Fireworks AI, and OpenRouter** (text processing), with **AWS** for storage. The Llama-cleanup stage is confirmed by Baseten's primary case study.

**Sourcing caveat:** the full subprocessor roster comes from the Baseten primary source plus secondary reporting. Wispr's own subprocessors help-center article returned an HTTP 404 during verification, so the per-vendor role breakdown does not rest on a currently-retrievable official page. **The exact ASR-model vs. LLM-cleanup split is not clearly disclosed.**

### 2.4 Context-awareness and the role of the Accessibility permission

Wispr's "type into any app" capability and its context-awareness rely on the OS accessibility services. On macOS, the **Accessibility permission** is, in Wispr's own words, what lets Flow *"insert spoken words into other apps"* ([Setup Guide](https://docs.wisprflow.ai/articles/3152211871-setup-guide)). Independent reviews note accessibility is also what lets the app read context such as which app/recipient you're in. The company does not document the precise insertion mechanism (Accessibility-API value-set vs. simulated keystrokes/paste).

---

## 3. The User Workflow

### 3.1 Installation and onboarding

Standard flow: **drag-to-Applications on Mac** or an **installer on Windows** (per-user `.exe`, `.msi` for MDM, or the Microsoft Store). The app **lives in the menu bar (Mac) or system tray (Windows)** and launches at login; there's **no traditional always-open main window** in normal use (though a "Flow Hub" dashboard can be opened). **Sign-in is browser-based** (Google, Apple, Microsoft, SSO, or email/password). Onboarding includes mic testing, shortcut config, language selection, privacy preferences, and a practice dictation.

### 3.2 Permissions

- **Mac:** **Microphone** + **Accessibility** (the latter enables text insertion). On macOS Sequoia (15)+, may also prompt for Input/Keyboard Monitoring for the global hotkey.
- **Windows:** only system microphone access; no app-level Accessibility permission.
- **Android:** the **Accessibility Service** + a **"Display over other apps" (overlay)** permission. Without Accessibility, core dictation does not work.

The "type into any app" capability is a recognized **privacy/trust consideration** — Android even requires a legal-consent screen before granting the service.

### 3.3 Activation: push-to-talk and hands-free

- **Push-to-talk (default):** hold the shortcut, speak while holding, release to transcribe. Default **Fn on Mac** (or **Ctrl+Opt** if no Fn key detected) and **Ctrl+Win on Windows**. **Esc** cancels (text retrievable from History).
- **Hands-free (toggle):** double-tap the shortcut, or a dedicated shortcut (**Fn+Space on Mac**, **Ctrl+Win+Space on Windows**). Sessions cap at **20 min desktop** / **5 min Android**.

Shortcuts are customizable (up to 4 bindings per action, max 3 keys, ≥1 modifier). Mouse buttons (Middle Click, Mouse 4–10) can trigger; left/right click and Caps Lock are excluded.

### 3.4 On-screen UI and text insertion

A floating **"Flow Bar" (desktop) / "Flow Bubble" (Android)** shows idle/recording/processing/error states. Finished text is inserted via the Accessibility service. **If direct insertion fails, Flow retries up to 5 times, then falls back to copying to the clipboard** ("Failed to paste text. It is still on your clipboard.") — a frequent real-world friction point that forces a manual paste and overwrites clipboard contents.

### 3.5 Beyond plain dictation

- **Command Mode (experimental, paid):** highlight text, hold the Command Mode shortcut, speak a command ("make this more concise," "translate to Polish"), release. Defaults: Mac **Fn+Ctrl**, Windows **Ctrl+Win+Alt**. Must be enabled in Settings → Experimental; caps selections under 1,000 words.
- **Whisper recognition:** Flow recognizes quietly-spoken/whispered speech with no toggle. (Note: there's no single feature literally branded "whisper to edit" — that conflates *whisper recognition* with *voice editing/Command Mode*.)
- **Snippets (text expansion):** trigger phrase (≤60 chars) → expansion text (≤4,000 chars). Static only — no dynamic variables.
- **Personal Dictionary:** custom vocabulary (1–60 chars) with replacement rules and "auto-add" learning; syncs across Mac/Windows/iOS/Android.
- **History & Usage:** transcript history with a "Retry transcript" option, plus an Insights/Usage tab (dictation speed vs. typists, corrections, total words, top apps, streak).

---

## 4. Privacy & Data Handling

Privacy controls govern **data retention, not processing location** — transcription happens in the cloud regardless of settings.

- **Default (Privacy Mode off):** Wispr states dictation data *may be used to evaluate, train, and improve* Wispr features and AI models.
- **Privacy Mode / disabling Cloud Sync:** zero data retention — *"Audio and transcripts are processed in real time and discarded after the request completes"* and *"No dictation data is stored or used for training."* Processing still happens remotely; only retention changes ([Privacy Mode & Data Retention](https://docs.wisprflow.ai/articles/6274675613-privacy-mode-data-retention)).
- **Transport/storage:** real-time HTTPS on port 443; backend decrypts audio to transcribe; storage reported on AWS (S3 us-east-1).
- **Trust concern:** the core value prop requires the OS Accessibility service; the dossier notes Wispr faced public backlash over context-awareness/screenshot-capture tied to these permissions.
- **Compliance:** the free tier is described as "HIPAA-ready," but formal compliance (SOC 2 Type II, ISO 27001, enforced HIPAA with BAAs, SSO/SAML) is associated with the **Enterprise** tier. Whether a new A-LIGN SOC 2 Type II report was issued is unconfirmed.

---

## 5. Platforms & Pricing

### Supported platforms

Officially **macOS, Windows, iOS (iPhone only), and Android**. **iPad, Linux, Chromebooks, and VMs/remote desktop are explicitly NOT supported** (a community Linux port exists unofficially).

| Platform | Minimum requirement |
|---|---|
| macOS | 11 (Big Sur)+; Apple Silicon **or** Intel |
| Windows | 10/11; **x64 only** (ARM not supported) |
| iOS | 18.3+ (iPhone only) |
| Android | 13 to 16 (API 36) |

Rollout: Mac/Windows first, **iOS June 2025**, **native Android Feb 23, 2026** ([TechCrunch](https://techcrunch.com/2026/02/23/wispr-flow-launches-an-android-app-for-ai-powered-dictation/)). Note: `wisprflow.ai/downloads` appeared **cached/stale** (still listing Android as "waitlist," Web as "coming soon") — contradicted by the company's own current Android docs.

### Pricing (USD, as of June 2026)

| Tier | Price | Notes |
|---|---|---|
| **Basic (Free)** | $0 | ~2,000 words/wk desktop, 1,000/wk iPhone (reset Sun 12am PT). Android listed "unlimited (limited time)." Includes dictionary, 100+ languages, Privacy Mode, "HIPAA-ready." |
| **Pro** | $15/mo, or **$12/mo annual (~$144/yr)** | Unlimited dictation, Command Mode, priority support, early access. |
| **Teams** | ~$12/user/mo (~$10 annual), 3-seat min *(secondary sources only)* | Admin controls, shared dictionary/snippets. Official doc confirms the tier but lists no price. |
| **Enterprise** | Custom | SOC 2 Type II, ISO 27001, enforced HIPAA, SSO/SAML. |

Every new account gets a **14-day Pro trial, no card required**; **students get ~50% off Pro** plus a 3-month trial. *(Treat exact Teams figures as unverified; prices may change.)*

---

## 6. The Company Behind It

Wispr was **founded in 2021 by Tanay Kothari (CEO) and Sahaj Garg (CTO)** — Stanford roommates and Stanford AI Lab (SAIL) researchers, confirmed across primary sources. The company began with a "thought-powered neural interface" concept (a 2021 PRNewswire release announced a $4.6M seed from NEA and 8VC for exactly that) and later pivoted to the Flow dictation app.

**Funding (reported):** a **$30M Series A led by Menlo Ventures (June 2025)** — with NEA, 8VC, and angels including Pinterest co-founder Evan Sharp and Carta CEO Henry Ward — plus a reported $25M extension, and reportedly in talks for a ~$260M round near a ~$2B valuation in 2026. Reported traction: ~2.5M downloads, 50% MoM user growth. *(The ~$260M/~$2B round appears in-progress, not confirmed closed; total-funding figures could not be cleanly reconciled — some cite ~$81M.)*

---

## 7. Reception, Accuracy & Limitations

**The ratings split:** the **iOS App Store shows ~4.8/5 across ~10,000–11,000 ratings**, while **Trustpilot shows ~2.7/5** (~47 reviews). Different sample sizes/populations matter — App Store reflects a broad base; Trustpilot skews toward self-selected complainants (billing/trial/support).

**Reported friction points:**
- **Latency:** the 1–2s cloud round-trip is noticeable for those wanting immediate input.
- **AI "over-editing":** a consistent complaint that the cleanup layer **rewrites what you said rather than transcribing verbatim**. Notably, Wispr's own accuracy-troubleshooting doc frames accuracy issues as *audio-input* problems and **does not address the AI-rewriting complaint**.
- **Windows resource use:** reports of heavy usage (~800 MB RAM, ~8% CPU idle) and freezing of target apps (VS Code, Notepad++) during dictation.
- **Cloud dependency:** no offline mode is a recurring criticism.
- **"Reliability drops after the 14-day trial":** a recurring Trustpilot/Reddit theme — **unverified** (pattern reported, no mechanism; company hasn't confirmed).

**Accuracy claims — handle with care:** figures like ~97.2% English accuracy and sub-700ms p99 latency trace to **vendor/blog claims, not peer-reviewed benchmarks**. No independent head-to-head WER benchmark could be verified.

---

## 8. How It Compares to Alternatives

Claimed differentiators: **cross-platform breadth, out-of-box AI cleanup/tone-matching, enterprise compliance, developer/IDE integrations (Cursor, Windsurf)**.

**Cross-platform breadth (partially-confirmed):** Wispr's four-platform support — and being the **only major direct rival with an Android app** — is real. But the framing that "most direct rivals are single-OS" is **outdated as of 2026**: **Superwhisper** now supports Mac/Windows/iOS (Windows launched Dec 2025), **Willow Voice** Mac/Windows/iPhone, **Aqua Voice** Mac/Windows/iOS. Only **MacWhisper** remains single-OS. The accurate differentiator is narrower: **Wispr is the broadest (uniquely all four, and the only one on Android)**.

**Where competitors win:**
- **Superwhisper** — privacy (**on-device, no retention**) and **one-time pricing** ($249.99 lifetime). Trade-off: less out-of-box polish.
- **Aqua Voice** — cheaper (**$8/mo**), **real-time word-by-word** text, claims stronger coding-vocabulary accuracy — but **also cloud-only**.
- **MacWhisper** — offline *file* transcription (podcasts/meetings) with local Whisper, ~€59 one-time; barely overlaps with live dictation.
- **Apple Dictation / Windows Voice Typing** — the **free baseline**; Apple's runs on-device, beating Wispr on cost/privacy but lacking AI cleanup.
- **Dragon NaturallySpeaking** — leads on trained medical/legal accuracy but legacy (Windows-only, ~$699, lengthy training).
- **Talon Voice** — different niche (voice-driven computer control); often used *alongside* Wispr.
- **Otter.ai** — adjacent (meeting transcription), not a direct rival.

**Sourcing caveat:** most comparison content is SEO/affiliate marketing; accuracy/latency numbers are vendor/affiliate claims, not independent benchmarks.

---

## 9. Open Questions / What's Not Publicly Documented

1. The exact **ASR model architecture** (Wispr names a pipeline, not the speech-recognition model; its subprocessors page 404'd).
2. Whether **"reliability drops after the trial"** reflects a real technical change or perception.
3. Officially-confirmed **Teams pricing / seat minimum**.
4. Whether the **Android free-tier "unlimited (limited time)"** promo is still active and what it reverts to.
5. Status/date of the **web/browser version**.
6. Exact **student discount** specifics (3 months vs. up to 90 days).
7. Whether the new **A-LIGN SOC 2 Type II report** has been issued.
8. Precise current **valuation/total funding** (~$260M/~$2B round appears in-progress).
9. No independent **peer-reviewed accuracy/WER benchmark** could be verified.
10. Whether competitors have closed Wispr's **enterprise-compliance gap** (self-reported).

---

## Key Sources

- [Wispr Flow — Data Controls](https://wisprflow.ai/data-controls) · [Privacy](https://wisprflow.ai/privacy) · [Pricing](https://wisprflow.ai/pricing) · [Features](https://wisprflow.ai/features)
- [Help Center — Setup Guide](https://docs.wisprflow.ai/articles/3152211871-setup-guide) · [Security & Compliance FAQ](https://docs.wisprflow.ai/articles/3467817258-security-and-compliance-faq) · [Supported devices](https://docs.wisprflow.ai/articles/1036674442-supported-devices-and-system-requirements) · [Privacy Mode & Data Retention](https://docs.wisprflow.ai/articles/6274675613-privacy-mode-data-retention) · [Command Mode](https://docs.wisprflow.ai/articles/4816967992-how-to-use-command-mode) · [Hands-free](https://docs.wisprflow.ai/articles/6391241694-use-flow-hands-free) · [Keyboard shortcuts](https://docs.wisprflow.ai/articles/2612050838-supported-unsupported-keyboard-hotkey-shortcuts)
- [Baseten — Wispr Flow customer case study](https://www.baseten.co/resources/customers/wispr-flow/)
- [TechCrunch — Android launch (Feb 2026)](https://techcrunch.com/2026/02/23/wispr-flow-launches-an-android-app-for-ai-powered-dictation/) · [TechCrunch — $30M Series A (June 2025)](https://techcrunch.com/2025/06/24/wispr-flow-raises-30m-from-menlo-ventures-for-its-ai-powered-dictation-app/) · [PRNewswire — $4.6M seed, neural interface (2021)](https://www.prnewswire.com/news-releases/wispr-ai-secures-4-6m-from-nea-and-8vc-to-build-thought-powered-neural-interface-301433912.html) · [Wikipedia — Wispr](https://en.wikipedia.org/wiki/Wispr)
- [Apple App Store](https://apps.apple.com/us/app/wispr-flow-ai-voice-keyboard/id6497229487) · [Microsoft Store](https://apps.microsoft.com/detail/9n1b9jwb3m35) · [Google Play](https://play.google.com/store/apps/details?id=com.wispr.flowapp)
- Reviews/comparisons: [Spokenly](https://spokenly.app/blog/wispr-flow-review) · [eesel AI](https://www.eesel.ai/blog/wispr-flow-review) · [Zapier](https://zapier.com/blog/wispr-flow/) · [Superwhisper for Windows](https://superwhisper.com/windows) · [Aqua Voice](https://aquavoice.com/download)
