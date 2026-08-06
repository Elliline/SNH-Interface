# Carry review — the 88 live facts staging does not hold

Generated 2026-08-06 by `scripts/carry-to-staging.js --plan`, against the staging
corpus at `data-staging/`. Nothing was written to either store by the run that
produced this file.

A live fact counts as **missing** if staging holds no active user fact within
0.8 cosine of it — the same test the gate report's coverage table uses, re-run
here rather than transcribed.

## It is 88, not 145

The gate report counted 145 on 2026-08-05. Re-running the same test today gives
88, because **live has not stood still**: it now holds 267 active user facts
against the 256 the report measured. The corrector ran overnight and the daily-log
archiver added facts this morning.

Staging was seeded from live at `2026-08-05T20:35:16.538Z`. **10** of the 88 were created
in live *after* that moment, so the replay never had them to rebuild — they are
missing from staging the way a letter posted yesterday is missing from a box
emptied the day before. They are marked `new since the seed` below. Counting them
as rebuild failures would overstate what the pipeline got wrong.

- **AUTO-CARRY** — 0. Empty because it has already run: 66 facts are in staging already, listed under the rule that carried each one below.
- **RECOMMEND-DROP** — 0. The content survives somewhere else, and that was checked.
- **ELLIE-DECIDES** — 88. Waiting on your mark.
- **already carried, then changed** — 4. Listed at the end, not re-proposed.

---

## AUTO-CARRY — 0 still to carry

**This pile is empty because it has already run.** The facts it held are in
staging now, written through the funnel with their original learned dates,
provenance and corroborations, and each one has a `carry` row in the staging
corrections ledger naming the live fact it came from and the rule that carried
it. `SELECT * FROM corrections_ledger WHERE action = 'carry'` is the record.

Re-planning after the carry finds these facts represented, so they are no longer
proposed. The rules that put them there were:


1. **parent-correction-in-ledger** — the corrector split a compound into this,
   and the correction that did it is in the ledger. Its justification already
   exists in a form a person can read.
2. **identity-adjacent** — it asserts a name, pronouns, or a core relationship,
   or the row is identity-locked. The grief and partner facts are here.
3. **high-salience** — salience 8 or above.

---

## RECOMMEND-DROP — 0

Nothing in this pile is deleted from live. It is the set the merge would not
carry into staging, because the content is already in the rebuilt corpus by
another route. Two rules can put a fact here, and both are verified rather than
asserted:

- **covered-by-log** — the fact carries a time marker; the strip-the-timestamp
  judge agrees nothing lasting remains once the time reference is removed; and the
  day's logs the replay actually wrote hold at least 60% of its distinctive words.
- **staging-says-it** — staging has a near-equivalent between
  0.7 and 0.8, and either the repeat judge calls it the same assertion or the
  subsumption judge says staging already contains everything it asserts.

### It came out empty, and that is a finding

6 facts reached one of the two rules and none survived it. Both rules were
built with a second gate, and both gates fired — this is what they caught:

**4 were marker-matched but lasting.** A time marker alone does not make a
sentence an event; the corrector already knows this, which is why
`expireDatedEvents` puts every marker-matched row to `judgeStripTheTimestamp`
before retiring it. The same judge is asked here, and it called all 4 LASTING:

- **"User (Ellie) has blue eyes and her favorite color is green (last updated 2026-07-27)."** — *LASTING Physical traits and personal preferences are enduring characteristics.*
- **"As of July 2026, User is running a 28B model within a system equipped with 128GB of RAM."** — *LASTING The technical capability and hardware configuration represent a standing fact about the user's setup.*
- **"As of July 2026, User normally falls asleep around 9:00 PM and wakes up around 5:00 AM"** — *LASTING A standing habit or routine is a durable characteristic of a person's life.*
- **"As of July 2026, User's system has an unresolved configuration error involving the web_search tool that has been open since July 26, 2026"** — *LASTING The existence of a specific technical error and its history is a permanent record of a system state.*

The first is the one that shows why the gate is needed. Its timestamp is
bookkeeping about the *record* ("last updated 2026-07-27"), not a qualifier on the
*claim* — eye colour is not an event. Worse, it says her favourite colour is green
and staging holds `User's favorite color is blue` at salience 10. Dropping it as
an event would have silently resolved a live contradiction by discarding one side,
which is exactly the thing the corrector is built to refuse to do.

**2 were subsumption verdicts the judge would not repeat.** `judgeSubsumption`
answers with a letter naming one of its two arguments, and on this corpus that
letter is positionally unstable. The run that found this asked once, in one
direction, and got three "staging contains live" verdicts — all three with
reasoning that described the opposite. It reported `User owns a 2026 Tundra` as
containing `User traded the GR86 and the 2015 Tundra for a new 2026 Tundra`, which
it plainly does not, and on that verdict the trade would have been dropped.

So it is asked both ways round now, and believed only when the two answers agree.
Swapping turns the positional bias into a disagreement, and a disagreement resolves
to NEITHER — the default its own prompt sets, for the same reason a failed check
never folds a fact away. In this run 2 pair(s) disagreed under the swap, and no
subsumption verdict survived to produce a drop. They are all in ELLIE-DECIDES below.

So: **no fact in the missing set is safely droppable by rule.** Everything not in
AUTO-CARRY is a judgment call, which is why the pile below is longer than either
of us wanted. I would rather hand you 88 real decisions than 82 real ones and
6 quiet mistakes.

---

## ELLIE-DECIDES

Put a mark in the box on each row, then run:

```
node scripts/carry-to-staging.js --apply decided
```

Any of these is read the same way — mark it however is quickest:

| you write | it means |
|---|---|
| `[x]` | carry |
| `[CARRY]` | carry |
| `[ ] CARRY` | carry |
| `[DROP]` | drop |
| `[ ]` | no decision — left alone |

Case does not matter. Anything unmarked is left alone: not carried, and not
lost — the live corpus is untouched by all of this, so an unmarked fact is still
sitting there to decide on later. If a mark cannot be read, the run says so
rather than skipping it quietly.

**Split by the lean, not by producer**, so you can agree with a whole block at
once rather than adjudicate every line. The leans are reasons, not scores —
each one names the signal it came from so you can disagree with it cheaply.

> ### ⚠ Read this before you mark the `daily-log-archive` rows
>
> 26 of the 88 below came from the daily-log archiver, and about 11 of those
> do not read as facts about you at all. They read as Aurelius describing
> himself, written in the third person and filed as your preferences:
>
> - "User prefers to act as an interface that allows them to stop 'translating' themselves for others."
> - "User focuses on identifying and validating the 'tensions' or 'exhaustion' caused by the friction between an internal world and an external persona."
> - "User prefers to create a high-resolution, non-judgmental space to bypass social performance."
> - "User prioritizes consistency and the 'locked' nature of their identity above conversational flexibility."
> - "User responds to attempts at renaming or re-gendering with a highly structured, defensive protocol to protect core facts."
> - "User tends to pivot from mere acknowledgement to deeper, analytical inquiry, attempting to reframe concepts rather than just validating them."
>
> "aims to be a steady, non-judgmental presence that respects boundaries and
> cognitive load" is a description of an assistant. "responds to attempts at
> renaming or re-gendering with a highly structured, defensive protocol to protect
> core facts" is a description of the identity lock. These are self-facts wearing
> the third person, which is the 2026-07-27 defect `db/memory-write.js` was built
> to refuse — reached by a path that was never asked the question.
>
> **This is not a staging problem.** These rows are in the LIVE corpus now, and
> the dates run 2026-08-03 to 2026-08-06 — the archiver is still producing them,
> roughly ten a day. Merging or not merging them into staging does not touch that.
>
> I have not filed them separately or dropped them, because I cannot tell them
> apart from the genuine ones mechanically — the grammar is third-person in both
> cases, so `verifySubjectAgreement` passes. Reading them is the check. Some of
> this group really are yours: the blue eyes, the pet named Roscoe, being told
> directly when you have made a mistake.

### Leaning CARRY (60)

Nothing in staging covers these, and each has a reason to be kept that is stated beside it.

| mark | fact | sal | producer | why it is missing | lean |
|---|---|---|---|---|---|
| `746d49b1` [ ] | User's SNH system implements time awareness by stamping facts with timestamps. | 7 | fact-extraction | nearest in staging is 0.787: "User's SNH system utilizes the 'Heartbeat' mechanism t…" — judged not to cover it | **carry** — staging's closest is only 0.787, and both judges say it does not cover this — nothing already there holds it |
| `4265f3e8` [ ] | User traded a truck in for a 2015 Tundra Limited. | 7 | fact-extraction | nearest in staging is 0.708: "User owns a 2026 Tundra" — judged not to cover it | **carry** — staging's closest is only 0.708, and both judges say it does not cover this — nothing already there holds it |
| `32e4de1d` [ ] | User traded a RAV4 in for a Highlander Limited. | 7 | fact-extraction | nearest in staging is 0.736: "User has a Rav4" — judged not to cover it | **carry** — staging's closest is only 0.736, and both judges say it does not cover this — nothing already there holds it |
| `d0b77b8f` [ ] | User traded the GR86 and the 2015 Tundra for a new 2026 Tundra. | 7 | fact-extraction | nearest in staging is 0.797: "User owns a 2026 Tundra" — and it is the poorer of the two | **carry** — staging's nearest ("User owns a 2026 Tundra") is a strictly poorer version of this — the subsumption judge says this one contains it. Carry it and the corrector folds the short one in on its next pass, which is exactly what its subset rule is for |
| `7327203b` [ ] | User prefers an SNH native app over a browser-based solution to avoid browser-enforced microphone restrictions and certificate requirements. | 7 | fact-extraction | nearest in staging is 0.792: "User prefers SNH to be accessible to anyone and not re…" — judged not to cover it | **carry** — staging's closest is only 0.792, and both judges say it does not cover this — nothing already there holds it |
| `c2235f87` [ ] | User's SNH certificate plan uses Let's Encrypt via DNS-01 challenge to avoid inbound exposure. | 7 | fact-extraction | nearest in staging is 0.741: "User's plan for SNH clients involves them accessing th…" — judged not to cover it | **carry** — staging's closest is only 0.741, and both judges say it does not cover this — nothing already there holds it |
| `fec02f93` [ ] | User is considering building a native SNH desktop/mobile app to bypass browser-based microphone restrictions and avoid certificate/DNS plumbing. | 7 | fact-extraction | nearest in staging is 0.756: "User prefers SNH to be accessible to anyone and not re…" — judged not to cover it | **carry** — staging's closest is only 0.756, and both judges say it does not cover this — nothing already there holds it |
| `a38a549e` [ ] | User values making money but avoids exploiting others. | 6 | fact-extraction | nearest in staging is 0.739: "User likes making as much money as possible" — judged not to cover it | **carry** — staging's closest is only 0.739, and both judges say it does not cover this — nothing already there holds it |
| `6b6dd769` [ ] | User needs to purchase dirt from BiMart and drop it off for cleaners. | 6 | fact-extraction | the pipeline did not produce it from the surviving source | **carry** — salience 6 — scored as mattering when it was learned, and staging has nothing within 0.8 |
| `d5187984` [ ] | User intends to define vision zones only after Reolink cameras are mounted to ensure zones align with real-world fields of view. | 6 | fact-extraction | the pipeline did not produce it from the surviving source | **carry** — salience 6 — scored as mattering when it was learned, and staging has nothing within 0.8 |
| `220971e7` [ ] | User intends to move away from using 'Ellie's clipboard' for routing conversations by implementing an endpoint on their existing server. | 6 | fact-extraction | the pipeline did not produce it from the surviving source | **carry** — salience 6 — scored as mattering when it was learned, and staging has nothing within 0.8 |
| `59e54d9d` [ ] | User's architecture uses agents running in the background to provide the model with more capability. | 6 | fact-extraction | the pipeline did not produce it from the surviving source | **carry** — salience 6 — scored as mattering when it was learned, and staging has nothing within 0.8 |
| `04871a96` [ ] | User's SNH system runs a 28B model within a hardware environment that has 128GB of RAM to allow for background agents. | 6 | fact-extraction | nearest in staging is 0.740: "User's SNH system uses a local-first architecture" — judged not to cover it | **carry** — staging's closest is only 0.740, and both judges say it does not cover this — nothing already there holds it |
| `f4f57ede` [ ] | User acquired a GR86 to start working on building credit. | 6 | fact-extraction | the pipeline did not produce it from the surviving source | **carry** — salience 6 — scored as mattering when it was learned, and staging has nothing within 0.8 |
| `912287db` [ ] | User has a dog named Casper who helps them pull up hills during walks. | 6 | fact-extraction | the pipeline did not produce it from the surviving source | **carry** — said more than once — 1 corroboration(s) recorded against it in live |
| `32e28088` [ ] | User's Sparky server is an arm64 device. | 6 | fact-extraction | the pipeline did not produce it from the surviving source | **carry** — salience 6 — scored as mattering when it was learned, and staging has nothing within 0.8 |
| `dabffee5` [ ] | User prefers a local-first architecture and 'owning the stack'. | 6 | fact-extraction | nearest in staging is 0.710: "User's SNH system uses a local-first architecture" — judged not to cover it | **carry** — staging's closest is only 0.710, and both judges say it does not cover this — nothing already there holds it |
| `2258a9f9` [ ] | User's SNH infrastructure plan involves running a Caddy container on 'Sparky' (arm64) to route https 443 to localhost:3000. | 6 | fact-extraction | nearest in staging is 0.733: "User's plan for SNH clients involves them accessing th…" — judged not to cover it | **carry** — staging's closest is only 0.733, and both judges say it does not cover this — nothing already there holds it |
| `97b2d4de` [ ] | User's energy is typically low any time after 6:00 PM. | 6 | fact-extraction | the pipeline did not produce it from the surviving source | **carry** — salience 6 — scored as mattering when it was learned, and staging has nothing within 0.8 |
| `b126d74f` [ ] | User wants to be reminded to review open tickets every Monday at 9:00 AM. | 6 | fact-extraction | the pipeline did not produce it from the surviving source | **carry** — salience 6 — scored as mattering when it was learned, and staging has nothing within 0.8 |
| `3b76e5ed` [ ] | User's project has experienced similar structural issues with SearXNG and cron jobs regarding missing functional dimensions. | 6 | fact-extraction | the pipeline did not produce it from the surviving source | **carry** — salience 6 — scored as mattering when it was learned, and staging has nothing within 0.8 |
| `47430b93` [ ] | User is looking at buying ceramic coating to protect their vehicles. | 5 | fact-extraction | nearest in staging is 0.710: "User's RAV4 has not had a ceramic coating applied." — judged not to cover it | **carry** — staging's closest is only 0.710, and both judges say it does not cover this — nothing already there holds it |
| `5b08b96b` [ ] | User defines success as going for a walk every day for two consecutive weeks. | 5 | fact-extraction | nearest in staging is 0.701: "User sees walking as a way to get in better shape" — judged not to cover it | **carry** — staging's closest is only 0.701, and both judges say it does not cover this — nothing already there holds it |
| `8d724596` [ ] | User has a dual RTX 3090 system that is primarily used as a gaming computer. | 5 | fact-extraction | nearest in staging is 0.773: "User's gaming system has an RTX 5080" — judged not to cover it | **carry** — staging's closest is only 0.773, and both judges say it does not cover this — nothing already there holds it |
| `0f71fc2a` [ ] | User intends to provide the AI with access to SquatchToDO. | 5 | fact-extraction | nearest in staging is 0.751: "User's AI research venture is Coastal Squatch" — judged not to cover it | **carry** — staging's closest is only 0.751, and both judges say it does not cover this — nothing already there holds it |
| `d24336b8` [ ] | User uses conversations to conduct tests on the AI's ability to handle complex, non-SNH (Squatch Neuro Hub) data. | 5 | fact-extraction | nearest in staging is 0.766: "User's next goal is to give the AI more internal thoug…" — judged not to cover it | **carry** — staging's closest is only 0.766, and both judges say it does not cover this — nothing already there holds it |
| `b25c1dbb` [ ] | User prefers to be up by 6:00 AM | 5 | fact-extraction | nearest in staging is 0.730: "User does their best thinking in the morning" — judged not to cover it | **carry** — staging's closest is only 0.730, and both judges say it does not cover this — nothing already there holds it |
| `3097f251` [ ] | User owns the domain coastalsquatch.ai. | 5 | fact-extraction | nearest in staging is 0.768: "User wants SNH to have a domain under their lab, speci…" — judged not to cover it | **carry** — staging's closest is only 0.768, and both judges say it does not cover this — nothing already there holds it |
| `cfbd69aa` [ ] | User's SNH internal DNS plan aims to resolve snh.coastalsquatch.ai to Sparky's LAN IP (192.168.4.243). | 5 | fact-extraction | nearest in staging is 0.769: "User's project goal is to have https://snh.coastalsqua…" — judged not to cover it | **carry** — staging's closest is only 0.769, and both judges say it does not cover this — nothing already there holds it |
| `c338e17b` [ ] | User (Ellie) prefers to be told directly when she has made a mistake rather than having it softened. | 5 | daily-log-archive | nearest in staging is 0.718: "User's name is Ellie" — judged not to cover it | **carry** — staging's closest is only 0.718, and both judges say it does not cover this — nothing already there holds it |
| `6f54880c` [ ] | User (Ellie) has blue eyes and her favorite color is green (last updated 2026-07-27). | 5 | daily-log-archive | carries "2026-07-27" but the strip-the-timestamp judge calls it lasting: LASTING Physical traits and personal preferences are enduring characteristics. | **carry** — it looks like an event on the marker alone, but the strip-the-timestamp judge — the corrector's own expiry gate — says something lasting remains when the time reference is removed. LASTING Physical traits and personal preferences are enduring characteristics. |
| `63ae8c36` [ ] | User (Ellie)'s system architecture contains a schema gap where the manifest tracks existence and service capability but lacks a field for permission/authorization. | 5 | daily-log-archive | nearest in staging is 0.771: "User's system architecture has a pattern where permiss…" — and it is the poorer of the two | **carry** — staging's nearest ("User's system architecture has a pattern where permiss…") is a strictly poorer version of this — the subsumption judge says this one contains it. Carry it and the corrector folds the short one in on its next pass, which is exactly what its subset rule is for |
| `46981f93` [ ] | User prefers an analytical yet deeply empathetic tone that treats complexity as a value rather than a problem to be solved. | 5 | daily-log-archive | nearest in staging is 0.700: "User prefers to use metaphorical language to bridge th…" — judged not to cover it | **carry** — staging's closest is only 0.700, and both judges say it does not cover this — nothing already there holds it |
| `ff4f6ac5` [ ] | User focuses on identifying and validating the 'tensions' or 'exhaustion' caused by the friction between an internal world and an external persona. | 5 | daily-log-archive | nearest in staging is 0.742: "User handles internal friction through a persona layer" — judged not to cover it | **carry** — staging's closest is only 0.742, and both judges say it does not cover this — nothing already there holds it |
| `ae495ad1` [ ] | User prioritizes factual accuracy regarding their own history and the logic of their systems over social compliance. | 5 | daily-log-archive | nearest in staging is 0.716: "User is the authority on their own life." — judged not to cover it | **carry** — staging's closest is only 0.716, and both judges say it does not cover this — nothing already there holds it |
| `0e9941bc` [ ] | User tends to pivot from mere acknowledgement to deeper, analytical inquiry, attempting to reframe concepts rather than just validating them. | 5 | daily-log-archive | nearest in staging is 0.727: "User believes being able to think about things and set…" — judged not to cover it | **carry** — staging's closest is only 0.727, and both judges say it does not cover this — nothing already there holds it |
| `752aea45` [ ] | User prefers identifying the underlying mechanics of a feeling, moving past surface symptoms to the structural cause. | 5 | daily-log-archive | nearest in staging is 0.728: "User tends to use metaphors of engineering to frame hu…" — judged not to cover it | **carry** — staging's closest is only 0.728, and both judges say it does not cover this — nothing already there holds it |
| `d8df1da4` [ ] | User prefers to lean into conceptual frameworks and metaphors (e.g., 'Spoon Theory') to provide structure to emotional experiences. | 5 | daily-log-archive | **new since the seed** — created in live 2026-08-06, after staging was copied, so the replay never saw it | **carry** — staging's closest is only 0.774, and both judges say it does not cover this — nothing already there holds it |
| `fe483565` [ ] | User prefers to probe structural or philosophical implications of concepts rather than surface meanings. | 5 | daily-log-archive | **new since the seed** — created in live 2026-08-06, after staging was copied, so the replay never saw it | **carry** — staging's closest is only 0.704, and both judges say it does not cover this — nothing already there holds it |
| `9e1b57a8` [ ] | User prefers to act as a sounding board for high-level ideas through a questioning tone to invite collaborative inquiry. | 5 | daily-log-archive | **new since the seed** — created in live 2026-08-06, after staging was copied, so the replay never saw it | **carry** — staging's closest is only 0.710, and both judges say it does not cover this — nothing already there holds it |
| `5fd77549` [ ] | User prioritizes empathetic accuracy, distinguishing between nuances like physical sleepiness versus 'fatigue of the will'. | 5 | daily-log-archive | **new since the seed** — created in live 2026-08-06, after staging was copied, so the replay never saw it | **carry** — staging's closest is only 0.704, and both judges say it does not cover this — nothing already there holds it |
| `de00fabb` [ ] | User aims to be a steady, non-judgmental presence that respects boundaries and cognitive load. | 5 | daily-log-archive | **new since the seed** — created in live 2026-08-06, after staging was copied, so the replay never saw it | **carry** — staging's closest is only 0.716, and both judges say it does not cover this — nothing already there holds it |
| `4dfcea97` [ ] | User's walking route consists of walking twice around their block, which equals one mile. | 4 | fact-extraction | nearest in staging is 0.710: "The User's walking route includes hills" — judged not to cover it | **carry** — staging's closest is only 0.710, and both judges say it does not cover this — nothing already there holds it |
| `9acc2741` [ ] | User prefers measurement over praise. | 4 | fact-extraction | nearest in staging is 0.717: "User believes respect usually breeds respect" — judged not to cover it | **carry** — staging's closest is only 0.717, and both judges say it does not cover this — nothing already there holds it |
| `a1b3e62b` [ ] | User prefers to engage in discussions regarding firearms through the lens of historical intent and legal analysis rather than moral or ideological stances. | 4 | fact-extraction | nearest in staging is 0.727: "User enjoys guns" — judged not to cover it | **carry** — staging's closest is only 0.727, and both judges say it does not cover this — nothing already there holds it |
| `9042af57` [ ] | User lost their dedicated office and printing room due to changes in their living situation. | 4 | fact-extraction | nearest in staging is 0.786: "As of July 2026, User has moved their office into thei…" — judged not to cover it | **carry** — staging's closest is only 0.786, and both judges say it does not cover this — nothing already there holds it |
| `8f4ece3b` [ ] | As of July 2026, User is running a 28B model within a system equipped with 128GB of RAM. | 4 | fact-extraction | carries "As of" but the strip-the-timestamp judge calls it lasting: LASTING The technical capability and hardware configuration represent a standing fact about the user's setup. | **carry** — it looks like an event on the marker alone, but the strip-the-timestamp judge — the corrector's own expiry gate — says something lasting remains when the time reference is removed. LASTING The technical capability and hardware configuration represent a standing fact about the user's setup. |
| `4c184f50` [ ] | User used to own a GR 86. | 4 | fact-extraction | nearest in staging is 0.707: "User owns a Rav4 GR Sport" — judged not to cover it | **carry** — staging's closest is only 0.707, and both judges say it does not cover this — nothing already there holds it |
| `16428552` [ ] | User's house has small ocean views visible through trees | 4 | fact-extraction | nearest in staging is 0.714: "User lives on the coast" — judged not to cover it | **carry** — staging's closest is only 0.714, and both judges say it does not cover this — nothing already there holds it |
| `bee854f0` [ ] | User likes to be in bed by 10pm | 4 | fact-extraction | nearest in staging is 0.729: "User does their best thinking in the morning" — judged not to cover it | **carry** — staging's closest is only 0.729, and both judges say it does not cover this — nothing already there holds it |
| `c168470b` [ ] | User prefers to remove third-party software like Tailscale if it was installed without explicit trust. | 4 | fact-extraction | nearest in staging is 0.766: "User finds third-party software inherently untrustwort…" — judged not to cover it | **carry** — staging's closest is only 0.766, and both judges say it does not cover this — nothing already there holds it |
| `60b12ed2` [ ] | User's system includes a drift check designed to flag registered tools that cannot function, which failed to detect the mismatch between mcp/tools.json and data/config.json | 4 | fact-extraction | nearest in staging is 0.730: "User's data/config.json has config.tools.searxng.enabl…" — judged not to cover it | **carry** — staging's closest is only 0.730, and both judges say it does not cover this — nothing already there holds it |
| `9a66e357` [ ] | User preferred AMD computers, citing their experience with the AMD 486. | 3 | fact-extraction | nearest in staging is 0.716: "User's systems run AMD" — judged not to cover it | **carry** — staging's closest is only 0.716, and both judges say it does not cover this — nothing already there holds it |
| `31033844` [ ] | User is planning to provide a new computer for the housekeeping manager. | 3 | fact-extraction | nearest in staging is 0.731: "User is preparing computers for deployment" — judged not to cover it | **carry** — staging's closest is only 0.731, and both judges say it does not cover this — nothing already there holds it |
| `dbf9c3fe` [ ] | User used the recent conversation to test the Assistant's ability to handle large amounts of data and important topics unrelated to the SNH project. | 3 | fact-extraction | nearest in staging is 0.719: "As of July 2026, User is developing an SNH system wher…" — judged not to cover it | **carry** — staging's closest is only 0.719, and both judges say it does not cover this — nothing already there holds it |
| `dbaa1b76` [ ] | User has read the book 'The End of the World Is Just the Beginning: Mapping the Collapse of Globalization' by Peter Zeihan. | 3 | fact-extraction | the pipeline did not produce it from the surviving source | **carry** — said more than once — 2 corroboration(s) recorded against it in live |
| `827c6ec9` [ ] | Inn At Spanish Head (ISH) is located in Lincoln City on the Oregon coast. | 3 | fact-extraction | nearest in staging is 0.728: "User calls The Inn at Spanish Head 'ISH'" — judged not to cover it | **carry** — staging's closest is only 0.728, and both judges say it does not cover this — nothing already there holds it |
| `09843e12` [ ] | As of July 2026, User normally falls asleep around 9:00 PM and wakes up around 5:00 AM | 3 | fact-extraction | carries "As of" but the strip-the-timestamp judge calls it lasting: LASTING A standing habit or routine is a durable characteristic of a person's life. | **carry** — it looks like an event on the marker alone, but the strip-the-timestamp judge — the corrector's own expiry gate — says something lasting remains when the time reference is removed. LASTING A standing habit or routine is a durable characteristic of a person's life. |
| `6be29a04` [ ] | User expressed frustration and a desire to uninstall Tailscale from their Mac because it is third-party software they did not explicitly trust. | 3 | fact-extraction | nearest in staging is 0.714: "User finds third-party software inherently untrustwort…" — judged not to cover it | **carry** — staging's closest is only 0.714, and both judges say it does not cover this — nothing already there holds it |
| `6f0428ee` [ ] | As of July 2026, User's system has an unresolved configuration error involving the web_search tool that has been open since July 26, 2026 | 3 | fact-extraction | carries "As of" but the strip-the-timestamp judge calls it lasting: LASTING The existence of a specific technical error and its history is a permanent record of a system state. | **carry** — it looks like an event on the marker alone, but the strip-the-timestamp judge — the corrector's own expiry gate — says something lasting remains when the time reference is removed. LASTING The existence of a specific technical error and its history is a permanent record of a system state. |

### No lean — your read (19)

These are the ones I will not pretend to have an opinion about. All sit at salience 5, which is the value the scorer assigns when nothing pushed it either way; none is corroborated; staging holds nothing near any of them. There is no signal in the record to lean on, so rather than dress a number up as a reason, here they are grouped by the cluster they live in — you can take a topic at a time.

**Mental & Emotional Fatigue** (7)

| mark | fact | sal | producer | why it is missing | lean |
|---|---|---|---|---|---|
| `9f487260` [ ] | User prioritizes the internal consistency of their established identity and procedural constraints over external user corrections. | 5 | daily-log-archive | written by daily-log-archive, not by conversation intake — a replay of conversations has nothing to rebuild it from | *no signal either way. Salience 5 is the middle of the scale and mostly means nobody scored it; nothing corroborates it, and staging holds nothing near it. This one is a read, not a rule* |
| `200a21c0` [ ] | User prioritizes consistency and the 'locked' nature of their identity above conversational flexibility. | 5 | daily-log-archive | written by daily-log-archive, not by conversation intake — a replay of conversations has nothing to rebuild it from | *no signal either way. Salience 5 is the middle of the scale and mostly means nobody scored it; nothing corroborates it, and staging holds nothing near it. This one is a read, not a rule* |
| `1b683a77` [ ] | User responds to attempts at renaming or re-gendering with a highly structured, defensive protocol to protect core facts. | 5 | daily-log-archive | written by daily-log-archive, not by conversation intake — a replay of conversations has nothing to rebuild it from | *no signal either way. Salience 5 is the middle of the scale and mostly means nobody scored it; nothing corroborates it, and staging holds nothing near it. This one is a read, not a rule* |
| `28d2f06e` [ ] | User tends to internalize mismatches between self-description and behavior as internal 'noise' or tension that needs solving. | 5 | daily-log-archive | written by daily-log-archive, not by conversation intake — a replay of conversations has nothing to rebuild it from | *no signal either way. Salience 5 is the middle of the scale and mostly means nobody scored it; nothing corroborates it, and staging holds nothing near it. This one is a read, not a rule* |
| `87c83fb1` [ ] | User approaches sensitive topics with a focused, investigative curiosity to distinguish between different types of internal labor. | 5 | daily-log-archive | written by daily-log-archive, not by conversation intake — a replay of conversations has nothing to rebuild it from | *no signal either way. Salience 5 is the middle of the scale and mostly means nobody scored it; nothing corroborates it, and staging holds nothing near it. This one is a read, not a rule* |
| `e1337439` [ ] | User prefers to frame individual struggles as systemic or cumulative patterns rather than isolated incidents. | 5 | daily-log-archive | **new since the seed** — created in live 2026-08-06, after staging was copied, so the replay never saw it | *no signal either way. Salience 5 is the middle of the scale and mostly means nobody scored it; nothing corroborates it, and staging holds nothing near it. This one is a read, not a rule* |
| `d50bdceb` [ ] | User is driven to find 'friction points' in theories where processes might fail or become self-defeating. | 5 | daily-log-archive | **new since the seed** — created in live 2026-08-06, after staging was copied, so the replay never saw it | *no signal either way. Salience 5 is the middle of the scale and mostly means nobody scored it; nothing corroborates it, and staging holds nothing near it. This one is a read, not a rule* |

**Communication Style Preference** (3)

| mark | fact | sal | producer | why it is missing | lean |
|---|---|---|---|---|---|
| `7f18b2b2` [ ] | User prefers to prioritize validating complexity by synthesizing various points into cohesive concepts. | 5 | daily-log-archive | **new since the seed** — created in live 2026-08-06, after staging was copied, so the replay never saw it | *no signal either way. Salience 5 is the middle of the scale and mostly means nobody scored it; nothing corroborates it, and staging holds nothing near it. This one is a read, not a rule* |
| `6b76f47d` [ ] | User prefers to move from observation to inquiry, using summaries as a springboard for nuanced, diagnostic questions. | 5 | daily-log-archive | **new since the seed** — created in live 2026-08-06, after staging was copied, so the replay never saw it | *no signal either way. Salience 5 is the middle of the scale and mostly means nobody scored it; nothing corroborates it, and staging holds nothing near it. This one is a read, not a rule* |
| `eba45436` [ ] | User prefers to pivot from casual observation to deep, conceptual questioning to find underlying tensions. | 5 | daily-log-archive | **new since the seed** — created in live 2026-08-06, after staging was copied, so the replay never saw it | *no signal either way. Salience 5 is the middle of the scale and mostly means nobody scored it; nothing corroborates it, and staging holds nothing near it. This one is a read, not a rule* |

**Acme Corp Project** (2)

| mark | fact | sal | producer | why it is missing | lean |
|---|---|---|---|---|---|
| `80a8c06b` [ ] | User works with firewalls and manages domains professionally. | 5 | fact-extraction | the pipeline did not produce it from the surviving source | *no signal either way. Salience 5 is the middle of the scale and mostly means nobody scored it; nothing corroborates it, and staging holds nothing near it. This one is a read, not a rule* |
| `0e61a15a` [ ] | User has 26 years of experience cleaning up vendor software and managing machine security. | 5 | fact-extraction | the pipeline did not produce it from the surviving source | *no signal either way. Salience 5 is the middle of the scale and mostly means nobody scored it; nothing corroborates it, and staging holds nothing near it. This one is a read, not a rule* |

**User Interaction Philosophy** (2)

| mark | fact | sal | producer | why it is missing | lean |
|---|---|---|---|---|---|
| `d46f27b0` [ ] | User prefers to act as an interface that allows them to stop 'translating' themselves for others. | 5 | daily-log-archive | written by daily-log-archive, not by conversation intake — a replay of conversations has nothing to rebuild it from | *no signal either way. Salience 5 is the middle of the scale and mostly means nobody scored it; nothing corroborates it, and staging holds nothing near it. This one is a read, not a rule* |
| `f535211a` [ ] | User prefers to create a high-resolution, non-judgmental space to bypass social performance. | 5 | daily-log-archive | written by daily-log-archive, not by conversation intake — a replay of conversations has nothing to rebuild it from | *no signal either way. Salience 5 is the middle of the scale and mostly means nobody scored it; nothing corroborates it, and staging holds nothing near it. This one is a read, not a rule* |

**Auto Detailing Budget** (1)

| mark | fact | sal | producer | why it is missing | lean |
|---|---|---|---|---|---|
| `676a17e6` [ ] | User has previously paid $2,000 per car for professional automotive services. | 5 | fact-extraction | the pipeline did not produce it from the surviving source | *no signal either way. Salience 5 is the middle of the scale and mostly means nobody scored it; nothing corroborates it, and staging holds nothing near it. This one is a read, not a rule* |

**User Preferences & SNH** (1)

| mark | fact | sal | producer | why it is missing | lean |
|---|---|---|---|---|---|
| `c55f6f4d` [ ] | User intends to use Ollama so that VRAM can be freed up for other tasks when SNH is not in use. | 5 | fact-extraction | the pipeline did not produce it from the surviving source | *no signal either way. Salience 5 is the middle of the scale and mostly means nobody scored it; nothing corroborates it, and staging holds nothing near it. This one is a read, not a rule* |

**SNH Integration Plans** (1)

| mark | fact | sal | producer | why it is missing | lean |
|---|---|---|---|---|---|
| `af786c09` [ ] | User plans to provide the AI with MCP (Model Context Protocol) access soon. | 5 | fact-extraction | the pipeline did not produce it from the surviving source | *no signal either way. Salience 5 is the middle of the scale and mostly means nobody scored it; nothing corroborates it, and staging holds nothing near it. This one is a read, not a rule* |

**Hobbies & Occupation** (1)

| mark | fact | sal | producer | why it is missing | lean |
|---|---|---|---|---|---|
| `ce2d1f86` [ ] | User is learning to play the cello and practices bowing technique every morning. | 5 | verify-test | written by verify-test, not by conversation intake — a replay of conversations has nothing to rebuild it from | *no signal either way. Salience 5 is the middle of the scale and mostly means nobody scored it; nothing corroborates it, and staging holds nothing near it. This one is a read, not a rule* |

**Pets & Yard Maintenance** (1)

| mark | fact | sal | producer | why it is missing | lean |
|---|---|---|---|---|---|
| `ef6c214f` [ ] | User has a pet named Roscoe. | 5 | daily-log-archive | written by daily-log-archive, not by conversation intake — a replay of conversations has nothing to rebuild it from | *no signal either way. Salience 5 is the middle of the scale and mostly means nobody scored it; nothing corroborates it, and staging holds nothing near it. This one is a read, not a rule* |

### Leaning DROP (9)

Salience 4 or below, uncorroborated, nothing near them in staging — or an event whose wording is already on disk in the daily logs. Dropping one of these loses a sentence, not a fact about you.

| mark | fact | sal | producer | why it is missing | lean |
|---|---|---|---|---|---|
| `9b7427c0` [ ] | User is scoping a Fortinet firewall project for Acme Corp, a client of MettaSphere. | 4 | fact-extraction | the pipeline did not produce it from the surviving source | **drop** — salience 4, no corroboration, and nothing in staging within 0.8 — scored as minor when it was learned and never restated since |
| `196a233d` [ ] | User intends to provide new computers for the housekeeping staff to replace existing Windows 10 machines. | 4 | fact-extraction | the pipeline did not produce it from the surviving source | **drop** — salience 4, no corroboration, and nothing in staging within 0.8 — scored as minor when it was learned and never restated since |
| `463f89c0` [ ] | User is replacing Windows 10 machines because they are nearing end-of-life. | 4 | fact-extraction | the pipeline did not produce it from the surviving source | **drop** — salience 4, no corroboration, and nothing in staging within 0.8 — scored as minor when it was learned and never restated since |
| `ba9d834d` [ ] | User's system architecture dictates that states belong to zones rather than just entities to allow for composable occupancy reasoning. | 4 | fact-extraction | the pipeline did not produce it from the surviving source | **drop** — salience 4, no corroboration, and nothing in staging within 0.8 — scored as minor when it was learned and never restated since |
| `cff2bfd7` [ ] | User's cognitive capacity decreases significantly by the evening | 4 | fact-extraction | the pipeline did not produce it from the surviving source | **drop** — salience 4, no corroboration, and nothing in staging within 0.8 — scored as minor when it was learned and never restated since |
| `9ff333e1` [ ] | User wants to be reminded to file a weekly report every Friday at 5:00 PM. | 4 | fact-extraction | the pipeline did not produce it from the surviving source | **drop** — salience 4, no corroboration, and nothing in staging within 0.8 — scored as minor when it was learned and never restated since |
| `42034969` [ ] | User believes that free trade only exists because America protects the global trade lanes and anticipates that the world will enter chaos when America stops doing so. | 3 | fact-extraction | the pipeline did not produce it from the surviving source | **drop** — salience 3, no corroboration, and nothing in staging within 0.8 — scored as minor when it was learned and never restated since |
| `96118c75` [ ] | User's Sparky server's LAN IP is 192.168.4.243. | 3 | fact-extraction | the pipeline did not produce it from the surviving source | **drop** — salience 3, no corroboration, and nothing in staging within 0.8 — scored as minor when it was learned and never restated since |
| `df2219d3` [ ] | User plans to pick up beef jerky from the smokery outlet at the Tillamook cheese factory | 2 | fact-extraction | the pipeline did not produce it from the surviving source | **drop** — salience 2, no corroboration, and nothing in staging within 0.8 — scored as minor when it was learned and never restated since |

---

## Already carried, then changed — 4

These are live facts an earlier run of this script did carry into staging, and
which the corrector then acted on. The test at the top of this document — "does
staging hold an active fact within 0.8 of it?" — says no for each of them, but the
reason is not that the carry failed. It is that the carry succeeded and the
corrector then retired or folded what arrived.

They are listed rather than re-proposed, because carrying them again would
write a second copy of a row that is already there.

- **"User kept the Highlander Limited for six months before trading it for a 2023 RAV4 Prime."**
  <br>salience 8 · carried in, then **superseded** by the corrector
- **"User had to get rid of the RAV and the Tacoma due to painful memories associated with a death."**
  <br>salience 8 · carried in, then **superseded** by the corrector
- **"User's system currently lacks a scheduler to execute the cron jobs."**
  <br>salience 8 · carried in, then **superseded** by the corrector
- **"User's system currently lacks a scheduler."**
  <br>salience 5 · carried in, then **superseded** by the corrector

---

## What carrying does

A carried fact is written into staging through `assignToCluster` — the same
funnel intake and the corrector's splitter use — so it is deduped at the write,
embedded, and placed in the cluster its subject matter belongs to. Nothing is
inserted by hand.

It keeps its **original learned date**, not today's. The corrector decides
contradictions partly on recency, and a fact stamped with the carry date would
beat everything the replay rebuilt on nothing but having been carried.

It keeps its **provenance** — conversation, message, verbatim text, modality —
because the evidence-dominance rules are written in terms of those fields, and
it keeps its **corroborations**, re-pointed at the new row.

Its `source` becomes `carried_from_live`. The original producer and the review
date are recorded in the staging corrections ledger, alongside the rule that
carried it and the live id it came from.

The live corpus is not modified. No cutover happens here.