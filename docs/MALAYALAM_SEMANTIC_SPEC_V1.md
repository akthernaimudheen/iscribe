# Malayalam Clinical Semantic Specification v1

Status: **IMPLEMENTED** — this document specifies what `scribe_engine/normalization.py`,
`semantic.py`, `asr_correction.py`, and `fact_graph.py` actually do, and every rule
here is pinned by a regression test (`tests/test_malayalam_semantic_spec.py`).

The fact graph produced under this spec is the ONLY source of truth for future
note generation. A later LLM note generator must consume the fact graph — never
raw ASR — and the anti-hallucination validator (`fact_graph.validate_fact_graph`)
guards that contract. An LLM may polish prose AFTER facts are fixed; it may never
introduce a fact.

## 0. Pipeline contract

```
ASR transcript (IMMUTABLE — always the record)
  → speaker segments            (diarization; roles may be UNKNOWN)
  → clauses                     (split at punctuation, verb-final polarity verbs,
                                 contrastive connectors)
  → clinical mentions           (alias matching with morphological variants)
  → scoped modifiers            (polarity / uncertainty / resolution / time,
                                 each scoped to the concept's predication)
  → clinical facts              (ClinicalEntity)
  → FACT GRAPH                  (fact_graph.build_fact_graph)
  → (later) HPI generator       consumes the FACT GRAPH, never raw ASR
```

Normalization NEVER rewrites the transcript. Corrections are applied to a
separate copy whose provenance points back to the raw string.

## 1. Clinical status model (do not collapse)

| Status        | Meaning                                | Example |
|---------------|----------------------------------------|---------|
| PRESENT       | explicitly reported                    | പനി ഉണ്ട് |
| ABSENT        | explicitly denied                      | പനി ഇല്ല |
| RESOLVED      | reported, then resolved                | പനി ഉണ്ടായിരുന്നു, ഇപ്പോൾ മാറി |
| UNCERTAIN     | speaker unsure / perception            | പനി ആണെന്ന് തോന്നുന്നു |
| QUESTIONED    | asked, not answered — never a finding  | പനി ഉണ്ടോ? |
| NOT_DOCUMENTED| never discussed — carried by the ABSENCE of a fact | (no mention) |

NOT_DOCUMENTED is never converted into ABSENT. A concept that was not discussed
produces NO fact; a consumer asking about it reads NOT_DOCUMENTED from the
fact graph's silence. Only an explicit denial produces ABSENT.

Entity-level statuses POSSIBLE / UNKNOWN map to fact-level UNCERTAIN; QUESTION
maps to QUESTIONED. Conditional (hypothetical) clauses assert nothing and are
flagged `conditional: true`.

## 2. Scoped uncertainty (never global)

Uncertainty markers ("തോന്നുന്നു" perceive/seem, "ആകാം" may, "സാധ്യതയുണ്ട്"
possible, "ഉറപ്പില്ല" not sure, "അറിയില്ല" don't know) apply to a concept only
inside its **predication window** — from the mention to the next genuine
clause-final polarity marker (an embedded copula subordinated by എന്ന് does not
end a predication), plus up to 25 chars before it.

Rules pinned by tests:

- "പനി ആണെന്ന് തോന്നുന്നു" → fever UNCERTAIN (POSSIBLE), reason
  `uncertainty_marker:െന്ന് തോന്നുന്നു` (fused എന്ന് form matched).
- "പനി ആണെന്ന് തോന്നുന്നു, പക്ഷേ ഉറപ്പില്ല" → UNKNOWN (explicit no-confidence).
- "ക്ഷീണം തോന്നുന്നുണ്ട്" → PRESENT (an experience verb is presence, not doubt).
- "എന്താണ് കാരണമെന്ന് അറിയില്ല, ക്ഷീണം ഉണ്ട്" → fatigue PRESENT; no "unknown"
  symptom is ever created.
- "പനി ഉണ്ടോ എന്ന് അറിയില്ല" → UNKNOWN.
- "പനി ഉണ്ട്, പക്ഷേ ഉറപ്പില്ല" → UNKNOWN (tail clause mutates the previous
  predication; it embeds ഇല്ല but means "not sure", never "not present").
- "ചിലപ്പോൾ ചുമയുണ്ട്" → PRESENT with `frequency=intermittent`; uncertainty_reason
  stays empty (frequency is not doubt).

Every UNCERTAIN fact carries a mandatory `uncertainty_reason`
(`uncertainty_marker:<phrase>` or `interrogative_tail:<phrase>`).

## 3. Scoped negation

Preserved mechanism (unchanged semantics, see tests §G): each concept binds to
its nearest following polarity marker; പക്ഷേ / എന്നാൽ split clauses so one
ഇല്ല never crosses into the next predication. Fused denials (പനിയില്ല,
കഫക്കെട്ടില്ല, ഭക്ഷണം കഴിക്കാൻ ബുദ്ധിമുട്ടില്ല) are morpheme-boundary-checked so a
probability suffix (ഉണ്ടാകാം) is never read as denial.

## 4. Scoped resolution

Preserved mechanism (see tests §H): a resolution verb governs only the mentions
it follows (verb-final); coordinated subjects sharing one final verb resolve
together; a resolution-only clause resolves the previously discussed symptom.

## 5. Temporal fields (no invention)

Populated only with explicit evidence; missing fields stay empty.

- `duration`: "ആറു ദിവസമായി പനി" → "6 days".
- `onset`: duration anchored by an onset construction — fused unit+ായി
  (word-boundary guarded so the past-tense ഉണ്ടായിരുന്നു never matches),
  ായിട്ട്, മുതൽ, English "since". "പനി ഉണ്ടായിരുന്നു" has NO onset.
- `frequency`: raw marker plus normalized label where justified
  (ചിലപ്പോൾ → intermittent, ദിവസവും → daily).
- `temporality`: CURRENT / PAST / FUTURE / RECURRENT / PAST_RESOLVED
  (PAST_RESOLVED is set exactly when status is RESOLVED).
- Course/severity: only from explicit severity markers.

## 6. Entity / fact shape

Entity (semantic layer, subset shown):

```
{concept, english, surface_text, status, temporality, duration, onset,
 frequency, frequency_normalized, severity, confidence, source_clause,
 subject, conditional, uncertainty_reason}
```

Fact (fact graph — the note generator's input):

```
{concept, english, status: PRESENT|ABSENT|RESOLVED|UNCERTAIN|QUESTIONED,
 certainty: CONFIRMED|UNCERTAIN, temporality, duration, onset, frequency,
 surface_text, source_clause, speaker, confidence, attributes,
 uncertainty_reason, raw_text}
```

Unsupported fields are never populated.

## 7. Provenance (mandatory)

Every fact retains: raw ASR text (verbatim, in `raw_text`), surface form,
source clause, speaker, normalized concept, status, certainty, temporal state,
confidence. The raw transcript is never modified to make it "cleaner":
"ചൊമ്മയുണ്ട്" stays queryable as-is, with the corrected copy
("ചുമയുണ്ട്" → cough PRESENT) carrying provenance back to it.

## 8. ASR correction policy (three separated classes)

A. **Semantic aliases** — colloquial-but-correct speech handled by the lexicon
   (പ്രഷർ → hypertension_history, ഷുഗർ → diabetes_history).
B. **Verified ASR corrections** — `asr_correction.py` table only. Every entry
   has: observed ASR form, intended form, resulting clinical concept, evidence
   source, and a regression test. Current verified entries:
   - ചൊമ്മയുണ്ട് → ചുമയുണ്ട് (cough) — real audio, human verified
   - തലവനൊക്കെ → തലവേദനയൊക്കെ (headache) — real audio, human verified
   - കബക്കെട്ട് → കഫക്കെട്ട് (congestion) — real audio, human verified
   - ചവദാഭയബഥ/ചവതാഭയപട → diabetes; ശപ ഐതുക → blood pressure;
     ലാൻഡിലാണ് → nasal spray (context-guarded) — benchmark-observed
C. **Unverified ASR errors** — pass through UNCHANGED (UNKNOWN beats GUESS).
   A lookalike that was never observed (e.g. ചൊമ്മപ്പാട്ട്) is never rewritten.
   No fuzzy/similarity corrections exist or are allowed.

## 9. Colloquial Malayalam support

Supported (linguistically justified or real-audio observed, never speculative):
- Malayalam-English code switching (BP/പ്രഷർ/ഷുഗർ/ഫുഡ് + English markers)
- spoken contractions and colloquial spellings (ഇപ്പൊ, മിനിഞ്ഞാന്ന്)
- suffix attachment and enclitic coordination (ചുമയും, കഫവും)
- final-virama elision (കഫക്കെട്ടും, ബുദ്ധിമുട്ടില്ല)
- conversational expressions (തോന്നുന്നുണ്ട്, അറിയില്ല tails)

## 10. Questions and speakers

A question is never a finding: പനി ഉണ്ടോ? → QUESTION/QUESTIONED. Ask+ഉണ്ട് →
PRESENT; ask+ഇല്ല → ABSENT; ask+no answer → QUESTIONED (no fact with a finding
status). Roles are never invented: when diarization cannot attribute roles the
fact's speaker is the neutral turn label or UNKNOWN, both in entities and in
the fact graph, and the QUESTIONED outcome holds on the roles-unknown path too.

## 11. Fact graph & anti-hallucination validator

`fact_graph.build_fact_graph(entities, turns, raw_text, roles_known)` builds the
graph; `validate_fact_graph(graph)` rejects:
- any fact missing surface_text, source_clause, confidence, or speaker;
- any fact whose concept has no surface evidence (inference-only concept);
- any section fact in a hallucination-prone category (assessment, medications,
  allergies, PMH, family/social history, physical exam, plan) without full
  provenance.

The validator's report (`{valid, violations, checked_facts}`) is the gate a
future note generator must pass before generating prose.
