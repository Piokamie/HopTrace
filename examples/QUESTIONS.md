# Designed multi-hop questions

Each question below is genuinely multi-hop by construction: the answer
document shares **no content words** with the question — it is reachable
only by following the bridge entity from the seed document. The floor
(hops=0) cannot surface the answer; hop retrieval must. Question 7 needs
two bridges in a row (`hop 2`): Marek's file names only his office, and
only the office file says what is outside it.

| # | Question | Seed doc | Bridge entity | Answer doc |
|---|---|---|---|---|
| 1 | Where does the manager of Alicja Rud sit? | people/alicja-rud.md | Marek Sosna | people/marek-sosna.md ("occupies Office B12") |
| 7 | What can the manager of Alicja Rud see from the window? | people/alicja-rud.md | Marek Sosna, then Office B12 (two bridges) | facilities/office-b12.md ("a view of the courtyard birches") |
| 2 | What equipment is in the room where the calibration team meets? | teams/calibration-team.md | Hala D | facilities/hala-d.md ("spectrometer rig and two argon lasers") |
| 3 | Which initiative involves the person who maintains the procurement ledger? | people/tomasz-gil.md | Tomasz Gil | projects/vega.md ("Tomasz Gil contributes weekly to Project Vega") |
| 4 | Who approves invoices sent by the vendor servicing the elevators? | vendors/koleo-serwis.md | Koleo Serwis | people/beata-lis.md ("Statements … cleared by Beata Lis") |
| 5 | When was the building that hosts the archive constructed? | facilities/archive.md | Budynek C | facilities/budynek-c.md ("constructed in 1962") |
| 6 | What colour of pass is needed for the floor where the vault is? | security/vault.md | Level 3 | security/badges.md ("requires an amber badge") |

Design rules used (mirror of the self-benchmark's construction, ADR 0004):

- The bridge entity is named in both the seed and the answer document —
  it is the only strong link between them.
- The question's verbs and nouns are chosen to miss the answer document's
  vocabulary ("sit" vs "occupies", "equipment" vs "houses", "constructed"
  appears only in the answer but the question's other words don't match
  the seed's… the overlap that matters — question ∩ answer-doc — stays
  empty apart from generic stopwords).
- Question 5 deliberately shares one stem with the answer doc
  ("constructed") to demonstrate a *weakly* lexical case: the floor can
  rank the answer somewhere, but only the hop path explains it.
