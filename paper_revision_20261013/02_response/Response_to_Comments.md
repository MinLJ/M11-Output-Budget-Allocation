# Response to the Review Comments

**Manuscript:** *Learning Prefix Marginal Utility for Resource-Aware Output Budget Allocation in Intelligent Vehicle Perception*

**Date:** 13 October 2026

**Status:** All 19 comments have been addressed. The revised manuscript is included in this package as `01_manuscript/M11_manuscript_v2_revised.docx` (with a PDF rendering and the Markdown source). Four points listed under "Items for confirmation" at the end of this document require the corresponding author's decision before submission.

All revision work used the existing frozen artifacts only: no model was retrained, no allocation was re-run, and no metric was recomputed for the manuscript. Where a number appears in the revision, it comes either from the frozen per-seed evaluation records shipped with each arm (recomputed by the script in `03_reproducibility/`) or from values already reported in the original submission.

---

## 1. Title

**Comment:** The title may change to one of two suggested options.

**Response:** The second suggested title has been adopted:

> Learning Prefix Marginal Utility for Resource-Aware Output Budget Allocation in Intelligent Vehicle Perception

The title has been changed on the title page. The first suggestion ("Resource-Aware Output Budget Allocation for Road-Scene Object Detection") was not adopted because the revised paper reports cross-detector evidence on both road scenes and the COCO2017 Road8 subset, and because the marginal-utility learning is the paper's central mechanism. If the first option is preferred, the change is confined to the title page and the running head.

## 2. References (about 30 requested)

**Comment:** Add more references, total about 30.

**Response:** The bibliography now contains 33 entries (from 16 in the original submission). Every added entry was checked against a venue record — title, authors, year, and pages — before inclusion; entries that could not be confirmed were not added. References are numbered in order of first citation, as IEEE style requires; the reference-list order therefore differs from the original submission. The new material supports the places that previously carried unsupported breadth:

- Efficient detector architectures and query-based selection: DETR [3], Deformable DETR [4], DINO [5], Sparse DETR [6], Efficient DETR [7]. These now anchor the "efficient detection" sentence in Section I and the first Related Work paragraph.
- Dense and two-stage candidate sources: SSD [8], YOLO [9], Focal Loss / RetinaNet [10], Cascade R-CNN [11]. These support the statement that dense and two-stage families set their candidate sets through different mechanisms.
- Dynamic inference beyond DynamicDet [12]: the dynamic-networks survey [13] and Resolution Adaptive Networks [14].
- Calibration and suppression: Guo et al. [17] for miscalibration of modern networks, and Learned NMS [25] for selection inside suppression.
- Collaborative perception beyond Where2comm and WhisperNet: V2X-ViT [29], When2com [30], DiscoNet [31].
- Adaptive proposal counts: Guan et al. [23], added in response to comment 8 (see that item).

**Verification outcomes.** All entries were checked against primary records (publisher DOI records, CVF Open Access pages, PMLR, IEEE). Three outcomes are worth recording:

1. *Two pre-existing CVPR 2026 entries were confirmed correct as written.* PaQ-DETR [21] is published at CVPR 2026, pp. 25504–25513, and WhisperNet [28] at CVPR 2026, pp. 32154–32163; both were matched against their CVF Open Access pages (title, authors, pages). No correction was needed for either.
2. *Page numbers of three entries were corrected to DOI-registered values.* Focal Loss [10] now reads pp. 2999–3007 (the CVF listing shows 2980–2988); Resolution Adaptive Networks [14] now reads pp. 2366–2375 (CVF: 2369–2378); When2com [30] now reads pp. 4105–4114 (CVF: 4106–4115). In each case the IEEE-registered pagination is used.
3. *One entry is retained as a preprint rather than a conference paper.* No peer-reviewed venue record could be verified for Efficient DETR [7], so it is cited as arXiv:2104.01318. ICLR and NeurIPS entries carry no page numbers because those proceedings are unpaginated.

## 3. Missing contributions paragraph

**Comment:** The Introduction has no contributions paragraph.

**Response:** A twelfth paragraph has been added at the end of Section I, following the suggested shape:

> "This paper contributes three things. First, output-budget allocation is defined as the joint selection of score-ordered detection prefixes under a group budget, and utility estimation is separated from the allocation decision (Sections III-A to III-C). Second, the predictor is supervised with same-class maximum-matching increments rather than candidate matchability, so that a candidate is credited only when its addition increases the number of covered objects (Section III-B). Third, the joint selection is solved exactly by dynamic programming, and the cost of doing so is measured against a greedy solver (Sections III-D and IV-D)."

The paragraph is written in the third person, per comment 19, so "we define" appears as "output-budget allocation is defined".

## 4. M00 ablation beats M11 on Coverage

**Comment:** Table V shows M00 beating the proposed M11 on Coverage; the paper states this too quietly.

**Response:** Both suggested options were combined (Option A plus a light version of Option B). Section IV-D now states explicitly:

- that Coverage counts matches at IoU 0.50 only, while QUALITY averages ten localization thresholds;
- that M00 therefore optimizes the reported Coverage metric directly (+0.2313 over the Score baseline of 8.0228), while M11 gives +0.0935 on that metric but improves QUALITY, AP, and AR100;
- why M11 is retained as the reference configuration — the allocator is intended for interfaces where a retained record must remain useful at stricter localization tolerances (downstream association or grouping), and optimizing a single threshold makes the allocation sensitive to that threshold choice;
- that M00 is a co-equal configuration of the same predictor, available by changing the aggregation rule alone, and that an interface judged only by IoU-0.50 coverage counting should prefer M00.

The sentence "It obtains the highest QUALITY, AP, and AR100 among these four policies, while M00 retains more Coverage" has been replaced by the expanded treatment; the trade-off is no longer left to the table.

## 5. The learned estimator only pays off with the exact solver

**Comment:** The solver contributes +0.5616 Coverage, the supervision target +0.1867 — the solver is worth three times the supervision target, and the paper does not say so.

**Response:** Section V now opens with a dedicated paragraph, "Solver and target: which part carries the gain", which reports the magnitudes explicitly: with identical predictions, exact DP adds 0.5616 core Coverage over next-slot greedy, whereas the prefix-increment target adds 0.1867 over matchability under the same solver. The paragraph states both configurations that fall below the Score baseline (greedy on learned predictions: −0.4681; matchability target under the exact solver: −0.0932), draws the conclusion that neither component is sufficient alone at this budget, and explains why the greedy-versus-DP comparison is reported as a primary result: it demonstrates that the allocation decision itself, not only the quality of the learned values, is a substantive part of the contribution.

Section IV-D likewise now states plainly that the solver contributes about three times the supervision target (+0.5616 versus +0.1867) and that the failure of the greedy solver on the same predictions is direct evidence that joint prefix selection separates the proposed configuration from score-based allocation. The previous framing that implied supervision was the sole contribution has been removed.

## 6. Pre-NMS AP and AR values are never explained

**Comment:** AP of 2.04–3.69 percent will be read as failed training; the cause is stated only in fragments.

**Response:** The two suggested sentences have been added verbatim to Section IV-A, immediately after the metrics paragraph:

> "Reported AP and AR values are low in absolute terms because they are computed on the raw pre-NMS candidate stream used for allocation, which retains duplicate and low-score hypotheses. These values are comparable across methods within a comparison, because all methods share the same candidate stream, prefix bounds, and budget. They are not comparable to published AP values, which are computed after non-maximum suppression."

The captions of Tables III and IV also carry a "fixed pre-NMS Top100 candidates" label, so the explanation is reachable from the tables themselves.

## 7. The allocation overhead is reported but never justified

**Comment:** 17.8× slower, 3.519 ms/image amortized — the paper never puts the cost and the benefit in the same sentence.

**Response:** Section IV-E now adds the requested three sentences and states the limits plainly:

- the ratio: M11 at 140.76 ms per 40-image group against 7.92 ms for Score, i.e. 17.8× slower for the same output decision, or an amortized 3.519 ms/image;
- the benefit in comparable units: the core TEST gain of +0.0792 covered objects per image, which translates to approximately 44 ms of allocator time per additional covered object per image at the reported median, flagged as the only comparable-unit figure available;
- an explicit statement that no downstream measurement — end-to-end latency, tracking quality, or task success — was performed, so the trade-off cannot be expressed in application-level terms and the ratio should be read as the price of the gain rather than a deployment recommendation.

The paragraph also notes the measured cheaper variants (No prefix statistics −53.49%, Next-slot greedy −16.53% in the matched profile) and that the greedy solver's saving comes with the larger quality loss, so the cost discussion now connects to the accuracy evidence in Table V.

## 8. Related Work omits adaptive proposal-count methods

**Comment:** Add one subsection of about 150 words engaging the wider literature on deciding how many candidates to generate, with one sentence stating the difference.

**Response:** A new paragraph, "Proposal-count decisions are the closest existing problem to output-count allocation", has been added to Section II. It engages two verified works directly: He and Li [15], who estimate foreground-region counts to set the number of proposals a two-stage head processes, and Guan et al. [23], who select a per-image adaptive proposal number from proposal statistics before region-based detection (ICASSP 2019, verified against its Crossref DOI record). The difference from the present work is stated in one sentence: those methods decide how many hypotheses a detector should *produce*, which changes candidate generation and the computation spent on it, whereas this paper decides how many already-produced records to *keep* under a group-level budget, with candidate generation, candidate order, and detector computation all fixed. The paragraph also notes that the two decisions are complementary, and that the post-detector decision remains open even when a proposal-count policy is optimal. The wider dynamic-inference line is cited through the dynamic-networks survey [13] and Resolution Adaptive Networks [14].

**On the two candidate references named in the comment.** Neither could be verified as described. No Cui-authored paper on objectness-guided proposal counts and no Guan-authored paper on adaptive proposal generation (in the sense suggested) was found in Crossref, OpenAlex, or Semantic Scholar. Two closest matches were identified: (a) W. Guan, X. Zhou, G. Li, and Y. Zou, "Selecting optimal proposal number for image-based object detection," ICASSP 2019 — this is an adaptive-proposal-number method, it is peer-reviewed, and it has been added to the bibliography as [23]; (b) Y. Cui, L. Yang, and D. Liu, "Dynamic proposals for efficient object detection," arXiv:2207.05252 — closely on topic, but no published venue could be verified, so it was **not** added; it can be added as a preprint citation if the editor prefers. A paper that search engines often surface for "Cui + objectness" (ICMR 2025, "Exploring Objectness Information via Progressively Decoupled Adaptation") has no Cui among its authors and was not used.

## 9. TEST annotation exposure disclosed in a footnote

**Comment:** Move the footnote disclosure into a short appendix that states what triggered the exposure, the scope of the excerpt, the controls that prevented use, and an artefact a reader can inspect.

**Response:** The footnote has been removed and replaced by **Appendix A, "Procedural Disclosure on TEST Annotation Access"**, which answers all four questions:

1. *Trigger:* a retrieval operation issued during the identity-recovery stage that binds LC-v1 split roles returned more file content than the identity fields it was intended to read, and a truncated excerpt of TEST annotations was displayed in the process output.
2. *Scope:* recorded as an early retrieval-scope deviation; the affected non-identity fields were not used for scientific analysis, split construction, model development, or delivered exclusion identities, and the excerpt did not include allocations, candidate records, model outputs, or metric values.
3. *Controls:* the allocation path consumes no annotations at all (its inputs are candidate records, detector-native state, and image-context features from the frozen Top100 stream); the retrieval operation is not part of the selection entry point; the confirmation run re-evaluates the frozen configuration and recomputes all metrics from frozen selection outputs; no configuration, hyperparameter, seed, budget, or aggregation changed after the exposure.
4. *Inspectable artifacts:* the disposition is recorded under the label PROTOCOL_QUALIFIED_IDENTITY_RECOVERY in the data-protocol document of the LC-v1 release package (`docs/DATA_PROTOCOL_PUBLIC.md`), where the frozen upstream decision is bound by the P0 decision SHA in that release's manifest; and the frozen allocator assets are hash-bound in the project handoff package `LC_ALLOC_M11_HANDOFF_v1` v1.0.0 (`provenance/SHA256SUMS.csv`, 117 files: three seed models, temperatures, PCA, scaler, class weights, feature schema, candidate freeze, and the group manifests). Recomputing those hashes verifies that the evaluated assets are the distributed frozen assets, so the configuration behind the reported TEST metrics is identifiable independently of any internal log.

## 10. "LC" is never expanded

**Comment:** LC appears 21 times, never defined.

**Response:** Section IV-A now expands the term at first use: the primary evaluation uses the BDD100K-derived protocol adopted for this project, **referred to as LargeClean-v1 (LC-v1)**, and the text states that LC-v1 is a project-defined split and evaluation protocol rather than an official BDD100K split name, that its roles are defined on BDD100K official-train identities after a frozen exclusion list (68,074 eligible identities remaining after the 1,926-identity historical exclusion), and that it must not be confused with the official train/val/test naming. "Frozen LC" has been replaced throughout by wording such as "the LC-v1-trained allocator transferred unchanged".

## 11. "M11" is a name without a meaning

**Comment:** Define the acronym, give the method a descriptive name.

**Response:** Section III-C now defines the method descriptively and then the shorthand:

> "The configuration that combines class weighting with multi-threshold averaging is the reference method of this paper, written descriptively as class-weighted multi-threshold prefix-increment utility and labeled **M11**. The label follows the two-bit naming used for the aggregation study in Section IV-D. The first bit marks class weighting and the second marks multi-threshold averaging. M00 (neither), M10 (weighting only), and M01 (threshold averaging only) are the remaining combinations of the same predictor."

The descriptive name is used at the point of definition; M11 remains the table and text shorthand.

## 12. Equations (1), (3), (5), (6), (7), (8) are never referenced

**Comment:** Cite each equation where used, or drop the numbers.

**Response:** All eight numbered equations are now referenced in the text:

- (1): "The feasible allocations of (1) are …"
- (2): "the utility of (2) is …" (already cited)
- (3): "the target at threshold τ is" → Eq. (3), and the following paragraph refers to the target defined in (3) as one of the two supervision candidates in the controlled comparison.
- (4): already cited.
- (5): "the weights of (5) are set to …"
- (6): "The true and estimated marginal utilities of (6) are therefore …"
- (7): "define the inference objective of (7)"
- (8): "The recurrence in (8) is …"

## 13. Figure captions and multiple-run reporting

**13(a).** The Figure 1 and Figure 3 captions have each been reduced to two sentences, and both now carry "Schematic; values are prescribed for illustration." The Figure 2 and Figure 4 captions were also reduced to two sentences for consistency (Figure 4 carries no schematic marker, as it plots measured results).

**13(b) — Mean ± STD.** The tables with multiple runs now report the spread. Learned entries average three independently trained and separately evaluated models, and the ± value is the standard deviation across those three model evaluations:

- **Table I(b)** (DEV core budgets): M11 = 8.1163 ± 0.0123 Coverage, 4.8665 ± 0.0011 QUALITY, 17.03 ± 0.02 AP, 25.27 ± 0.11 AR100.
- **Table II** (COCO): core-budget spreads are stated in the text (Frozen LC 2.6389 ± 0.0032 / 3.5304 ± 0.0025; COCO-fitted 2.6530 ± 0.0007 / 3.5538 ± 0.0008).
- **Table IV** (Faster R-CNN): core spreads stated in the text (3.8233 ± 0.0029 Coverage, 2.2976 ± 0.0014 QUALITY).
- **Table V** (ablations): ± added to every learned row.

The captions state the semantics: the ± value is the standard deviation across the three independent model evaluations, and no spread is defined for the deterministic baselines (Score, Fixed-K, greedy-criterion and threshold baselines are single deterministic runs). The recomputation is reproducible from the frozen per-seed CSV artifacts with `03_reproducibility/recompute_seed_spread.py`; its output is shipped as `seed_spread_report.md` / `seed_spread.csv`.

One branch could not be completed in this revision: the YOLOv8n allocator's per-seed records are not present in the project repository (the `experiments/yolov8` directory was removed in commit 693b9d5, 10 October 2026, and the entry is a directory link with no recoverable file contents). Table III therefore reports three-model means with a caption stating that per-run values for this branch are not reproduced in the manuscript. If the arm's per-seed result table can be recovered, the same treatment can be applied to Table III.

## 14. The Abstract does not name the challenge it solves

**Comment:** Add one sentence naming what makes the problem hard.

**Response:** The Abstract now names both challenges explicitly, in three short sentences: the supervision signal must be inferred from the detector's own outputs because object annotations are unavailable when the allocation is made, and the shared budget couples the decisions across images, so prefix lengths have to be chosen jointly instead of per image.

## 15. Target journals

**Comment:** Target journals: (1) IEEE T-ITS, (2) IEEE T-IV, (3) IEEE Access, (4) Neurocomputing.

**Response:** Noted. The manuscript file is a review-stage document, not yet formatted to a specific journal template, so the ordering above can be applied at submission without further content changes. The current revision keeps the generic two-column IEEE-style layout of the original submission.

## 16. Resubmission checklist

- [x] Contributions paragraph exists and maps to sections (C1)
- [x] M11 versus M00 is justified or reframed (M1) — justified and reframed as a co-equal configuration (Section IV-D)
- [x] Solver-dominance paragraph added to Section V (M2)
- [x] Pre-NMS AP and AR explanation added to Section IV-A (M3)
- [x] Related Work covers adaptive proposal counts (M5)
- [x] TEST annotation disclosure moved to an appendix (M6)
- [x] Allocation overhead justified with the 17.8× ratio (M4)
- [x] "LC" expanded (LargeClean-v1), "M11" explained (two-bit naming + descriptive name)
- [x] Six unreferenced equations cited — all of (1)–(8) now referenced in text
- [x] "labelled" changed to "labeled" (US spelling applied consistently, including "artifact")
- [x] Four long sentences split — five in total: the M11 naming sentence (III-C), the reference-configuration sentence (IV-D), the cost trade-off sentence (IV-E), the appendix artifact sentence, and the Abstract challenge sentence
- [x] Figure 1 caption shortened and marked schematic (Figure 3 likewise; both are two sentences)

## 17. Author block

**Comment:** Author block as provided.

**Response:** The title page reproduces the supplied author block verbatim:

> Yuyi Zhu, Yang Luo*, Yi Chen**
>
> Industrial Intelligent Agent Research Center, School of Intelligent Manufacturing Ecosystem, XJTLU Entrepreneur College (Taicang), Xi'an Jiaotong-Liverpool University, Suzhou 215123, China

The asterisks were not explained in the comment. They have been interpreted as corresponding-author markers and an affiliation footnote `*Yang Luo and Yi Chen are the corresponding authors` has been added, together with a correspondence address already in use on the laboratory's pages (`Yi.Chen02@xjtlu.edu.cn`). **This interpretation needs confirmation** (see below): the markers may instead denote other roles, and the correspondence address is the only place in this package where a contact detail was supplied rather than reproduced.

## 18. Acknowledgment

**Comment:** Acknowledgment text as provided.

**Response:** An Acknowledgment section has been added before the References, reproducing the supplied text verbatim (NSFC Grant No. 42671235; XJTLU facilities and data collection contribution; NSFC research networking support).

## 19. Third-person writing

**Comment:** Use third-person writing; do not use "We", "I", etc.

**Response:** The manuscript was converted systematically to third person. Examples:

- "We measure prefix utility by …" → "Prefix utility is measured by …"
- "We scan the candidate order …" → "The candidate order is scanned …"
- "We optimize complete prefix lengths with dynamic programming" → "Complete prefix lengths are optimized with dynamic programming"
- "We denote the configuration … M11" → "the configuration … is labeled M11"
- "We use the BDD100K-derived protocol" → "The primary evaluation uses the BDD100K-derived protocol"

A final pass over the revised manuscript finds no occurrences of "we", "our", or "us" as words.

---

## Items for confirmation (four)

These are the only points where a decision or a fact is needed from the corresponding author before submission.

1. **Author block interpretation (comment 17).** The asterisks after Yang Luo and Yi Chen are rendered as corresponding-author markers, and the footnote uses the address `Yi.Chen02@xjtlu.edu.cn`. If the markers mean something else, or if the contact address is wrong, both the footnote and the address should be corrected.
2. **Reference verification (comments 2 and 8).** The bibliography now has 33 entries, each checked against a primary record. Summary of outcomes: the two pre-existing CVPR 2026 entries (PaQ-DETR [21], WhisperNet [28]) are correct as written; three page ranges were corrected to DOI-registered values (Focal Loss [10], Resolution Adaptive Networks [14], When2com [30]); Efficient DETR [7] is cited as a preprint because no peer-reviewed venue record exists. Neither candidate named in comment 8 could be verified as described; the closest verified works are Guan et al., ICASSP 2019 (added as [23]) and the Cui et al. preprint arXiv:2207.05252 (not added — say the word and it will be included as a preprint citation).
3. **YOLOv8n per-seed spread (comment 13b).** The per-seed records for the YOLO arm are not available in the current repository (removed in commit 693b9d5). Table III reports three-model means with an explicit caption note. If the arm's frozen per-seed table can be provided, Table III will receive the same ± treatment as Tables I(b), II, IV, and V.
4. **Appendix A artifact identifiers (comment 9).** The appendix now names two concrete artifacts: the public data-protocol document in the LC-v1 release package and the handoff package `LC_ALLOC_M11_HANDOFF_v1` v1.0.0 with its SHA256SUMS provenance. If the submission process requires the exact release manifest file name and its P0 decision SHA to appear in the manuscript text, that string should be inserted in the appendix before submission.

## Reproducibility note

`03_reproducibility/recompute_seed_spread.py` recomputes every mean and standard deviation reported in Tables I(b), II (core budgets), IV, and V from the frozen per-seed result CSVs of each arm. The script trains nothing and allocates nothing; it reads the frozen artifacts and reproduces the reported values. Its output (`seed_spread_report.md`, `seed_spread.csv`) is included so the ± figures in the manuscript can be checked line by line.
