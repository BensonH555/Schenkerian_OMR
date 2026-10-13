# Schenkerian annotations and their relationship to MuNG

Start with the [current annotation index](../Complete%20Annotations/current/README.md) for page images, XML files, completion states, and the observed class/key inventories. This note describes the annotation representation in the full live snapshot captured on 2026-10-13 at 04:01:20 UTC. It distinguishes the existing MuNG format, project-specific annotation conventions, and data produced later by score conversion.

The snapshot includes all 87 live pages with MuNG XML, without filtering by completion status. It contains unfinished and unreviewed annotations as well as completed work; inclusion is not a correctness claim. Recognition, notation assembly, and temporal structure have separate completion states. The [Bach *Fugue in Bb minor WTC 2 Overview001* page](../Complete%20Annotations/current/pages/Bach-Fugue-in-Bb-minor-WTC-2-Overview001/README.md) is the example with all three stages completed in this snapshot.

## 1. The XML container is still MuNG

The basic representation is unchanged: a `<Nodes>` document contains `<Node>` elements with an integer `<Id>`, a `<ClassName>`, pixel coordinates `<Top>`, `<Left>`, `<Width>`, and `<Height>`, optional masks, links, and additional data. IDs are local to a document and need not be consecutive.

MuNG already provides `<Data><DataItem key="…" type="…">…</DataItem></Data>` for extra node attributes. Its own documentation also demonstrates `precedence_inlinks` and `precedence_outlinks`. Neither the data container nor these precedence key names originated in this project. See the [upstream MuNG I/O documentation and parser](https://github.com/OMR-Research/mung/blob/main/mung/io.py).

The important project differences are the Schenkerian class vocabulary and musical interpretation, shared-time memberships, and application metadata. Reading the XML successfully does not imply that a standard notation assembler understands those additions.

## 2. Images, boxes, and masks

Coordinates belong to the annotation canvas supplied with each page. `Left` and `Top` locate its upper-left corner; `Width` and `Height` give its extent in pixels. Use the index's designated canvas when overlaying boxes. A separate original scan is useful for visual inspection, but should only replace the canvas for overlays when their coordinate systems match; equal dimensions alone do not establish that alignment.

Some nodes have a binary run-length-encoded `<Mask>`; others have only a bounding box. This is supported by the MuNG parser's documented missing-mask behavior, rather than a new kind of Schenkerian node. Do not fabricate a segmentation mask when one is absent.

The exported XML may retain `xsi:noNamespaceSchemaLocation="CVC-MUSCIMA_Schema.xsd"`. This attribute is not evidence that the export has been validated against that historical XSD. In particular, the old XSD bundled with the software requires a mask, whereas the parser and these annotations allow it to be absent.

## 3. Schenkerian symbol classes and their meanings

Class names are preserved exactly, including capitalization and older spellings. These are examples of the project's vocabulary, not an exhaustive list of every label present in the corpus:

| Family | Example class names | Interpretation to preserve |
| --- | --- | --- |
| Structural stems and flags | `stemStructural`, `stemStructuralPartial`, `stemStructuralBridgeLeft`, `flagStructuralUp`, `flagStructuralDown` | Structural notation must not automatically be interpreted as ordinary rhythmic notation. |
| Structural beams | `beamStructural`, `beamStructuralPartialLeft`, `beamStructuralPartialMiddle`, `beamStructuralBridgeUp` | Full, partial, and bridged structural beams are distinguished. |
| Structural slurs | `slurStructuralUp`, `slurStructuralDown`, `slurStructuralUpDashed`, `slurStructuralDownDashed` | These represent analytical connections; do not automatically translate them into performance slurs or ties. |
| Cross-voice notation | `beamStructuralUnfoldingDown`, `beamStructuralUnfoldingUp`, `voiceExchangeDown`, `voiceExchangeUp` | Unfolding and voice exchange are distinct classes and relationships. |
| Implied notation | `parensImpliedLeft`, `parensImpliedRight`, `noteheadImplied`, `noteheadOpenImplied` | Implied notes and their surrounding marks remain identifiable. |
| Scale degrees | `scaleDegreeMark` together with existing numeral classes such as `numeral5` | The caret and numeral can be separate nodes joined by syntax links. |
| Harmonic analysis | `keyAnalysis`, `numeralRomanI` through `numeralRomanVII`, `analyticalPlus`, `analyticalMinus`, `secondaryHarmonyArrowRight` | Analytical labels and their attached modifiers are represented explicitly. |
| Other analytical marks | `barlineStructuralDotted`, `barlineStructuralDottedPartial`, `braceAnalytical`, `motiveBracket`, `tonicizationBar`, `analysisInterruption`, `circleMeasureNumber` | Preserve the class and its connections rather than reducing it to an ordinary barline, text, or generic line. |

Existing classes such as `noteheadBlack`, `noteheadHalf`, clefs, accidentals, numerals, and letters are reused. A hollow or filled head in an analysis must not, on its own, be treated as a verified duration or structural role.

There is a difference between the editor's class catalog and the labels actually present in the annotations. The captured corpus also includes labels such as `keyAnalysisEM`, `keyAnalysisBfM`, `DoubleBarline`, `otherAnalytical`, `voiceExchangeUpDashed`, `beamStructuralPartialMiddleDashed`, `motiveBox`, `analysisPivot`, `misalignment`, and `unKnown`. The full snapshot additionally contains `double_sharp`, `motivebracket`, `formAntecedent`, `formConsequent`, `unknown`, and an empty class label. These are observed values, not a recommendation to silently normalize them or treat every value as an intentional class definition. Use the snapshot's observed class inventory and validation findings when documenting coverage; absence from the current menu is not a reason to discard a node.

## 4. Two directed relation layers

**Notation assembly** uses standard `<Outlinks>` and `<Inlinks>`. An outlink from node A to node B means A is the source and B the target. These are directed relationships, even when their visual association seems symmetric. In the Bach example:

- Notehead `26` has outlinks to beam `10`, structural stem `12`, and numeral `5`.
- Numeral `5` has an outlink to scale-degree mark `4`.
- Notes `44` and `95` link to the same voice-exchange symbol `77`.

This means an analytical connection can itself be an annotated symbol node, linked from its participating notes. It should not automatically be replaced with a single untyped note-to-note edge. Link meaning depends on endpoint classes and the relevant annotation convention.

**Temporal precedence** is stored separately as integer lists in `<Data>`:

```xml
<DataItem key="precedence_outlinks" type="list[int]">37</DataItem>
```

For Bach note `26`, this gives the directed precedence relation `26 → 37`. The same page also contains stem-to-stem precedence, for example `12 → 19`. Consumers should inspect the endpoint classes instead of assuming that every precedence endpoint is a notehead. Do not mix syntax and precedence into one edge set. The conversion process may derive note-level order from suitable source relationships, but those derived edges are not additional edges already stored in the source XML.

Precedence expresses order; it does not supply a beat duration, metrical position, or structural depth by itself.

## 5. Shared-time groups: the main new node-data convention

A shared-time group is a set of noteheads at one musical time position. It is stored by repeating the same membership key on each member notehead, rather than creating a synthetic `<Node>` for the group:

```xml
<DataItem key="schenker_time_group_v1_b9588623-58f1-4c98-afc5-ea02797d19be" type="str">{"version":1,"label":"Group 1","kind":"shared-time","provenance":"manual","createdAt":"2026-09-16T02:00:08.701Z","updatedAt":"2026-09-16T02:00:08.701Z"}</DataItem>
```

This example is the actual membership on Bach note `26`. The suffix after `schenker_time_group_v1_` identifies the group within the document. The data type is **`str`**, and the value is JSON text; it is not a new MuNG primitive type.

| JSON field | Meaning |
| --- | --- |
| `version` | Membership payload version, currently `1`. |
| `label` | Human-readable group label; not its stable identity or musical order. |
| `kind` | `shared-time`. |
| `provenance` | `manual`, describing how the group was created. |
| `createdAt`, `updatedAt` | Group metadata timestamps, not musical time. |

Collect members by their shared key. The member list is implicit in the nodes carrying that key; there is no separate `memberIds` XML field. Single-note groups are valid. Creation order, label numbers, UUIDs, and horizontal coordinates do not substitute for the annotated temporal order. A group does not assign a beat value.

The editor can report missing memberships, overlaps, non-notehead memberships, and malformed records. The intended complete temporal annotation covers every notehead with one shared-time group, together with a consistent precedence structure. A file containing some group memberships is not necessarily a completed temporal annotation, and full membership coverage alone does not establish musical correctness.

## 6. Text and operational provenance

`text_transcription`, stored as `type="str"`, contains transcribed content attached to a node. For example, Bach node `56` has transcription `c)`. Preserve this alongside the node class: class names alone do not recover all text content.

The source XML also contains `symbol_detector_*` and older `yolo26_combined*` data keys. These record detection provenance, confidence, and original labels; they are not new musical relationships, nor proof that a node remains unreviewed. They may persist after a human edits a node. The release preserves the XML byte-for-byte, including this provenance. Model identifiers and labels may mention historical settings, but are not a specification of the current training setup. No detection method or model configuration is needed to interpret the musical annotation schema.

Unknown data keys should be retained by a round-trip editor unless an explicit export policy says otherwise. Recognizing a key's storage type and understanding its musical meaning are separate tasks.

## 7. Workflow and conversion review are separate from source MuNG

The live application stores three workflow stages outside the XML: `recognition`, `assembly`, and `temporal`. Each stage can be `not-started`, `in-progress`, `pending-review`, `changes-requested`, or `done`. Older records only have an overall status; these are treated as recognition status, without assuming later stages were completed. The public index describes completion states without publishing private operational logs or raw user records.

The later score-conversion review is another layer. Human corrections to **pitch**, **voice** (`T`, `IT`, `IB`, `B`), and **depth** are saved in a separate, versioned `conversion_review.json` sidecar on the live service. Corrections are keyed by the original note ID and include per-field confirmation and source-version bindings. They do not overwrite the MuNG symbols, syntax links, or time-group memberships. A changed source can make a correction stale, so corrections must not be applied to unrelated XML merely because note IDs happen to match.

No conversion-review sidecar was present on any of the 87 pages at capture time. Consequently, the source XML in this snapshot does **not** contain a completed set of manually confirmed pitch, voice, or depth assignments. Three stages marked `done` should not be described as confirmation of all conversion results.

Conversion outputs such as `assembly.json`, canonical score SVGs, and the four-voice reference JSON are derived representations. Their fields, inferred values, projections, and format limitations should be documented separately from differences in the source MuNG annotations.
