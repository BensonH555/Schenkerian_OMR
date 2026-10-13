# Schenkerian OMR

Tools and annotations for recognizing handwritten Schenkerian analysis and representing its symbols, connections, and temporal structure as Music Notation Graphs (MuNG).

## Browse the annotations

Start with the **[current annotation index](Complete%20Annotations/current/README.md)** to inspect page images, MuNG XML, completion status, and examples of Schenkerian notation. For a description of the fields and symbol conventions that extend the standard MuNG representation, see **[MuNG extensions](docs/mung_extensions.md)**.

The snapshot captured on **2026-10-13 at 04:01 UTC** includes **all 87 live pages**, including unfinished annotations. The index shows recognition, notation assembly, and temporal status separately for every page. **41 pages have recognition recorded as `done`**, and **one has all three stages explicitly marked `done`: Bach Fugue in Bb minor WTC 2 Overview001**. It is a useful starting example for reviewing the complete annotation structure.

Of the 41 recognition-complete pages, 25 have an explicit recognition-stage status and 16 inherit the older page-level `done` status. The other 46 pages have recognition recorded as `not-started` (37), `pending-review` (6), or `in-progress` (3). Missing later-stage records are not evidence of completion.

- [Current annotation index](Complete%20Annotations/current/README.md): the organized snapshot for inspection and schema review.
- [MuNG extensions](docs/mung_extensions.md): representation details and examples.
- [Annotation guide](Complete%20Annotations/README.md): collection contents and how to interpret workflow status.
- [MuNGStudio](https://mung-studio-schenker.com/): the live annotation workspace. This repository snapshot is not automatically synchronized with later edits.

## Annotation workflow

The current workflow supports recognition directly on original page images, followed by human review. It tracks three independent annotation stages:

| Stage | What is reviewed |
| --- | --- |
| Recognition | Symbol classes, locations, missing symbols, and duplicate detections. |
| Notation assembly | Connections and attachment relationships between symbols. |
| Temporal structure | Notehead groups at the same time position and their ordering relationships. |

The source annotations are MuNG XML with Schenkerian symbol classes and additional metadata, including temporal grouping information. Workflow status is recorded separately. A stage marked `done` records annotation-workflow completion; it is not a claim that every derived pitch, voice assignment, structural depth, or conversion output has been independently verified.

## Scope of this update

This update publishes the full live page snapshot, annotation statuses, and format documentation. It does not synchronize all currently deployed application code, conversion code, or model implementations. Historical scripts, configurations, and model files already present in this repository remain available, but they should not be taken as the current production implementation or evidence of current model performance.

The previous README is preserved as the **[historical pipeline guide](docs/legacy_pipeline.md)** for readers using the earlier experiments.
