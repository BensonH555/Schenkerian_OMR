# Snapshot validation

The snapshot captured on 2026-10-13 at 04:01:20 UTC contains all 87 live pages, including unfinished annotations and their current workflow statuses. Only one page has all three workflow stages explicitly marked done. Validation checks file integrity and annotation structure; it does not certify the musical interpretation or complete the remaining annotation stages.

From the repository root, run the validator with Python 3.10 or later. It requires no third-party packages:

```sh
python3 scripts/annotations/validate_snapshot.py "Complete Annotations/current"
```

Add `--report validation.json` to save the JSON result. A successful run returns exit code 0, with source quality issues listed as warnings. Missing/corrupted files, malformed data, invalid masks other than the documented `0:0` placeholder, count mismatches, missing workflow fields, and invalid graph references return a nonzero exit code. The script never changes source annotations.

## Checks and results

- Verify byte lengths and SHA-256 hashes of all 348 files listed in the manifest.
- Read PNG/JPEG image dimensions and compare them with the manifest and paired original image.
- Parse all XML documents and check node ID uniqueness, required geometry, typed `DataItem` values, and valid binary masks of the expected dimensions.
- Check syntax and precedence graphs for duplicate references, dangling references, and reciprocity between incoming and outgoing links.
- Parse time-group JSON, check its version and kind, and check membership and metadata consistency.
- Recompute page and collection counts and compare workflow sidecars with the manifest.

The collection contains **56,448 symbols**, **13,736 noteheads**, **136 observed class values** (135 nonempty values plus one empty value), **17,501 directed syntax edges**, **2,716 directed precedence edges**, and **855 time groups**. There are **1,976 grouped symbols**, of which **1,975 are noteheads**. Of **7,135 stored RLE masks**, **7,101** have valid binary runs and dimensions and **34** contain the zero-length `0:0` placeholder. Another **49,313 nodes omit a mask**.

The validator returns **zero errors and 42 source quality warnings**; see the [machine-readable validation report](validation.json). No malformed XML, duplicate IDs/links/data keys, dangling or nonreciprocal graph references, invalid typed metadata, inconsistent group metadata, or multiple time-group memberships were found. Apart from the documented zero-length placeholders, mask dimensions and values are valid. Canvas and original-image dimensions agree on every page.

## Preserved source quality issues

These 42 warnings already exist in the source annotations: five boxes extend outside the canvas, one non-notehead has time-group membership, two nodes have unassigned classes, and 34 masks are zero-length placeholders. The snapshot preserves them so that the published XML remains byte-identical to the captured source.

| Document | Node ID | Finding |
| --- | ---: | --- |
| [Bach St Matthew Passion Part II Sketches004](pages/Bach-St-Matthew-Passion-Part-II-Sketches004/) | 540 | `flagStructuralUp` belongs to time group `95abe151-2039-4a51-9953-14a651d5bbb3`, alongside noteheads 621 and 833. Time groups are intended for noteheads; this membership needs review. |
| [Rachmaninov The Bells Second Mvmt 005](pages/Rachmaninov-The-Bells-Second-Mvmt-005/) | 583 | `slurStructuralDown` has left coordinate −2 pixels, outside the canvas. |
| [Schubert_Einsamkeit_Graphs001](pages/Schubert_Einsamkeit_Graphs001/) | 329 | `otherText` has top coordinate −19 pixels, outside the canvas. |
| [Schubert_Einsamkeit_Graphs003](pages/Schubert_Einsamkeit_Graphs003/) | 8 | `otherText` has top coordinate −1 pixel, outside the canvas. |
| [Schubert_Einsamkeit_Graphs003](pages/Schubert_Einsamkeit_Graphs003/) | 543 | `characterCapitalB` has left coordinate −1 pixel, outside the canvas. |
| [WTCII F-sharp Fugue p3](pages/WTCII-F-sharp-Fugue-p3/) | 337 | `voiceExchangeDown` extends one pixel beyond the right edge: left 2378 + width 321 = 2699, versus canvas width 2698. |
| [Schubert_Einsamkeit_Graphs006](pages/Schubert_Einsamkeit_Graphs006/) | 632 | Present but empty `ClassName` element; the symbol remains in the snapshot with an unassigned class. |
| [WTCI C-sharp Fugue](pages/WTCI-C-sharp-Fugue/) | 550 | Present but empty `ClassName` element; the symbol remains in the snapshot with an unassigned class. |

In [Schubert_Einsamkeit_Graphs011](pages/Schubert_Einsamkeit_Graphs011/), the following 34 nodes store the literal mask `0:0`, which does not provide a full pixel mask for the bounding box: **0, 2, 9, 23, 25, 31, 42, 44, 51, 58, 60, 72, 86, 205, 291, 543, 551, 552, 569, 570, 571, 576, 582, 607, 610, 612, 639, 651, 682, 714, 723, 754, 764, 775**. The application parser accepts these values; the validator reports them as source quality warnings rather than changing or dropping them. Other invalid mask encodings remain errors.

## MuNG parser compatibility

As a separate compatibility check, the application's vendored `mung.io.read_nodes_from_file` loaded all 87 files and all 56,448 nodes without modifying them. This check used Python 3.12.13, NumPy 1.26.2, lxml 4.9.3, and scikit-image 0.22.0. The public validator above is independent of that parser.

The MuNG parser supports omitted masks as bounding-box-only annotations. However, the historical bundled `CVC-MUSCIMA_Schema.xsd` requires a `Mask` element. The XML's schema-location attribute therefore must not be interpreted as proof of strict validation against that historical XSD.
