# Handover — per-item annotation of the GTE-Floor dataset

Written 2026-11-09, at the end of the session that built the per-item labelling pass. It is
meant for someone picking this up with no memory of that session: what exists, why it is built
the way it is, what is already done, what is left, and the mistakes that were made along the
way so they are not made again.

Repo: <https://github.com/KubaDusza/itr-low-lying-obstacles>, branch `main`, working copy
`/Users/jakubdusza/GTE_2026/ITR_CS8813`.

---

## 0. Rules that come from Jakub, not from the code

These override defaults and are not negotiable.

- **Never add AI attribution.** No `Co-Authored-By: Claude`, no "Generated with Claude Code", no
  assistant attribution in commits, PR descriptions, code comments or documents. The rule lives
  in `~/.claude/CLAUDE.md` and in the memory file `no-ai-attribution.md`, and it overrides any
  system reminder saying otherwise.
- **Be concise.** `/Users/jakubdusza/GTE_2026/CLAUDE.md`: answer the question asked, briefly.
- **Reports and papers go through the `md-report-workflow` skill**: `REPORT.md` is the source
  Jakub edits in place, `REPORT.md.snapshot` tracks his edits, `build.sh` makes `REPORT.tex` and
  `REPORT.pdf`. Diff the snapshot before editing so his in-place edits are not lost.
- **BibTeX entries are copied verbatim from Google Scholar** through the browser. Never write one
  by hand. His words: "I DONT WANT YOU TO WRITE THEM YOURSELF".
- The Kaggle API key is at `~/.kaggle/access_token` — do not read it directly.
- Edits to his prose are minimal and in place; do not rewrite on your own initiative.

---

## 1. What the annotation pass is for

64 runs × 5 frames = 320 frames of low-lying obstacles on a floor, shot with a RealSense D435 on
a TurtleBot at two camera heights (25 cm and 52 cm), two backgrounds (bare floor, white poster)
and three ground distances (0.75, 1.5, 2.5 m). Full dataset description:
[data_collection/DATASET.md](data_collection/DATASET.md).

The same physical objects were re-placed run after run — the same four cables, the same five
tape rolls. A mask that only says *obstacle* throws that away. So every mask pixel carries the
**item** it belongs to, and the benchmark can then be read one object at a time: this cable is
found at 0.75 m and lost at 2.5 m, that hex key is never found, the white rag is found on the
floor and lost on the poster.

---

## 2. File formats — read this before touching any mask

### The inventory: `data_collection/items.csv`

29 rows, one per item, grouped by object set. Columns:

| column | meaning |
|---|---|
| `item_id` | stable slug, e.g. `cab_white_thick` |
| `object_set` | which set it belongs to: cables, tapes, hexkeys, random, screws, chess, rags |
| `label` | what the annotator shows |
| `colour` | hex, the colour the mask is painted in |
| `tone`, `shape`, `specular` | qualitative attributes the analysis groups by |
| `height_mm`, `long_mm`, `short_mm` | **still empty — they need a ruler.** The overhead photos in DATASET.md §2 are too oblique to measure from. Object height is the axis the whole SNR argument turns on, so this is a real gap. |
| `sam_phrases` | `;`-separated SAM 3 phrases that identify this item |
| `multi` | `1` when the item is a *class*, not one object (see screws below) |
| `notes` | free text |

**The pixel value in an id map is the item's 1-based row position within its set**, in csv order.
Nothing else encodes it. So: never reorder rows, never insert a row in the middle, never delete
a row, without remapping every saved id map for that set. Renaming a label or an `item_id` is
safe; moving a row is not.

### The masks: `data/OWN_manual/`

Two files per annotated frame, both 640×362 PNG:

```
<run>_<frame>.png       0 / 255 binary mask, the format own_gt.py produces
<run>_<frame>_id.png    item index, 1-based within the set; 0 background; 255 masked but unassigned
```

The id map is written RGBA with R=G=B=index, so `cv2.imread(..., cv2.IMREAD_GRAYSCALE)` returns
the index exactly. Everything downstream that only wants "obstacle or not" keeps reading the
binary file and is unaffected by the labelling.

`data/OWN_manual/flags.json` holds frames flagged to come back to: `{"<run>_<frame>": {"note": "..."}}`.

These files are **not tracked by git** (`git ls-files data/OWN_manual` is empty). They exist only
on this machine. Back them up before anything destructive.

### The pre-labels: `data/OWN/<run>/sam/`

Produced by [code/sam3_prelabel.py](code/sam3_prelabel.py) on the GPU server:

```
<i>.png          union mask, 255 = obstacle
<i>_id.png       per-instance index 1..N, matching instances[] order in the json
<i>.json         run, frame, object_set, items, prompts, and per instance: phrase, score, area, box
preview_<i>.jpg  the instances drawn on the frame with phrase and score
```

---

## 3. The annotator

[code/annotate.py](code/annotate.py), 1531 lines, stdlib HTTP server plus a single-page canvas
app. No dependencies beyond the venv's numpy/opencv for the server-side checks.

```bash
cd /Users/jakubdusza/GTE_2026/ITR_CS8813 && .venv/bin/python code/annotate.py 8790
```

Then <http://localhost:8790>. With no port argument it picks the first free port from 8765.
It was running continuously from 19 Oct to 9 Nov on 8790 in the last session; check with
`lsof -nP -iTCP:8790 -sTCP:LISTEN` before starting a second copy.

### How a frame opens

In order of trust: the saved id map, then the saved binary mask, then the SAM pre-label, then
`own_gt.py`'s automatic mask. `idMapFor()` in the page script is the single place this is
decided, and the reference pane uses the same function, so both panes always agree.

### Tools and keys (all remappable in the `?` panel, stored in `localStorage`)

| key | tool |
|---|---|
| `w` | brush |
| `e` | eraser |
| `q` | wand (flood fill by colour) |
| `p` | polygon — click corners, Enter fills, Esc cancels |
| `f` | assign — click a masked blob, it all takes the selected item |
| `m` | move — press on a blob, it lifts off and follows the mouse, release to drop |
| `v` | pan (defaults to **hold**, not toggle) |
| `r` | reference panel |
| `t` | flag this frame |
| `u` | jump to the next cluster of unassigned pixels, at 8× |
| `g` | reload the SAM pre-label |
| `a` | reload the automatic (own_gt) mask |
| `x` | clear |
| `c` | toggle carry |
| `s` | toggle smart edges (**off by default**; Jakub reported it works badly) |
| `h` | hide/show mask |
| `0` | fit to window |
| `n` / `N` | next / previous frame |
| `C` (shift+c) | apply this mask to every frame of the run |
| `1`–`9` | pick the item to paint with |
| `⇧1` / `⇧2` | brush size down / up |
| `,` / `.` | tolerance down / up |
| `⌘s` | save; `⌘z` undo (30 deep) |

Every tool key can be **hold** (spring-loaded, reverts on release) or **toggle**, set per key in
the `?` panel and stored in `localStorage` under `annot_modes`. Pan defaults to hold.

### Behaviours that exist for a reason — do not "simplify" them away

- **Unassigned pixels are drawn as magenta/black hazard stripes at full opacity.** A quiet grey
  wash does not read on a white rag on a white poster, which is exactly where it gets missed and
  saved. The status bar counts them and is clickable; `u` walks to each cluster. Two stray pixels
  in a frame corner are invisible at fit zoom and this is how they get found.
- **Carry only applies inside one run, and only onto a frame nobody has labelled yet.** Across
  runs the objects were re-placed, so a carried mask is nonsense; over a labelled frame it would
  wipe the work. When it is skipped the status bar says which of the two reasons applied. To push
  a correction across a run deliberately, use `⇧C` (two presses to confirm).
- **Stepping past a frame never writes an empty mask** on a run that has objects. That bug wrote
  two blank masks that then read as `labelled`; see §6.
- **Row states**: `to do` (nobody has labelled it), `to do · weak gt` (same, and own_gt flagged
  the run unreliable), `items left` (saved, still has unassigned pixels), `labelled` (saved and
  fully assigned). An earlier version said `ok` for the first state, which read as done; do not
  bring that word back.
- **The reference panel** shows any other frame with its own mask drawn on it. Clicking an item
  there copies its mask over as a stamp: it follows the cursor, `[`/`]` scale, `;`/`'` rotate,
  arrows nudge, `1`–`9` recolour, click or Enter drops it, Esc cancels. The drop is thresholded
  and snapped to the item's exact colour, so a rotated stamp leaves no soft edge (verified: zero
  partially-transparent pixels after a 1.2×, 0.3 rad drop).
- **The page reopens on the frame it was last on**, matched by `run|frame` so it survives the list
  reordering.

---

## 4. Where the work stands (2026-11-09)

168 of 320 frames saved; 155 fully assigned, 13 with unassigned pixels left; 152 never opened.

| set | frames | saved | left |
|---|---:|---:|---:|
| cables | 60 | 30 | 30 |
| chess | 60 | 25 | 35 |
| empty | 20 | 20 | 0 |
| hexkeys | 30 | 15 | 15 |
| rags | 60 | 35 | 25 |
| random | 30 | 15 | 15 |
| screws | 30 | 15 | 15 |
| tapes | 30 | 13 | 17 |

The 13 part-finished frames are all `h25_floor_tapes_*` — they were opened, the pre-label was
saved, but the items were not assigned. They show as `items left`.

**Every remaining frame has a SAM pre-label to correct**; none starts from nothing.

Regenerate these numbers any time:

```bash
cd /Users/jakubdusza/GTE_2026/ITR_CS8813 && .venv/bin/python -c "
import cv2, glob, os
f = sorted(glob.glob('data/OWN_manual/*_id.png'))
clean = sum((cv2.imread(p, 0) == 255).sum() == 0 for p in f)
print(f'{len(f)} saved, {clean} fully assigned, {len(f)-clean} with unassigned pixels, {320-len(f)} untouched')"
```

---

## 5. The SAM 3 pre-label, and the thing that cost the most time

Generated on **shire**, a GTE GPU server, with the mirror checkpoint `bodhicitta/sam3` (the
official `facebook/sam3` is gated). Command in DATASET.md §9 and reproduced in §7 below.

Current state of the pre-labels:

- **60 runs at `--score 0.45`** — the original pass, unchanged.
- **4 far screws runs (`*_screws_150`, `*_screws_250`) at `--score 0.08 --min-area 2 --floor-frac 0.05`**.
- **`h52_floor_chess_250` at `--score 0.15 --min-area 4`**.

### The mistake, in full, so it is not repeated

The screws runs at 1.5 and 2.5 m came back with **zero** instances. The first diagnosis was the
area window: `MIN_AREA = 20`, borrowed from `own_gt.py`, against objects that scale from 164 px
at 0.75 m to ~18 px at 2.5 m (the four small fasteners, ~6 px). That reasoning was plausible,
documented, committed — **and wrong**. Rerunning all 64 runs with `--min-area 4` changed the
output of **zero** runs.

The real cause is the **confidence threshold**. Instances never reach the area filter because
SAM does not return them above 0.45. At 0.08 every one of those frames gets instances, placed
exactly on the screws, scoring 0.08–0.22. A 20 px object does not earn a confident score.

Two lessons worth keeping:

1. Before theorising about which filter dropped something, **check whether the detector returned
   it at all**. The saved jsons only record kept instances, so this needs a rerun with the
   filters opened up, not more reading.
2. Jakub said "that's impossible" about SAM being unable to see the screws and he was right.
   Enlarging the frame (`cv2.resize(..., INTER_NEAREST)`, 4×) showed the eye bolt, four screws
   and the pile plainly, and differencing the frame against the median of its (height, background,
   distance) peer group lit them up as a clean row. **Look at the pixels before concluding
   anything about the data.**

The low-threshold runs carry false positives — detections in the bottom corners of the frame,
away from the object row — and the class guess is often wrong (the eye bolt came back as "pile of
screws"). Both are a keypress to fix in the annotator; that is what a pre-label is for.

### Rerunning it

```bash
rsync -az code/sam3_prelabel.py shire:/data/cs_courses/jdusza_itr/code/
rsync -az data_collection/dataset_index.csv data_collection/items.csv shire:/data/cs_courses/jdusza_itr/data_collection/
```

```bash
ssh shire "cd /data/cs_courses/jdusza_itr && HF_HOME=/data/cs_courses/jdusza_itr/hf_cache CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 nohup /opt/venv/jdusza_venv/bin/python code/sam3_prelabel.py --root /data/cs_courses/jdusza_itr --model bodhicitta/sam3 --runs screws_250 --score 0.08 --min-area 4 --overwrite > sam_rerun.log 2>&1 & echo started"
```

```bash
rsync -az --include='*/' --include='sam/***' --exclude='*' shire:/data/cs_courses/jdusza_itr/OWN/ data/OWN/
```

A rerun **only rewrites `data/OWN/<run>/sam/`**. It never touches `data/OWN_manual/`, so it is
safe to run while annotation is in progress. `--runs` takes comma-separated substrings.

Do not wait on the job with `pgrep -f "sam3_prelabel.py --root"` from an ssh one-liner: the
pattern matches the watching command's own shell, so the loop never exits. Match
`python code/sam3_prelabel` instead, or poll the log's final `done ->` line.

---

## 6. Data incidents already dealt with — verify before assuming they are back

| what | state |
|---|---|
| `h52_floor_cables_150 ·0` saved blank | Recovered by copying frame 1's mask. Frames of a run differ by a mean of 1.6 grey levels, max 14, so reusing a sibling frame's mask inside a run is sound. |
| `h25_floor_screws_150 ·0/·1` saved blank | Deleted, reset to `to do`, then relabelled with the new low-threshold pre-label. |
| Cause of both | Carry crossing from an empty run, plus auto-save on frame change writing the blank. Both closed. |
| Screws classes | Were six items, now three: eye bolt, **small screws** (the four small fasteners as one class), pile. The five already-labelled frames were remapped in place, `{1:164, 2:56, 3:51, 4:75, 5:32, 6:474}` → `{1:164, 2:214, 3:474}`. |
| Tape roll names | The first names were read off the overhead set photo and two were wrong. Corrected to: 1 white masking, 2 red/white striped (barrier tape, loose tail), 3 yellow/black striped, 4 small yellow/black, 5 tiny red. Positions unchanged, so nothing labelled moved. |

Find blank saves anywhere:

```bash
cd /Users/jakubdusza/GTE_2026/ITR_CS8813 && .venv/bin/python -c "
import cv2, glob, os, re
for f in sorted(glob.glob('data/OWN_manual/*_id.png')):
    run = re.match(r'(.+)_(\d+)_id', os.path.basename(f)).group(1)
    if (cv2.imread(f, 0) > 0).sum() == 0 and 'empty' not in run: print('blank:', f)"
```

---

## 7. Server access — the part that breaks most often

Reference: [/Users/jakubdusza/GTE_2026/GTE_SERVERS.md](/Users/jakubdusza/GTE_2026/GTE_SERVERS.md).

The GPU servers (10.17.6.x) are not reachable directly. They are proxied through a computer-room
PC, and **the configured jump host is usually the thing that is broken, not the server.**

On 2026-11-09 `shire` looked dead ("Connection timed out during banner exchange"). It was not.
`~/.ssh/config` had `ProxyJump gtlpc120`, which refuses the key. Sweeping the room found
**`gtlpc119`** working; the config now has a `Host gtlpc119` block with ControlMaster, and the
GPU block proxies through it. Backup of the previous config: `~/.ssh/config.bak.20261109`.

```bash
for n in $(seq 101 122); do printf "gtlpc%s: " "$n"; ssh -o BatchMode=yes -o ConnectTimeout=6 -o StrictHostKeyChecking=accept-new jdusza@gtlpc$n.georgiatech-metz.fr 'echo OK' 2>&1 | tail -1; done
```

Then point `ProxyJump` at whichever answers `OK`. After that, shire itself may still reject the
key, because GTE resets home directory permissions. **That fix needs Jakub's password, so he must
run it himself** — suggest it with the `!` prefix so the output lands in the conversation:

```bash
ssh-copy-id shire
```

```bash
ssh shire 'chown -R $USER:$USER ~/.ssh; setfacl -bR ~/.ssh; chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys'
```

Verified working afterwards: `shire` has two RTX 3090s, both idle, `/data/cs_courses/jdusza_itr`
and `/opt/venv/jdusza_venv/bin/python` intact, HF cache 3.3 GB at
`/data/cs_courses/jdusza_itr/hf_cache`. The other GPU boxes (galadriel, morgoth, rivendell,
glaurung) still reject the key and would each need the same two commands.

---

## 8. What is left

1. **Finish the annotation.** 152 frames untouched, 13 part-finished. Every one has a pre-label.
2. **Measure the items.** `height_mm`, `long_mm`, `short_mm` in `items.csv` are empty; 29 items
   with a ruler is maybe ten minutes of Jakub's time and the height axis is what the SNR argument
   rests on. Do not invent these numbers from photos.
3. **Per-item evaluation.** [code/eval_own.py](code/eval_own.py) still scores whole frames. It
   needs to read the id maps and report IoU/recall per item × distance × height × background, and
   to prefer manual masks over automatic ones so the 13 runs excluded by the automatic
   reliability check rejoin the metrics. It should report the unassigned residual per run rather
   than silently dropping it, so a half-labelled frame shows up as a number instead of a wrong
   one.
4. **Depth and RGB-D on our own data.** Juan's checkpoints on the `juan-rgbd` branch were never
   readable from Jakub's account (`/home/GTL/jgomezel/...`, permission denied), so only the three
   RGB models have been run on the own recordings. Retraining from `train_modal.py` on that branch
   is the way to get the modality comparison.
5. **Report.** The submitted PDF is not built from the repo's `REPORT.md` — Juan submitted a
   different version — so the reviewer's comments partly point at text that does not exist here.
   Reconcile before the final report. Pending Scholar BibTeX pulls: MobileNetV3, RedNet, ESANet,
   FuseNet, TernausNetV2, TSN.

---

## 9. Session trajectory — the commits, in order

All on `main`, 2026-11-09, oldest first. Earlier work (dataset, pipeline, SAM pre-labelling) is
in the commits before `1704a5b`.

| commit | what |
|---|---|
| `1704a5b` | Per-item labelling: items.csv, palette, assign tool, id maps saved beside the binary masks |
| `86b28ef` | Unassigned pixels drawn as hazard stripes; counted in the status bar; `items left` row tag |
| `4102ea8` | Brush size moved to `⇧1`/`⇧2` (digits now pick items); smart edges off by default |
| `bbcc7e5` | Pan tool |
| `8b465c8` | Hold vs toggle per tool key |
| `fcdd67e` | Reference panel and copy-item-as-stamp |
| `84d625b` | Cursor visible over the reference image |
| `286263c` | Move tool: lift a blob off the frame and drag it |
| `fb2542b` | Flag frames to come back to, with notes and a filter |
| `34fd61e` | Fix: the flag button cancelled itself when a note had just been typed (blur race) |
| `ef8bceb` | `u` walks to the unassigned pixels |
| `8984c72` | Carry restricted to one run and to unlabelled frames; sidebar follows the selection |
| `c3edf77` | Reopen on the last frame |
| `b6f069a` | `⇧C` applies a mask to the whole run; carry says why it was skipped |
| `833acfd` | Screws become three classes; `multi` column |
| `dbbb7bc` | `ok` → `to do` |
| `bd1aaf0` | Never write an empty mask by stepping past a frame |
| `c9b53f0` | `--min-area`/`--max-area`/`--floor-frac` flags (the diagnosis behind it was wrong, see §5) |
| `bfdf720` | The confidence threshold is what the distant runs turn on |
| `b9bdac5` | Tape rolls named after what they actually are |

### Smaller things that were got wrong and cost a round trip

- Naming objects from the overhead set photos instead of from the run frames. Two of five tape
  rolls were described wrongly. **Read the item names off a frame of the actual run.**
- A saved row's state (`items left`) came from the server's scan at page load and was not updated
  on save, so a corrected frame kept claiming it had work left.
- `canvas { cursor:none }` applies to every canvas, including the reference pane, which draws no
  brush ring of its own.
- Restarting the annotator while Jakub is using it is fine — the page is static and his saves go
  to the new process — but he has to reload to get new code, and he will not know that unless he
  is told. Say "reload the page" every time.
