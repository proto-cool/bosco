---
name: bosco.proto.cool
description: One specimen under glass and a label column; a fruit fly brain, every neuron, lit where it fired.
colors:
  ground: "rgb(3, 3, 7)"
  divider: "rgb(24, 24, 30)"
  border-default: "rgb(34, 34, 41)"
  border-bright: "rgb(51, 51, 64)"
  text-primary: "rgb(232, 232, 236)"
  text-secondary: "rgb(150, 152, 162)"
  text-muted: "rgb(142, 142, 155)"
  text-ghost: "rgb(133, 133, 154)"
  mushroom-blue: "rgb(77, 126, 247)"
  descending-violet: "rgb(149, 103, 255)"
typography:
  wordmark:
    fontFamily: "Host Grotesk, system-ui, sans-serif"
    fontSize: "clamp(22px, 2.5vw, 26px)"
    fontWeight: 500
    lineHeight: 1.2
  numeral:
    fontFamily: "JetBrains Mono, ui-monospace, monospace"
    fontSize: "clamp(24px, 3vw, 36px)"
    fontWeight: 200
    lineHeight: 1
  body:
    fontFamily: "Host Grotesk, system-ui, sans-serif"
    fontSize: "17px"
    fontWeight: 500
    lineHeight: 1.7
  label:
    fontFamily: "Tomorrow, system-ui, sans-serif"
    fontSize: "16px"
    fontWeight: 600
    lineHeight: 1.4
    letterSpacing: "2px"
  label-inline:
    fontFamily: "Tomorrow, system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 600
    letterSpacing: "0.75px"
  annotation:
    fontFamily: "JetBrains Mono, ui-monospace, monospace"
    fontSize: "calc(16px - 2px)"
    fontWeight: 500
    lineHeight: 1.7
rounded:
  xs: "2px"
  full: "9999px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "16px"
  lg: "24px"
  xl: "40px"
  2xl: "64px"
components:
  top-line:
    backgroundColor: "{colors.ground}"
    textColor: "{colors.text-muted}"
    typography: "{typography.annotation}"
    height: "48px"
  rail-row:
    backgroundColor: "{colors.ground}"
    textColor: "{colors.text-primary}"
    typography: "{typography.body}"
    padding: "24px 0 40px"
  rail-row-heading:
    textColor: "{colors.text-muted}"
    typography: "{typography.label}"
  rail-row-heading-current:
    textColor: "{colors.text-primary}"
    typography: "{typography.label}"
  readout-line:
    textColor: "{colors.text-muted}"
    typography: "{typography.annotation}"
    height: "16px"
  readout-line-over:
    textColor: "{colors.text-primary}"
    typography: "{typography.annotation}"
    height: "16px"
  projection-word:
    textColor: "{colors.text-muted}"
    typography: "{typography.label}"
    padding: "0"
    height: "24px"
  projection-word-pressed:
    textColor: "{colors.mushroom-blue}"
    typography: "{typography.label}"
    padding: "0"
    height: "24px"
  caption:
    textColor: "{colors.text-muted}"
    typography: "{typography.annotation}"
    width: "60ch"
  figure:
    textColor: "{colors.text-primary}"
    typography: "{typography.numeral}"
  live-dot:
    backgroundColor: "{colors.mushroom-blue}"
    rounded: "{rounded.full}"
    size: "8px"
  live-dot-asleep:
    backgroundColor: "{colors.text-ghost}"
    rounded: "{rounded.full}"
    size: "8px"
  record-line:
    textColor: "{colors.text-muted}"
    typography: "{typography.annotation}"
    height: "24px"
  record-line-acted:
    textColor: "{colors.text-primary}"
    typography: "{typography.annotation}"
    height: "24px"
---

# Design System: bosco.proto.cool

## Overview

**Creative North Star: "The Specimen and the Label"**

The page is one specimen under glass and a label column beside it. It is
built entirely from ARC UI 4.2.2 tokens (base stylesheet imported from the
pinned package; no ARC components), so the world is ARC's dark ground, its
two accents, and its type contexts. This file records how bosco uses that
world, not ARC's rulebook; ARC's own rules (two-colour contract, tokens not
literals, state by tint/glow/accent text, hierarchy by lifting to primary)
live in the ARC repo and are inherited here unchanged.

The brain owns the left two thirds of the screen for the whole scroll and
never leaves; everything else is a single rail of rows separated by
hairlines. There are no cards, no boxes, no tiles of unequal size, no stat
wall. With the content removed the page is a dark field, one lit cloud of
dots on the left, and a ladder of hairlines on the right. Every figure is
one number and one sentence; text is lowercase, in the fly's voice; the
brain is the only thing that moves.

**Key Characteristics:**
- One sticky specimen (left, 8/12) and one scrolling rail (right, `minmax(20rem, 4fr)`), nothing else.
- No surfaces: rows are separated by one hairline (`--divider`), never boxed.
- Three brain colours: mushroom body in the primary accent, descending and motor in the secondary, the rest in `--text-secondary`.
- One lobe of light on the page, behind the brain; none elsewhere.
- A fourth text size below ARC's `--text-sm`: the annotation tier, `calc(var(--text-sm) - 2px)`, worn by captions, data rows, and the top line.
- Motion is decay of spikes and a threshold glow; nothing else animates.

## Colors

ARC's dark scheme, used with two accents that mean two things and a grey ramp
that means "the rest".

### Primary
- **Mushroom Blue** (`--accent-primary`): the mushroom body on the canvas (Kenyon cells, output neurons, dopamine), the live dot when he is reachable, the pressed projection word, and the readout fill for approach behaviours (engage, reply, like, groom). Positive learned valence in the people rows. The one lobe of light behind the brain is this colour at 0.22 alpha.

### Secondary
- **Descending Violet** (`--accent-secondary`): descending and motor neurons on the canvas, the `leave` readout fill, and negative learned valence. It is the colour of aversion and of the motor side; it appears nowhere decorative.

### Neutral
- **Ground** (`--bg-deep`): the only background on the page; the top line paints it again so the sticky header covers the rail.
- **Hairline** (`--divider`): row separators, the readout track, and the valence baseline. This is the page's only structural line.
- **Border Default / Bright** (`--border-default`, `--border-bright`): the valence centre mark, the scrollbar thumb, and the link underline at rest.
- **Text Primary** (`--text-primary`): body prose, numerals, the current rail heading, an over-threshold readout line, a record line where he acted, `<b>` counts inside sentences.
- **Text Secondary** (`--text-secondary`): the resting brain (every neuron outside the two named groups), the "his time" caption, muted sentences, handles in the people list, the threshold tick.
- **Text Muted** (`--text-muted`): the default colour for labels, captions, rail headings, readout lines, and record lines; everything lifts from here to primary.
- **Text Ghost** (`--text-ghost`): the live dot when asleep or unreachable.

### Named Rules
**The Three Colours Rule.** The brain is painted in exactly three colours, resolved from tokens at runtime (`rgbOf('--accent-primary')`, `--accent-secondary`, `--text-secondary`): mushroom body, descending/motor, the rest. No fourth group, no per-neuron hue, no legend swatches; the groups are named in one caption sentence.

**The Lift Rule.** Hierarchy on the rail is made by lifting muted text to `--text-primary` (the current heading, the over-threshold line, the acted window), never by stepping down the grey ramp or adding a colour.

**The One Lobe Rule.** Light exists once, as a radial gradient of the primary accent behind the brain. It is written directly on `.glass` rather than through ARC's `--lobe-ambient`, because a custom property resolves its `var()` inputs where it is declared (`:root`), not on the box that uses it; the ARC lobe token would resolve to the neutral default. No other element carries a lobe, tint, or gradient.

## Typography

**Display Font:** Host Grotesk (with system-ui, sans-serif); the wordmark only
**Body Font:** Host Grotesk (with system-ui, sans-serif)
**Label Font:** Tomorrow (with system-ui, sans-serif); rail headings, projection words, captions' terms
**Mono Font:** JetBrains Mono (with ui-monospace, monospace); numerals, handles, times, rates, the record

**Character:** A lab label: quiet grotesk prose, tracked Tomorrow headings that read as section tabs, and thin mono numerals that sit next to their label at the baseline. All lowercase in the fly's voice.

### Hierarchy
- **Wordmark** (500, `--heading-size`, 1.2): "bosco" in the top line, and nowhere else. `h1` on the page is visually hidden.
- **Numeral** (200, `--numeral-size`, `--glyph-lh`, tabular): the large figure beside a label: neurons firing, kenyon cells active, silence, people he can smell. One figure per row at most.
- **Body** (500, `--body-size`, 1.7): sentences on the rail and his own post text. Measure `max-width: 44ch` on sentences and posts.
- **Label** (600, `--text-sm`, `--ui-lh`, 2px tracking): rail headings (`now`, `today`, `memory` …) and the three projection words. Lowercase; ARC's label context without the uppercase.
- **Label-inline** (600, `--text-xs`, 0.75px tracking): the term beside a numeral and the behaviour name on a readout line.
- **Annotation** (mono or body at `--annot-size` = `calc(var(--text-sm) - 2px)`, 1.7 on captions): captions under the brain and under each row (`max-width: 60ch`), readout values, the top line's handle and status, people rows, post metadata, record lines, the fine print.

### Named Rules
**The Annotation Tier Rule.** Data rows, captions and the top line wear one size, `--annot-size`, derived from `--text-sm` minus 2px so it moves with ARC's scale. Nothing on the page is smaller; nothing between it and `--text-sm` is invented.

**The Figure-and-Label Rule.** A number is a `numeral` beside a `label-inline` term, both on the same baseline with `--space-sm` between. A number never stands alone, and a row never shows more than one.

## Layout

The page is a two-column grid, `8fr` / `minmax(20rem, 4fr)`, with a
`--space-xl` column gap and `--space-lg` page gutters, under a sticky top line
`--top-h` (`--space-xl + --space-sm`, 48px) tall. The specimen column is
sticky at `top: var(--top-h)` and exactly `100dvh - var(--top-h)` tall, so the
brain fills the viewport for the whole scroll; the canvas takes the flex
remainder, and the projection words and atlas caption sit under it at
`--space-md`. Captions overlay the canvas's top-left corner, one per line,
`--space-xs` apart.

The rail is a single column of rows. Each row is padded `--space-lg` above
and `--space-xl` below, stacks its contents at `--space-md`, and ends in one
hairline; the last row has none. Inside a row, lists (readout, people, the
record) stack at `--space-xs` to `--space-sm`; posts at `--space-md`. Row
order is fixed: now, today, memory, what he said, the record, what this is.

Readout lines are a three-column grid (`4.5rem 1fr 4.5rem`): behaviour name,
hairline track with a threshold tick at 62.5% (the fill scales to
`ratio / 1.6`, so threshold sits at 1/1.6), rate in mono. People rows are
`1fr 5rem 2.5rem`; record lines are `3rem 4rem 1fr max-content` with the
action column sized to content so it never wraps.

**Phone (≤ 64rem):** one column. The specimen is no longer sticky (a pinned
4:3 canvas would cover most of a phone screen); the canvas locks to 4:3,
captions fall below it as static text, the handle leaves the top line, the
projection words and caption wrap, and the record drops its "smell" column
(`3rem 1fr max-content`). The rail follows at full width.

## Elevation & Depth

No shadows and no surfaces. Depth is one ambient lobe behind the brain, the
additive glow of lit neurons on the canvas (`globalCompositeOperation:
'lighter'`), and two token glows used as state: `--glow-status` on the live
dot when he is reachable, and `--glow-xs` on a readout fill that has crossed
threshold and on the pressed projection word. The sticky top line covers the
rail by repainting the ground, not by a shadow.

### Named Rules
**The Hairlines-Not-Boxes Rule.** Structure is a 1px `--divider` line: under
the top line, between rows, as the readout track, as the valence baseline.
Nothing is boxed, tinted, bordered on four sides, or lifted.

**The Glow-Is-State Rule.** A glow appears only when something crosses a
threshold (live, over-threshold, pressed). It is never decorative and never
at rest.

## Shapes

Square. Nothing has a radius except the live dot (`--radius-full`, 8px) and
the focus ring (`--radius-xs`, 2px outline corner). Tracks and fills are 1px
and 2px rules; the valence mark is a 2px segment from a centre line. Links
are underlined with `--border-bright` at `0.2em` offset, lifting to the
accent on hover. The projection words are buttons stripped to text with a
24px minimum height.

## Components

### Top line
A sticky 48px strip on the ground with one hairline beneath. Wordmark left,
handle in mono annotation beside it (hidden on phones), then a spacer, then
the live indicator and "in development" in mono annotation. The handle lifts
to primary on hover.

### Live indicator
- **Style:** an 8px `--radius-full` dot and a mono annotation word.
- **State:** asleep or unreachable is `--text-ghost` with muted text; live is `--accent-primary` with `--glow-status` and secondary text. Polled every second from `activity.bin`.

### Specimen
- **Canvas:** every central-brain neuron at its soma, 2D projection (front, top, side). Base layer at 0.5 alpha for the two named groups, 0.3 for the rest; lit cells drawn additively on top at 0.9 (0.45 for the rest) with radius growing with intensity.
- **Motion:** a spike adds `0.35 + 0.12 × count` intensity, capped at 1.5, and decays by 0.9 per frame (0.7 under reduced motion). The OG still sets `hold` so lit cells never decay.
- **Captions:** overlaid top-left: two figure-and-label pairs and one mono line ("lived · in step / N min behind"). The canvas has an `aria-label`; the `h1` is screen-reader only.
- **Projection words:** three text buttons in label context; pressed is `--accent-primary` with `--glow-xs` text-shadow, hover lifts to secondary. No legend buttons; groups are named in the caption.

### Rail row
- **Shape:** no box; `--space-lg` above, `--space-xl` below, one `--divider` beneath.
- **Heading:** `h2` in label context, muted; lifts to `--text-primary` with a `--duration-base` `--ease-out` transition while the row is the current one (IntersectionObserver, root margin `-10% 0 -70% 0`).
- **Contents:** at most one figure, one or two sentences at `44ch`, an optional list, one caption at `60ch`.

### Readout line
- **Style:** behaviour name (label-inline, muted), a 16px-tall track with a hairline and a 1px `--text-secondary` tick at threshold, and a 2px fill scaled from the left; rate in mono annotation, right-aligned, tabular.
- **State:** fill is `--accent-primary`, or `--accent-secondary` for `leave`. Over threshold the whole line lifts to primary (`--duration-fast`) and the fill takes `--glow-xs`.

### Sentence
Body prose at `44ch`; counts inside it are `<b>` at body weight lifted to
primary and tabular. A `muted` sentence is `--text-secondary` throughout.

### People row
`1fr 5rem 2.5rem`: handle in mono annotation (secondary, lifts to primary on
hover, ellipsised), a valence mark (hairline baseline, 1px `--border-default`
centre, 2px segment right in blue or left in violet, up to 50% each way), and
a familiarity count `×n` in muted mono.

### Post
A block link at `44ch`: his text in body, then a mono annotation line (kind ·
age · likes) in muted that lifts to secondary on hover. Text is fetched by
URI at view time; the server never stores it.

### Record line
`3rem 4rem 1fr max-content` in mono annotation, muted; a window where he acted
lifts to primary. Columns: age, smell (dust / mention / browse), who, what
(silence, or the action, with "· withheld" when a cap held it).

### Empty state
One muted sentence in the fly's voice ("nobody yet. he has not met anyone he
remembers."; "nothing yet."). Never a placeholder box or skeleton.

### Top line navigation
Three words in the label context at annotation size after the handle: `now`,
`days`, `how he works`. The current page's word is lifted to `--text-primary`
(`aria-current="page"`); the rest are muted and lift to secondary on hover. On
phones the handle and "in development" leave the line so the words fit.

### Live indicator, three states
The activity file carries the wall time it was written. Within 20 s the dot
is the accent with `--glow-status` and the word is `live`; up to four minutes
the dot is `--text-secondary` without a glow and the word is `between
seconds · Ns` (he is fetching or composing; no second has been simulated);
past that the dot is ghost and the word says when the last second was. Three
missed fetches in a row read `asleep or unreachable`.

### Sheet (days, how he works)
A second composition for pages without the specimen: a sticky ladder column
(`minmax(14rem, 4fr)`) and the same rail (`8fr`). The ladder is a list of
mono annotation rows under hairlines (dates with day-of-life and silence, or
section names in the label context); the current one is lifted. On the days
page the ladder is the day picker; on the how page it follows the current row
with the same IntersectionObserver rule as the rail. Below 64rem the ladder
sits above the rail, static.

### Day heading
The rail's first row on a day report carries the date in the display context
(`--heading-size`, lowercase, "monday, september 14") inside the label-context
`h2`, with a mono sub-line beneath (day of life, whole day or so far, his
time zone). The one place a display word appears besides the wordmark.

### Hour strip
Twenty-four bars on a hairline baseline, `max-width: 44ch`, 56px tall,
`--border-bright` for everything he read and `--accent-primary` for what he
acted on, a 3px `--text-secondary` dot beneath an hour where dust landed.
Four mono axis words beneath (midnight, 6 am, noon, 6 pm). One strip per day,
nothing else charted.

### People row (revised)
`1fr 7rem 3.2rem max-content`: handle, valence mark, the value in mono
secondary (`+0.21`, `−0.05`, `0`), the count. The mark's track is `--space-md`
tall with a `--text-secondary` centre tick; the segment is 3px, full at
±0.5 since his verdicts live well inside ±1. Phones narrow the track to 5rem.

### Prose row
The how page's rows are several paragraphs at the sentence measure (44ch) in
`--text-secondary`, lifted to primary while current; `<b>` inside is the
lifted term. Captions carry `<b>` in secondary for a defined word (withheld).

### Social card (og.html → og.png)
The same world as a still: 1200×630, the specimen at 700px wide under the
same lobe, the wordmark at `2.4 × --heading-size`, one line at `1.45 ×
--body-size` in secondary, the URL in tracked mono. Rendered by
`scripts/render_og.cjs` from the atlas and a real second of activity; the
PNG carries its provenance in an `impeccable:prompt` tEXt chunk.

## Do's and Don'ts

### Do:
- **Do** paint the brain from tokens at runtime (`rgbOf`), so a theme change recolours the specimen without touching `brain.js`.
- **Do** keep every row to one figure, one sentence, and one caption; put the counts inside the sentence as `<b>`.
- **Do** separate rows with one `--divider` hairline and nothing else.
- **Do** put captions, data rows, and the top line on `--annot-size`, and derive it from `--text-sm` rather than writing a pixel size.
- **Do** lift to `--text-primary` for the current heading, an over-threshold line, or a window where he acted; that is the whole state vocabulary for text.
- **Do** write the lobe on the box that needs it, as `rgba(var(--accent-primary-rgb), 0.22)` fading to 0 at 68%, and nowhere else.
- **Do** show absence honestly: a muted sentence, the live dot going ghost, "asleep or unreachable", lag in minutes.
- **Do** drop stickiness and lock the canvas to 4:3 below 64rem.

### Don't:
- **Don't** add a card, panel, tile, box, tint, or four-sided border anywhere on the rail; the specimen is the only figure and the hairlines are the only structure.
- **Don't** add a fourth brain colour, a legend, or per-neuron hues; three colours named in one caption sentence.
- **Don't** use `--accent-secondary` for anything but aversion and the motor side (descending neurons, `leave`, negative valence).
- **Don't** animate anything but spike decay on the canvas and the threshold-crossing glow and lift; no scroll effects, no counters, no fades between polls.
- **Don't** add a glow at rest; glow is a crossed threshold.
- **Don't** put a stat wall, a grid of numbers, or a second numeral in a row.
- **Don't** capitalise; the page speaks in the fly's lowercase register, including headings and the wordmark.
- **Don't** use ARC's `--lobe-ambient` on a box expecting it to take the accent; it resolves at `:root`.
