# Specialist gate 2 results

Pre-registration: `docs/SPECIALIST-GATE-2.md`. Sealed test sets, balanced accuracy, scored once on the CPU; ECE after temperature.

| specialist | design | options | chance | **brain** | ECE | bar | ships | nose alone | prototype alone | plain baseline | CPU s/question |
|---|---|---|---|---|---|---|---|---|---|---|---|
| topic | A | 14 | 0.071 | **0.884** | 0.027 | 0.800 | **yes** | 0.761 | 0.953 | 0.983 | 0.74 |
| intent | B | 151 | 0.007 | **0.810** | 0.119 | 0.800 | no | 0.649 | 0.901 | 0.949 | 7.89 |
| support | B | 18 | 0.056 | **0.829** | 0.035 | 0.800 | **yes** | 0.534 | 0.845 | 0.884 | 1.68 |
| junk | A | 2 | 0.500 | **0.906** | 0.023 | 0.800 | **yes** | 0.616 | 0.946 | 0.908 | 0.10 |
| hate | A | 2 | 0.500 | **0.694** | 0.021 | 0.658 | **yes** | 0.487 | 0.644 | 0.711 | 0.10 |
| politeness | A | 2 | 0.500 | **0.600** | 0.028 | 0.599 | **yes** | 0.554 | 0.642 | 0.661 | 0.10 |

**Ships: topic, support, junk, hate, politeness.**


## Reading (after the numbers; the rules are as pre-registered)

- **Five of six ship: topic, junk, support area, hate and politeness.** Each passed its sealed test,
  scored once on the CPU, with good calibration (ECE 0.021–0.035). All run on the one family v1 brain and
  its fixed antenna.
- **intent does not ship on calibration alone.** Its accuracy, 0.810 across 151 options, cleared the bar;
  its ECE of 0.119 did not (limit 0.10). Likely causes: the temperature is fit on GPU validation logits and
  applied to CPU test logits, and the validation set is small (500 items over 151 options). A fix
  (temperature fit on CPU validation logits, a larger calibration set) goes into a later pre-registered gate.
  This gate is not re-graded.
- **politeness passes by 0.0006** (0.600 vs 0.9 × 0.666 = 0.599), with disputed labels. Its card must say it
  is barely above its bar and that people often disagree on politeness.
- **hate** passes its disputed-labels bar at 0.694 (bar 0.658). Its card carries the limit found on
  validation: it catches blatant hate but often flags texts that merely mention a group.
- **The report-only comparisons:** the plain baseline beats the fly on topic, intent and support area (0.95–0.98
  vs 0.81–0.88), ties him on junk (0.908 vs 0.906), and is close on hate and politeness. The fly does not beat
  a plain classifier; he decides, and he is watchable (decision 25).
- **CPU time per question** is 0.10 s for yes/no questions, 0.74 s for 14 options, 1.7 s for 18 options (one
  memory per option), and 7.9 s for 151 options (intent).
