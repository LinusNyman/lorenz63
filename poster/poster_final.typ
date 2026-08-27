// =============================================================================
// Deterministic and probabilistic forecasting of the Lorenz 63 system
// A0 portrait poster · Digital Futures Summer Research Internship 2026
//
//   typst compile --root .. poster_final.typ poster_final.pdf
//
// --root .. is required: the poster reads ../artifacts/summary.json.
// Figures:  PYTHONPATH=. ./.venv/bin/python poster/make_figures.py
//
// CONVENTIONS
//
//   1. No number is typed in by hand. Every one is read from artifacts/summary.json,
//      written by run/report.py, so a stale figure and a stale number are the same failure
//      and one command fixes both. The exceptions are sigma, rho and beta, which define the
//      system rather than measure it.
//   2. Red marks the model, grey marks a reference, inside every figure and on the page
//      around them. In the §2 architecture diagrams red marks what differs from the plain
//      one-step MLP; §2's caption states that, because a diagram is not a data figure.
//   3. Figures carry the sheet. Prose appears only where a figure cannot state something:
//      the motivation and objectives, what the system is, and what each ruler measures.
//
// ORDER. The sheet runs question, system, models, rulers, results. The question is
// un-numbered; the four sections under it are numbered, because those four are the
// machinery. The cost of ending on the result is that §4 starts at about 1020 mm, below
// standing eye level, so its headline states the finding rather than naming the section.
// To reverse the order, move the whole §4 block, headline and all, above §1 and un-number
// it rather than renumbering headings.
//
// SCOPE. Five models, pinned here and in make_figures.py rather than read from
// run/report.py's REPRESENTATIVE, which carries seven: flow matching and the transformer
// are trained and scored in this repository but are not on the sheet.
//
//     row   key in summary.json   label on the sheet
//     02    02_mlp                MLP, one step
//     03    03_rollout_k8         MLP, rollout k = 8
//     04    04_leadtime           lead time
//     05    05_lstm               LSTM
//     06    06_gaussian           Gaussian
//
// This is the ladder the project brief describes (aotd_i_internship/aotdi_description.pdf
// §4): the rollout loss is attached to one rung, and lead-time prediction and the sequence
// models carry no rollout loss. Red therefore appears on one deterministic row, and rows 02
// and 03 are the same network at k = 1 and k = 8 -- the pair that answers the brief's third
// research question. Both arms have to stay drawn, or the comparison becomes a caption
// rather than a measurement.
//
// WHAT THAT PAIR SHOWS: no clear difference. On the ODE the rollout loss moves the MLP from
// 336 to 362 steps, with five-seed ranges of [242,399] against [269,482], which overlap
// almost entirely; on the SDE the median does not move (41 -> 41). The k-sweep is flat for
// the same reason across all of k = 1 / 4 / 8 / 16: ODE 336 / 423 / 362 / 333, SDE
// 41 / 41 / 41 / 43. At this step size k = 8 spans 0.2 time units, or 0.18 tau against a
// Lyapunov time of 44.6 steps, so the composed map the loss sees stretches by only
// exp(0.18) ~ 1.2 and the gradient sees little compounding. Reaching the chaotic regime
// needs k ~ 44.
//
// ROW 06 trains per step, which is the standard choice for a probabilistic forecaster
// (GenCast). Rows 02, 04 and 05 also train on true states only, so no row needs a tag.
// Unrolling the Gaussian's loss would be wrong in two ways: an MSE through a sampled chain
// drives log-sigma to the floor and silently makes the model deterministic, and an AR-NLL
// makes a one-step sigma head carry k-step uncertainty and then injects it at every step.
//
// ROW 05's ODE horizon of 440 is the longest number on the sheet and is inflated. Its
// one-step error is 0.114, eight times the MLP's, but its error plateaus for about ten steps
// after the 8-state warm-up instead of compounding, giving an effective growth rate of 0.66
// per time unit against lambda_1 = 0.897. Nothing on the sheet shows this.
//
// TYPE AND GRID. A0 portrait, 45 mm margins, 3 columns of 237 mm with 20 mm gutters,
// headlines 36-60 pt, body 25-30 pt, nothing inside a figure under 18 pt. Three bounds are
// bent deliberately, all to buy height: the title is 86 pt on two lines rather than 90-120
// on three, definitions and captions run 19.5-22 pt, and figure text sits at 18.07-19.1 pt,
// at the floor rather than above it.
//
// HEIGHT IS THE BINDING CONSTRAINT. The footer band starts at 1113 mm and the last ink above
// it ends at about 1099, so roughly 14 mm of slack remains. The figures account for 513 mm of
// the body: p_hook 236 (half width), p_conditional 95 (sharing its row with §1's prose), and
// p_scorecard 182. Adding a block means removing one; `typst compile` reporting two pages is
// the only warning, and it is a cliff rather than a gradient. To measure the slack, rasterise
// the bottom of the page and find the first fully black row:
//
//   pdftoppm -png -r 100 -x 200 -y 4050 -W 2000 -H 400 poster_final.pdf gap
//
// The figure widths are not free either: make_figures.py sets each figure's type size from
// the width it is placed at here, so that no glyph lands under 18 pt. Moving a figure to a
// different column span means resizing its type to match.
// =============================================================================

#let page-w = 841mm
#let page-h = 1189mm
#let margin = 45mm
#let gutter = 20mm
#let col = (page-w - 2 * margin - 2 * gutter) / 3   // 237 mm
// Full width is 3 * col + 2 * gutter = 751 mm. It has no binding of its own because a
// full-width figure is placed with `width: 100%`; the number matters only in make_figures.py,
// which sizes that figure's type from it.
//
// There is no `col2` any more. Nothing on the sheet spans exactly two columns: the attractor
// grid did and came off, and the hook that briefly replaced it now runs on HALVES (365.5 mm)
// rather than on this grid -- see its own comment. Reintroduce the binding if a block needs it;
// an unused `let` is worse than none, because the next reader assumes it is load-bearing.

// ── Colour ───────────────────────────────────────────────────────────────────
// Two inks and nothing else. The figures use exactly two colours, so the page around them
// does too.
#let ink     = rgb("#111111")
#let red     = rgb("#d1264b")   // the model -- identical to l63/plots.py RED
#let grey    = rgb("#5a6472")   // a reference -- identical to l63/plots.py GREY
// KTH blue is deliberately NOT on this sheet. It used to number the sections and rule them;
// section furniture is structure, and structure on a two-ink page should be the text colour.
// The only blue left is inside the KTH mark, where it is the mark.
#let tint    = rgb("#f1f4f8")
#let redtint = rgb("#fdeef2")
#let hair    = rgb("#c9d0d8")

// ── Type ─────────────────────────────────────────────────────────────────────
// Inter's optical-size axis stops at 32 pt, so anything above that is set in Inter Display,
// the cut Inter ships for display sizes.
#let display = "Inter Display"
#let sans    = "Inter"
#let serif   = "Charter"

#set page(width: page-w, height: page-h, margin: margin, fill: white)
#set text(font: serif, size: 25pt, fill: ink, lang: "en")
#set par(justify: false, leading: 0.54em)   // left justified, never block
#show math.equation.where(block: true): set align(left)   // ... and so is the one equation

// ── Data ─────────────────────────────────────────────────────────────────────
// The one place a number enters the poster.
#let S  = json("../artifacts/summary.json")
#let GT = S.ground_truth
#let M(key) = S.models.at(key)
#let DS(kind) = GT.datasets.at(kind)

#let n3(x) = if x == none { [—] } else { str(calc.round(x, digits: 3)) }
#let ni(x) = if x == none { [—] } else { str(calc.round(x, digits: 0)) }
// One decimal ALWAYS, including on a value that rounds to an integer: `str(14.0)` in Typst is
// "14", which set beside "15.8" in the same column reads as two different precisions.
#let n1(x) = if x == none { [—] } else {
  let t = str(calc.round(x, digits: 1))
  if t.contains(".") { t } else { t + ".0" }
}
#let n2(x) = if x == none { [—] } else { str(calc.round(x, digits: 2)) }

// Thin-space thousands. Read at two metres, 17411 is a different number from 17 411.
#let th(x) = {
  let s = str(calc.round(x, digits: 0))
  let parts = ()
  let i = s.len()
  while i > 3 { parts.push(s.slice(i - 3, i)); i -= 3 }
  parts.push(s.slice(0, i))
  parts.rev().join(sym.space.thin)
}

// ── Blocks ───────────────────────────────────────────────────────────────────
#let headline(n, t) = block(above: 0pt, below: 8mm)[
  #grid(columns: (auto, 1fr), column-gutter: 6mm, align: (bottom, bottom),
    text(font: display, size: 38pt, weight: 700, fill: ink)[#n],
    text(font: display, size: 38pt, weight: 600, fill: ink)[#t])
  #v(2mm)
  #line(length: 100%, stroke: (paint: ink, thickness: 1.6pt))
]

// The opening question, and the only heading on the sheet that is neither numbered nor ruled.
// Both omissions are the point: a number would file it as machinery alongside the four sections
// that ARE machinery, and the rule under a `headline` is what separates a section from the one
// above it -- there is nothing above this one but the title band. 58 pt sits between the title's
// 86 and the sections' 38, which is the order of importance it has.
#let question(t) = block(above: 0pt, below: 7mm,
  text(font: display, size: 58pt, weight: 600, fill: ink)[#t])


// ── The architecture diagrams ────────────────────────────────────────────────
// Drawn here, in vector, rather than pulled from figures/*c_arch_*.png: a box-and-arrow
// diagram is the one thing on an A0 sheet a reader stands close enough to see the pixels of,
// and the poster needs the part that differs from the MLP marked, which the shared plotting
// code has no notion of.
//
// `blk` is a layer -- a filled block with its shape under its label. `unit` is the same with
// no shape, for a block whose shape says nothing (the composed map F, a sampling step).
// `state` is an open block: a state going in or coming out, not a layer at all.
// `mark: true` is the part that differs from the plain MLP.
// SCALED UP 2026-08-27 EVENING (BH 20 -> 24 mm, fonts 24/21 -> 26/23, widths ~ +12 %): the
// diagrams read clamped at A0 distance. The budget came from the §2 caption, which halved.
// Longest chains after the bump: rollout ~320 mm, Gaussian ~340 mm, against a 365 mm column.
#let BH = 24mm

#let blk(top, bot, mark: false, w: 47mm) = box(
  width: w, height: BH, radius: 2pt,
  fill: if mark { redtint } else { tint },
  stroke: (paint: if mark { red } else { hair }, thickness: if mark { 1.8pt } else { 1.2pt }),
)[
  #set align(center + horizon)
  #stack(spacing: 1.8mm,
    text(font: sans, size: 26pt, weight: 600, fill: if mark { red } else { ink })[#top],
    text(font: sans, size: 23pt, fill: grey)[#bot])
]

#let unit(top, mark: false, w: 36mm) = box(
  width: w, height: BH, radius: 2pt,
  fill: if mark { redtint } else { tint },
  stroke: (paint: if mark { red } else { hair }, thickness: if mark { 1.8pt } else { 1.2pt }),
)[
  #set align(center + horizon)
  #text(font: sans, size: 26pt, weight: 600, fill: if mark { red } else { ink })[#top]
]

#let state(top, w: 44mm) = box(
  width: w, height: BH, radius: 2pt, stroke: (paint: ink, thickness: 1.8pt),
)[
  #set align(center + horizon)
  #text(font: sans, size: 26pt, weight: 600)[#top]
]

// A chain of blocks, laid out as a stack rather than as inline boxes in a paragraph: inline,
// each block sits on the text baseline and the line box grows to the block height PLUS the
// font's descent and the paragraph leading, which is about 10 mm of invisible padding per
// diagram. A stack has neither, and the arrows carry their own spacing.
#let chain(..items) = block(above: 4mm, below: 3mm,
  stack(dir: ltr, spacing: 0mm, ..items.pos()))
#let gap(w) = box(width: w, height: 0mm)

#let arrow(len: 12mm, mark: false) = {
  let c = if mark { red } else { grey }
  box(width: len, height: BH)[
    #place(left + horizon, line(length: len - 6mm, stroke: (paint: c, thickness: 2.4pt)))
    #place(left + horizon, dx: len - 6mm,
           polygon(fill: c, (0mm, 0mm), (6mm, 2.9mm), (0mm, 5.8mm)))
  ]
}

// One model, as a card: a name, a size, a diagram, and at most one line of red. There is no
// sentence describing what the model is -- the diagram is the description, and the author is
// standing next to the poster for the rest.
// A name, a size and the diagram. Nothing else: the diagram IS the description, and the
// author is standing next to the poster for the rest.
#let model-card(name, params, diagram) = block(breakable: false)[
  #text(font: sans, size: 28pt, weight: 700)[#name]#h(5mm)#text(font: sans, size: 20pt,
    fill: grey)[#params]
  #diagram
]

// =============================================================================
// HEADER: full-bleed black band
// =============================================================================
#let head-soft = rgb("#B8BCC4")

// The byline sits in the notch the title leaves. Because it is `place`d it is out of flow, so
// `measure` below sees the title's height only -- which is what takes ~50 mm off the black
// band that carrying the byline on a line of its own used to cost.
//
// THREE CONSTRAINTS, and every size here is set by them:
//   * the block sits on the title's FLOOR, not under its ceiling: `bottom` plus a 5 mm nudge,
//     because bottom-aligning two boxes aligns their descender lines and 86 pt has a far
//     deeper descent than 20 pt, so without the nudge the last byline line floats above the
//     title's baseline rather than sitting on it;
//   * one entry per line: name, supervisors, lab, university;
//   * right-set, not left-set. Ragged-left means every line ends at the margin, so the only
//     thing that can collide with the title is the LONGEST line's left end. Bottom-set, only
//     the NAME sits beside the title's first line (which runs to 627 mm of the page against a
//     right margin at 796 mm); the other three sit beside the second line, which stops at
//     385 mm and leaves them 410 mm. That is why this type could be larger than it is.
#let byline = align(right)[
  #set par(leading: 0.62em)
  #text(font: sans, size: 24pt, weight: 600, fill: white)[Linus Nyman] \
  #text(font: sans, size: 20pt, fill: head-soft)[
    Supervisors: Erik Sieburgh, Shahab Mirjalili \
    MultiphysX Lab, Engineering Mechanics \
    KTH Royal Institute of Technology
  ]
]

#let header-body = block(width: 100%)[
  #place(bottom + right, dy: 5mm, byline)
  // 86 pt, four under the how-to deck's 90 pt floor for a title. That floor is for a title
  // read at five metres and this one is two lines rather than three, so the block is taller
  // than the deck's own; the four points buy 10 mm of page, which the body needs more.
  #text(font: display, size: 86pt, weight: 600, fill: white)[
    #set par(leading: 0.24em)
    Deterministic and probabilistic forecasting \
    of the Lorenz 63 system
  ]
]

#let head-skirt = 24mm

#context {
  let h = measure(block(width: page-w - 2 * margin, header-body)).height
  place(top + left, dx: -margin, dy: -margin,
        rect(width: page-w, height: margin + h + head-skirt, fill: black))
  header-body
  v(head-skirt)
}

// White under the band, before the question. `head-skirt` cannot do this job: it is the BLACK
// padding under the title -- the rect is `margin + h + head-skirt` tall and the flow advances by
// the same amount, so content starts exactly at the band's lower edge and raising it only grows
// the black. This is a plain fixed spacer, deliberately not a `1fr`: it must not shrink when a
// block below it grows, because the gap between the band and the opening question is the first
// thing the eye reads on the sheet.
#v(20mm)

// =============================================================================
// CAN THE BUTTERFLY EFFECT BE PREDICTED?   -- un-numbered
// =============================================================================
// The hook, added 2026-08-27. One column of why-this-project, one butterfly, and the question
// at 58 pt over the text column. It is the only section on the sheet with neither a number nor
// a rule under its heading, and the only place the sheet says why the work exists.
//
// UN-NUMBERED ON PURPOSE. The four numbered sections are the machinery -- the system, the
// models, the rulers, the results -- and a reader consults machinery. This one is the question
// they came for, so numbering it would file it as the first of five equal things. `question`
// rather than `headline` for the same reason: the rule under a headline separates a section
// from the one above it, and there is nothing above this but the title band.
//
// THE PROSE WAS REWRITTEN 2026-08-27 EVENING and is no longer a near-quote of the brief.
// Motivation P1 still tracks aotdi_description.pdf §1. P2 now states the brief's central claim
// AS THE HYPOTHESIS UNDER TEST rather than as fact -- §4's own result is conditional (on the
// ODE the deterministic models win), so asserting the claim up front promised what the sheet
// then complicates. "Learns the MEAN of the next state" is deliberate: it plants the concept
// §1's point-vs-cloud figure draws. The objectives are now the concrete inventory (five
// models, two ground truths, one budget, four rulers) with §-refs as a map of the sheet.
// A FOUND line closed this column for a few hours on 2026-08-27 and was removed the same
// evening at the author's direction: with §4's headline stating the conclusion, the sheet said
// it twice, and the opening now reads as motivation rather than verdict. If the motivation is
// ever restated for a supervisor, quote the PDF, not this file.
//
// This is the largest block of connective prose on the sheet and a deliberate exception to
// rule 3 above, made because a poster with no statement of why the work exists reads as a set
// of measurements in search of a question.
//
// TWO COLUMNS, AND THE FIGURE GETS THE NARROW ONE. That is forced, not chosen:
// `set_aspect('equal')` is not optional for a butterfly, so the panel cannot be made wide by
// squashing it. Its axes are SQUARE as of 2026-08-27 -- squared by padding the shorter data
// range, never by stretching. As of the half-width layout it is 365.5 mm wide and 236 mm tall:
// the panel is WIDER than the butterfly's own proportions, so the surplus is air inside the
// frame rather than a stretched shape. See the frame block in make_figures.hook.
// The text takes the other half, 365.5 mm, and runs at 24 pt.
//
// THE FIGURE IS TOP-ALIGNED WITH THE HEADING, not with the body text: the heading lives inside
// the left cell rather than above the grid, so `align: (top, top)` puts the panel's top edge
// level with the cap-height of "Can". That is worth about 50 mm against the old layout, where a
// full-width headline block sat above both columns.
//
// The figure answers the heading on its own: one MLP rollout drawn over the true trajectory it
// started from, red on grey while they agree and red alone once they do not -- same butterfly,
// different path. It carries NO step count on purpose; see the two warnings in
// make_figures.hook.
//
// IT DRAWS ROW 02, THE ONE-STEP MLP, on INITIAL CONDITION 47, for 223 steps -- which is
// 5.00 TAU, the window stated in the unit §1 prints rather than in raw steps. The model is the
// BASELINE on purpose: this heading asks whether the butterfly effect can be predicted, and the
// model the reader should watch fail at it is the one the rest of the sheet is read against,
// not a rung trained to do better. `p_hook_ar` and `p_hook` are therefore the same figure;
// `HOOK_TOPIC_AR` is '02' in make_figures.py and says so.
//
// 47 IS THE 54th PERCENTILE OF THAT MODEL'S 128 STARTS -- typical, and that is the point. It
// crosses the attractor scale at 156 steps against a median of 148. The earlier picks were
// both flattering by comparison: 122 crosses at 267, the 69th percentile, and 84 belonged to a
// different model entirely. An above-average rollout on the opening figure of a sheet whose one
// real difficulty is that the MLP already looks too good is the wrong picture.
//
// THE IC MOVES WITH THE WINDOW AS WELL AS WITH THE MODEL. Two of the filters that choose it are
// measured inside the drawn window. At 9.0 tau this figure used IC 113; at 5.0 tau that same IC
// has the two curves only 4.3 apart at the last drawn step, because they happen to be near each
// other again there, and it would read as "the forecast recovered". Re-pick with
// scratchpad/pick_ic.py, which takes the window as an argument.
//
// That 156 is NOT §4's 336 for the same row: `horizon` crosses with the MEDIAN ERROR CURVE over
// 128 starts, a different statistic. Print neither number here, and never the two in one
// sentence.
// HALVES, not the 3-column grid. This is the one block on the sheet that does not sit on the
// 237 mm column grid, and it is deliberate: the question and its prose take the left half, the
// butterfly the right, so the two together span the full 751 mm. Each half is 365.5 mm.
// The figure is drawn for exactly this width (`WH` in make_figures.py); move it back onto the
// column grid and its type has to be redrawn with it.
#grid(columns: (1fr, 1fr), column-gutter: gutter, align: (top, top),
[
  #question[Can the butterfly effect be predicted?]

  // `question` already carries 7 mm below it; at 58 pt that reads as nothing, because the
  // block's box ends at the descender and the eye measures from the baseline.
  #v(6mm)
  #set text(size: 24pt)
  #text(font: sans, size: 22pt, weight: 700)[Motivation]
  #v(2mm)
  Many important systems are chaotic: weather, climate, turbulent fluid flows. A small change
  in the initial state grows quickly, so individual trajectories over long times are hard to
  predict #text(weight: 600)[even when the governing equations are known.]

  #v(3mm)
  Neural networks forecast far faster than classical solvers, but trained on mean squared
  error a network learns the #text(weight: 600)[mean] of the next state. This project tests
  the claim that averaging costs a deterministic model the statistics of a noisy system, and
  that predicting a #text(weight: 600)[distribution] over next states recovers them.

  #v(6mm)
  #text(font: sans, size: 22pt, weight: 700)[Objectives]
  #v(2mm)
  Train five neural forecasters (§2), from a one-step MLP to a Gaussian sampler, on a
  deterministic and a stochastic Lorenz 63 (§1), under one training budget, and score them
  all with four measured rulers (§3): #text(weight: 600)[accuracy over short horizons]
  against #text(weight: 600)[statistical fidelity over long ones.]

  // A tinted caption box sat here until 2026-08-27 -- "Right: one MLP rollout drawn over the
  // true trajectory it started from ... the shape is learnable; the trajectory is not." It was
  // removed. The figure carries its own legend and its own "same start" marker, so the only
  // thing the box added was the reading, and the reading is what the author says out loud.
  // Note what went with it: nothing on the sheet now states that the two curves begin from the
  // SAME initial state. The black dot and its label are the whole of that claim.
],
[
  #image("figures/p_hook.pdf", width: 100%)
])

// HALF A JOINT, not a whole one. The other four joints are `1fr` and share what the figures
// leave; this one takes 0.45 of a share so that §1 sits closer under the hook than the numbered
// sections sit under each other. The reason is that the hook already ends in a tall column of
// white -- the butterfly is 236 mm against about 250 mm of prose beside it -- so a full joint
// here reads as a second gap stacked on the first. Kept as a fraction rather than a fixed
// length so it still breathes when a block above or below it changes size.
#v(0.45fr)

// =============================================================================
// 1  THE SYSTEM
// =============================================================================
// The object of study, before any of the machinery: what it is, what makes forecasting it
// hard, and what the two datasets are. Three columns, and the two figures are drawn to the
// same placed height so they read as a pair -- one trajectory, then one step of it.
//
// sigma, rho and beta are typed in by hand and they are the ONLY numbers on the sheet that
// are. They are the definition of the system (Lorenz 1963), not a measurement of it, and the
// no-typed-numbers rule exists to stop a *result* going stale. summary.json checks them
// anyway: known.divergence is -(sigma + 1 + beta) and known.C_plus is
// (sqrt(beta(rho-1)), same, rho-1), both computed from these values in run/ground_truth.py.
#headline[1][The system]

// HALVES, matching the hook above it. p_conditional is placed at the same 365.5 mm as the
// butterfly, so the two figures line up down the right side of the sheet and the prose down the
// left. It is drawn for exactly that width (`WH` in make_figures.py).
#grid(columns: (1fr, 1fr), column-gutter: gutter, align: (top, top),
[
  // ROW 1, REDESIGNED 2026-08-27 EVENING: equations + parameters on the left, one prose
  // paragraph on the right that says WHY LORENZ and carries the measured constants inline.
  // The previous four-column grid (equations | params | constants | 18 pt glosses) packed four
  // type sizes into one row and never said what the system IS -- "messy" was the user's word
  // for it, and the missing why-this-system line was the other half of the complaint. The
  // constants lost nothing: lambda_1, tau and dt all moved into the sentence, still read from
  // summary.json, never typed by hand.
  //
  // The equations are three independent math cells, so their `=` signs align only because
  // x-dot, y-dot and z-dot happen to be the same width. Change a left-hand side and check by
  // eye. The parameters line under them is the definition of the system (Lorenz 1963) and is
  // hand-typed on purpose -- the one exception the no-typed-numbers rule names.
  #grid(columns: (auto, 1fr), column-gutter: 14mm, align: (top, top),
    stack(spacing: 4.6mm,
      text(size: 26pt)[$dot(x) = sigma (y - x)$],
      text(size: 26pt)[$dot(y) = x (rho - z) - y$],
      text(size: 26pt)[$dot(z) = x y - beta z$],
      text(font: sans, size: 21pt, weight: 600)[$sigma = 10 quad rho = 28 quad beta = 8 slash 3$],
    ),
    [
      #set text(size: 20pt)
      Lorenz 63 is the classic minimal chaotic system: a strange attractor, sensitive
      dependence on initial conditions, and equations known exactly, so every forecast can be
      judged against the truth. The state is $bold(u) = (x, y, z)$; a hat marks a model's
      prediction. Errors e-fold every
      $tau approx #ni(1 / (GT.lambda_true * DS("ode").dt))$ steps
      ($lambda_1 = #n3(GT.lambda_true) plus.minus #n3(GT.lambda_true_sd)$, Benettin estimate);
      one stored step, $Delta t = #DS("ode").dt$, is one model call.
    ],
  )

  // ROW 2: THE TWO INTEGRATORS, one per dataset, in the same left-right order as the two panels
  // beside them and as the two rows of p_scorecard. This replaced a three-column dataset table
  // and a standalone sentence about the Euler bar; both said one integrator's worth of the
  // story each, and neither named the scheme that produced the SDE.
  //
  // BOTH BLOCKS ARE UPDATE RULES ON THE SAME LINE OF ALGEBRA, so the pair is a 1:1 comparison:
  // identical up to `+ sqrt(h) g(u_n) xi_n`. Until 2026-08-27 the right-hand block carried the
  // CONTINUOUS SDE, du = f(u)dt + g(u)dW, against the left-hand block's discrete update -- two
  // different kinds of object under two headings that both name a discrete scheme, so a reader
  // could not see what the noise actually adds. Both forms now match l63/data.py exactly:
  // solve_ode does `z + f(z)*dt`, solve_sde does `z + f(z)*dt + g(z)*w*sqrt(dt)`.
  //
  // The sqrt(h) is not a stylistic choice and must not be tidied to h: a Wiener increment over
  // h has standard deviation sqrt(h), and anything else makes the limit depend on the grid.
  //
  // The 16.5 % survives from that table on purpose. It is the project's control variable, and
  // without it the sheet states the conditional width nowhere at all -- it would be readable
  // only as the size of the red clouds in the figure alongside.
  #v(7mm)
  #grid(columns: (1fr, 1fr), column-gutter: 14mm, align: (top, top),
  [
    #text(font: sans, size: 21pt, weight: 700)[Explicit Euler]#h(3mm)#text(font: sans,
      size: 19pt, fill: grey)[ODE, $b = #DS("ode").b$]
    #v(2.5mm)
    #text(size: 24pt)[$ bold(u)_(n+1) = bold(u)_n + h thin bold(f)(bold(u)_n) $]
    #v(1.5mm)
    #text(size: 20pt)[$h = Delta t slash 100$ for the ground truth. At $h = Delta t$ one step
      costs the same as #text(weight: 600)[one model call], the bar every model is measured
      against.]
  ],
  [
    // g(u) = b u is stated in words rather than inlined with a Hadamard symbol: Typst has no
    // `dot.circle`, and b is a scalar here anyway, so the product needs no operator.
    #text(font: sans, size: 21pt, weight: 700)[Euler–Maruyama]#h(3mm)#text(font: sans,
      size: 19pt, fill: grey)[SDE, $b = #DS("sde").b$]
    #v(2.5mm)
    #text(size: 24pt)[$ bold(u)_(n+1) = bold(u)_n + h thin bold(f)(bold(u)_n)
                         + sqrt(h) thin bold(g)(bold(u)_n) bold(xi)_n $]
    #v(1.5mm)
    #text(size: 20pt)[The same step and #text(weight: 600)[one term more], with
      $bold(xi)_n tilde cal(N)(0, bold(I))$ and $bold(g)(bold(u)) = b bold(u)$, so the noise
      scales with the state. #text(weight: 600)[$b$ is the project's control parameter:] at
      #DS("sde").b it spreads the next state over
      #n1(DS("sde").conditional_width_pct) % of the attractor scale.]
  ])
],
[
  // p_system -- the Lorenz-distance divergence plot -- sat here and came off 2026-08-27. It is
  // still written by make_figures.py and still on disk. What went with it: the sheet no longer
  // shows sensitive dependence anywhere, nor the measured lambda_1 as a visible slope, nor the
  // 483-vs-23 contrast between the two datasets. The constants list to the left states lambda_1
  // as a number; nothing draws it.
  #image("figures/p_conditional.pdf", width: 100%)
])

// Half a joint, like the one above §1. §1's left column now runs nearly the full height of the
// figure beside it, so a full `1fr` here reads as a gap stacked on the column's own tail.
#v(0.3fr)   // above §2 -- tightened 2026-08-27 evening at the author's direction

// =============================================================================
// 2  THE MODELS
// =============================================================================
// THE SIX ROWS ARE make_figures.py's `MODELS_AR`, and they are in the SAME ORDER as the six
// rows of p_scorecard in §4 -- top-left, top-right, then down. A reader who wants to know what
// "LSTM, AR" is in the scorecard finds it here by name, so the names have to match `SHORT_AR`
// character for character. Change one and change the other.
//
// `ar` is the red tag every deterministic row carries, and it is the one thing this sheet is
// about: the same net, trained through the composed map rather than on true states only. It is
// in the SUBTITLE rather than in the diagram for rows 04 and 05 because an unrolled loss is not
// an architecture -- rows 02 and 03 draw the unroll because for the MLP the unroll IS the whole
// of what distinguishes them, there being nothing else in an MLP to draw.
#let ar(k) = [#h(3mm) · #h(3mm) #text(fill: red, weight: 600)[AR, $k = #k$]]

// "The models", not "Five models, one contract", since 2026-08-27 evening: the plain titles
// match §1 "The system" and §3 "The rulers", and "contract" is programmer jargon of exactly
// the kind the supervisor flagged. What the old title carried -- one interface, one budget,
// one judge -- is the caption's last sentence.
#headline[2][The models]

#grid(columns: (1fr, 1fr), column-gutter: gutter, row-gutter: 3mm,

// THE BASELINE, and the only deterministic row with no red on it -- which is the point. It is
// the model the other four are an improvement on, and the sheet cannot say "training through
// the composed map helps" without it: with one arm drawn, that claim is a caption rather than
// a measurement, and the brief's third research question (can a rollout loss improve long-term
// stability) has no figure behind it at all. Restored 2026-08-27 after a few hours off the
// sheet. Its subtitle carries a grey tag for the same reason the Gaussian's does: a reader
// comparing six cards reads a missing tag as an oversight unless the card names what it is.
model-card(
  [MLP, one step],
  [#th(M("02_mlp_ode").n_params) parameters · one-step MSE#h(3mm) ·
    #h(3mm) #text(fill: grey, weight: 600)[the baseline]],
  chain(state[$bold(u)_n$], arrow(len: 10mm), unit(w: 38mm)[$F_theta$],
        arrow(len: 10mm), state(w: 52mm)[$hat(bold(u))_(n+1)$]),
),

// The sweep lives in this subtitle and not in a second card. k = 8 was a row of its own for a
// few hours; two rows of the same net at two unroll depths is a k-response the sheet has no
// room to explain, and it cost §2 a third card row. `$k in {1, 4, 8, 16}$` says the sweep
// happened in seven characters, which is what a poster can afford.
model-card(
  [MLP, rollout $k = 8$],
  [#th(M("03_rollout_k8_ode").n_params) parameters · rollout loss ·
    $k in {1, 4, 8, 16}$#ar(8)],
  chain(state[$bold(u)_n$], arrow(len: 10mm, mark: true), unit(mark: true, w: 38mm)[$F_theta$],
        arrow(len: 10mm, mark: true), state[$hat(bold(u))_(n+1)$],
        arrow(len: 10mm, mark: true), unit(mark: true, w: 38mm)[$F_theta$],
        arrow(len: 10mm, mark: true),
        box(width: 16mm, height: BH, align(center + horizon, text(size: 32pt, fill: red)[…])),
        arrow(len: 10mm, mark: true), unit(mark: true, w: 38mm)[$F_theta$],
        arrow(len: 10mm, mark: true), state(w: 52mm)[$hat(bold(u))_(n+8)$]),
),

model-card(
  "Lead time",
  [#th(M("04_leadtime_ode").n_params) parameters · $s = 1 dots
    #M("04_leadtime_ode").kwargs.s_max$],
  chain(state[$bold(u)_n$], gap(2mm), unit(w: 24mm)[$s$], arrow(),
        blk[SiLU][128], arrow(), blk[SiLU][128], arrow(),
        state(w: 52mm)[$hat(bold(u))_(n+s)$]),
),

model-card(
  "LSTM",
  [#th(M("05_lstm_ode").n_params) parameters · window
    #M("05_lstm_ode").history],
  chain(state(w: 63mm)[$bold(u)_(n-7 dots n)$], arrow(),
        blk(w: 53mm)[LSTM cell][64], arrow(), blk(w: 38mm)[linear][3], arrow(),
        state[$hat(bold(u))_(n+1)$]),
),

// NO `teacher-forced` TAG ON THIS SHEET. On the AR sheet the Gaussian was the one row not
// trained through the composed map, so the gap needed naming or it read as an oversight. Here
// FOUR of the five are trained on true states only -- rows 02, 04, 05 and this one -- so
// tagging one of them would say the others are something else. Red appears on TWO rows, 03
// and this one, and the caption defines it once: what differs from the baseline MLP -- the
// unrolled loss there, the distribution head and sampling here. (An earlier comment claimed
// red appeared only on row 03; the marked mu/log-sigma and sample boxes below disproved it.)
model-card(
  "Gaussian",
  [#th(M("06_gaussian_ode").n_params) parameters · diagonal $Sigma$],
  chain(state[$bold(u)_n$], arrow(), blk[SiLU][128], arrow(), blk[SiLU][128],
        // The head label is set a notch smaller than blk's 26 pt: the math line (bar, log,
        // bold sigma's descender) is taller than a text label, and at 26 pt it crowded the
        // "3 + 3" underneath it inside the fixed-height box.
        arrow(mark: true),
        blk(mark: true, w: 55mm)[#text(size: 23pt)[$bold(mu), thin log bold(sigma)$]][3 + 3],
        arrow(mark: true), unit(mark: true, w: 38mm)[sample], arrow(mark: true),
        state[$hat(bold(u))_(n+1)$]),
),

// EVERY SENTENCE STATES WHAT WAS DONE. The version before 2026-08-27 spent two of its five on
// absences and defences: "it is not itself a row", and a clause explaining that unrolling the
// Gaussian would drive its log-sigma to the floor. Both were true and neither belonged on the
// sheet, which has room for the design and not for the road not taken. Per-step training for a
// probabilistic forecaster is the standard choice, so it is stated as one, and the failure mode
// that rules out the alternative lives in make_figures.py's MODELS_AR block for anyone who asks.
block(width: 100%, fill: tint, inset: 6mm, radius: 4pt)[
  #set text(size: 20pt)
  // HALVED 2026-08-27 EVENING, and one claim corrected with it: the long version said "red
  // marks the ONE row that differs" while the Gaussian's diagram above it is also red. Red
  // marks what differs from the baseline MLP -- the unrolled loss on row 03, the distribution
  // head and sampling on row 06 -- and the caption now says exactly that. The rows-04/05/06
  // training detail moved off the sheet; the card subtitles (s = 1...16, window 8) carry it.
  *Every model is forecast free-running*: handed only a true start (one state; eight for the
  LSTM), then fed #text(weight: 600)[its own output], with no correction from the truth.
  *#text(fill: red)[Red] marks what differs from the plain one-step MLP*: the rollout row
  unrolls the same network's loss $k$ steps #text(fill: red, weight: 600)[through the composed
  map], and
  the Gaussian predicts a #text(weight: 600)[distribution] and samples it. Widths and depths
  are held fixed; one trainer runs every model for 3000 Adam iterations on
  #M("03_rollout_k8_ode").n_seeds seeds, and one set of rulers judges all five.
],
)

#v(0.55fr)  // above §3 -- tightened 2026-08-27 evening at the author's direction

// =============================================================================
// 3  THE RULERS
// =============================================================================
// Five columns, one ruler each, and every column answers the same three things in the same
// order: what question the ruler asks in plain words, what is actually computed, and what a
// value means. The previous version gave each ruler a single clause -- "W_1 to the truth's
// marginals / truth's own W_1" -- which is exact and tells a reader who does not already
// know Wasserstein distance nothing at all.
//
// What each ruler is READ AGAINST is drawn and labelled inside p_scorecard, in §4 directly
// below (the dashed lines and their values), so it is not restated here. The four columns there
// are in the same order as the four definitions here -- keep them that way.
#headline[3][The rulers]

// EACH HEADING CARRIES ITS QUANTITY since 2026-08-27 evening: the grey tag after the name is
// the same string as the §4 column header directly below it (time steps, sigma ratio, W_1
// ratio, lambda_1 ratio), so a reader maps definition to panel by matching the pair, and a
// literature reader gets the formal handle the informal name lacks. Horizon's body was
// CORRECTED with it: the attractor scale is the attractor's own spread ||sigma||, not "the
// distance between two unrelated true states" -- that distance is larger by roughly sqrt(2).
#let ruler(name, quantity, question, body) = [
  #text(font: sans, size: 23pt, weight: 700)[#name]#h(4mm)#text(font: sans, size: 19pt,
    fill: grey)[#quantity]
  #v(1mm)
  #text(font: sans, size: 19pt, fill: grey)[#question]
  #v(2.5mm)
  #text(font: serif, size: 20pt)[#body]
]

#grid(columns: (1fr,) * 4, column-gutter: 14mm, align: (top,) * 4,
  ruler[horizon][time steps][how long a forecast stays usable][
    Time steps until the median of $||hat(bold(u))_n - bold(u)_n||$ over 128 starts reaches
    the attractor scale, the spread of the attractor itself. Past it a forecast is no better
    than a random state from the attractor.
  ],
  ruler[spread][$sigma$ ratio][whether it knows what it does not know][
    Ensemble standard deviation of repeated forecasts from one state, divided by the truth's.
    1 means the forecast is as uncertain as the system itself; 0 means the model reports no
    uncertainty.
  ],
  ruler[climate][$W_1$ ratio][whether long rollouts visit the right places][
    Wasserstein-1 distance from the model's marginals to the truth's, over the distance
    between two halves of the truth alone. Marginals only; joint structure is not measured.
  ],
  ruler[chaos][$lambda_1$ ratio][whether it stretches errors like the system][
    Largest Lyapunov exponent of the learned map over that of the true map, same estimator.
    1 means small errors grow at the true rate; 0 or below means the map is not chaotic
    at all.
  ],
)

#v(1.7fr)   // above §4 -- widened 2026-08-27 evening; the results get the biggest breath

// =============================================================================
// 4  RESULTS
// =============================================================================
// ONE FIGURE, and it is the whole finding: `p_scorecard` is every ruler on every model with the
// five-seed range drawn through it. There is no prose: an earlier draft carried three findings
// beside a single-model error plot, and both were cut.
//
// It sits LAST, and numbered, as of 2026-08-27. It used to be first and un-numbered -- the
// finding at the top, the machinery numbered under it, so a reader who stopped after 400 mm
// still had the result. Reversed, the sheet reads as a paper: system, models, rulers, results.
// The cost is that the finding is now at 900 mm, below standing eye level, and a reader who
// stops early leaves with the apparatus and no conclusion. It is directly above the footer
// instead, which is where an eye lands last.
//
// Its four ruler columns sit directly under section 3's four ruler definitions, in the same
// order (horizon, spread, climate, chaos). That alignment is the one thing the move bought and
// it is worth keeping: do not reorder either without the other.
//
// The attractor grid (`p_attractors` -- truth beside every model, on both ground truths) used
// to sit above the scorecard and was removed 2026-08-27. It is still written by
// make_figures.py and still on disk; nothing on the sheet reads it. The joint structure it
// covered by eye is now not covered at all, and `climate` is marginals-only -- say so out loud.
//
// `p_scorecard`, the un-suffixed bank. THE FIVE ROWS ARE THE FIVE §2 CARDS IN THE SAME ORDER,
// top to bottom against §2's top-left-then-down. What it says, and the two-line version to have
// ready standing next to it:
//
//   ODE (Euler bar 32, truth's own floor 362). Four of the five clear the bar 7-14x over --
//   LSTM 440, MLP rollout k=8 362, MLP one step 336, Gaussian 240 -- and lead time does not
//   clear it at all, 15 against 32, with chaos -12.2 and alive 0.00. Its seed range runs 10 to
//   23, so it fails on every seed.
//   THE TOP TWO ROWS ARE THE SAME NET AT k = 1 AND k = 8 and their seed bars overlap almost
//   entirely ([242,399] against [269,482]), so the honest reading of the rollout loss on this
//   axis is "no clear difference", not 336 -> 362. Say that before someone reads the two dots
//   as an improvement.
//   ⚠️ THE LSTM'S 440 IS THE BEST NUMBER ON THIS SHEET AND IS PROBABLY INFLATED -- see the
//   header block. Its e1 is 8x the MLP's and its error plateaus after warm-up instead of
//   compounding. Do not lead with it without the caveat.
//   Note also that 362 IS the truth's own floor: the MLP at k = 8 sits exactly on the line past
//   which "tracks the truth" can only mean "tracks OUR discretisation". The line is drawn and
//   not explained.
//   SDE b=0.6 (one-step conditional 16.5 % of attractor scale). Every horizon falls to 8-45
//   against a bar of 26. The four deterministic models draw the NOISELESS attractor: spread
//   0.00 to four significant figures, climate 20-146 against truth's own 0.66. The Gaussian is
//   the only row with spread 1.00 and chaos 1.00, and on climate it is nearest the truth at
//   5.6 -- which is still 8x truth's own score. It pays for that with horizon, 28 against the
//   deterministic rungs' 41-45. horizon and the other three rulers rank the five in opposite
//   orders. That is the poster.
//   DO NOT SAY the models are "past the point where the reference can be trusted" on the SDE.
//   The 23 there is the spread of two INDEPENDENT realisations, i.e. the system's own noise,
//   not a resolution limit -- and a conditional-mean predictor beats it by sqrt(2) by
//   construction, which is exactly the gap measured (model e1 1.78 against the floor's 2.79).
//
// WHAT THIS SHEET DOES NOT SAY, and the first thing to have ready when asked: what AR training
// on the OTHER rungs would do. It is in artifacts/summary.json and it is not uniformly good --
// on the ODE it costs the LSTM 440 -> 147, with climate 1.8 -> 73 and chaos 1.00 -> 0.10; on
// the SDE it takes the lead-time rung from alive 0.00 to 1.00 on every seed. `poster.typ` is
// the sheet that draws those arms.
// The one heading on the sheet that states a conclusion rather than a topic -- a deliberate
// exception, made 2026-08-27 on supervisor feedback that the core takeaway has to be visible
// without reading down to the figure. "The results", not "Results", so the four sections read
// as one family: The system, The models, The rulers, The results.
//
// A CUSTOM HEADER, not `headline`, because the right-hand end of the rule row carries the
// scorecard's legend -- dot = median over 5 seeds, bar = full seed range, dashed = reference.
// Until 2026-08-27 evening that encoding was explained nowhere on the sheet and the author had
// to say it to every reader. In the headline row it costs zero height.
#block(above: 0pt, below: 8mm)[
  #grid(columns: (auto, auto, 1fr), column-gutter: 6mm,
    align: (bottom, bottom, bottom + right),
    text(font: display, size: 38pt, weight: 700, fill: ink)[4],
    text(font: display, size: 38pt, weight: 600, fill: ink)[The results],
    text(font: sans, size: 19pt, fill: grey)[dot = median over 5 seeds #h(2.5mm) · #h(2.5mm)
      bar = full seed range #h(2.5mm) · #h(2.5mm) dashed = reference])
  #v(2mm)
  #line(length: 100%, stroke: (paint: ink, thickness: 1.6pt))
]

#image("figures/p_scorecard.pdf", width: 100%)

// =============================================================================
// FOOTER: full-bleed black band, matching the header
// =============================================================================
// TWO ANCHORED CORNERS, reworked 2026-08-27: the institutions at the left margin, the author
// and the QR at the right, text flowing inward from each. The old footer was one left-anchored
// row -- three marks and a text block that ended mid-page -- which left the right two-fifths of
// the band dead black. The header never has that problem because the title holds one margin and
// the byline the other; the footer now does the same. Bottom-right is also where reading
// gravity exits the sheet, so the exit carries the two things a leaving reader needs: the face
// to find in the room, and the QR to take the work home.
//
// The KTH mark here is the BLUE lockup -- a white seal on a solid blue square. On black the
// black-square version loses its ground and only the seal shows, so the blue one is the file
// that reads as a mark rather than as loose white line-work.
//
// The DF symbol is aotdoa_df_logo_black.png, derived 2026-08-27 from ../../digital_futures.jpg
// (900 px black-on-white JPG): thresholded and trimmed to the ink. It sits on a WHITE TILE at
// the full 44 mm, matching the KTH tile beside it and the QR tile in the other corner -- two
// tiles per corner, same geometry. (A bare white-on-band version was tried first and read as
// loose line-work next to three filled squares.) The 5 mm inset is the symbol's clear space;
// the QR's 2.2 mm does not transfer, because a logo wants breathing room where a QR wants
// every module it can get. The white "Digital Futures" text beside it does wordmark duty; the
// grey line under names the programme. The references are still not printed: they are behind
// the QR, and the caption says so.
//
// The QR stays dark-on-light on a white tile rather than being inverted to match the band.
// Inverted codes look tidier and some scanners refuse them; a white tile is what the format
// specifies and the quiet zone comes with it.
//
// The URL is printed as text on purpose, and sits beside the tile it duplicates: people
// photograph posters more often than they scan them, and a printed address survives even if
// the page behind the QR goes up late. The grey list under it is the QR's caption -- what a
// visitor finds there -- next to the thing it describes, not dangling under the contact line
// as it used to.
#let mark-w = 44mm
#let foot-pad = 16mm

#let footer-body = grid(
  columns: (auto, auto, auto, 1fr, auto, auto, auto), column-gutter: 9mm,
  align: (horizon,) * 7,
  // Left corner: who ran the work.
  box(width: mark-w, height: mark-w, clip: true, radius: 2pt,
      image("assets/aotdoa_kth_logo_blue.png", width: mark-w)),
  box(width: mark-w, height: mark-w, fill: white, radius: 2pt, inset: 5mm,
      image("assets/aotdoa_df_logo_black.png", width: 100%)),
  text(font: sans, size: 21pt, fill: head-soft)[
    #text(weight: 600, fill: white)[Digital Futures] \
    #v(1mm)
    Summer Research Internship 2026
  ],
  [],
  // Right corner: who did it, and where it lives.
  align(right, text(font: sans, size: 21pt, fill: head-soft)[
    #text(weight: 600, fill: white)[Linus Nyman] #h(3mm) · #h(3mm) linusnym\@kth.se \
    #v(1mm)
    #text(weight: 600, fill: white)[linusnyman.com/academic] \
    #v(1mm)
    code · figures · references · presentation
  ]),
  box(width: mark-w, height: mark-w, clip: true, radius: 2pt,
      image("assets/aotdoa_profile.jpg", width: mark-w)),
  box(width: mark-w, height: mark-w, fill: white, radius: 2pt, inset: 2.2mm,
      image("assets/aotdoa_qr_academic.svg", width: 100%)),
)

#v(1fr)

// The band bleeds off the bottom of the sheet, so its lower padding is the 45 mm page margin
// whether anyone wants it or not. Left in flow, the marks therefore sat 12 mm below the top of
// the band and 45 mm above the bottom of it -- top-heavy, with 30 mm of the band black and
// doing nothing. So BOTH the band and its content are placed: the band is `foot-pad` + marks +
// `foot-pad` tall, and the marks are pushed down into the margin so the two paddings are
// equal. That centres what is in the band and makes the band 25 mm shorter.
//
// Placing the content costs the one warning this layout has -- content that overruns no longer
// forces a second page, it silently prints over the band -- so the trailing `v` puts the part
// of the band that overlaps the text area back into the flow as reserved space.
#context {
  let h = measure(block(width: page-w - 2 * margin, footer-body)).height
  let band = h + 2 * foot-pad
  place(bottom + left, dx: -margin, dy: margin,
        rect(width: page-w, height: band, fill: black))
  place(bottom + left, dy: margin - foot-pad, footer-body)
  v(band - margin)
}
