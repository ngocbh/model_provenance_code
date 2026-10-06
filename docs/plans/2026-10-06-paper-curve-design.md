# Paper figure for the fine-tuning curve

Use a single panel with target and control means, each surrounded by a shaded
band of one sample standard deviation across the three runs (`ddof=1`). Bands
keep the trajectory readable; error bars would emphasize individual checkpoints
but add clutter. Individual-run lines remain available in the diagnostic plot.

Use larger serif type sized for a paper column, a white background, light
horizontal guides, blue target and gray dashed control lines. Keep only two axis
labels and a two-entry legend; omit the title, run identifiers, progress text,
and secondary axis. The caption explains the bands and number of runs.

Aggregate only checkpoints with all runs present for both target and control.
Reject duplicate observations and inconsistent exposure; keep the actual SD
without clipping its lower band to zero. Export PDF, SVG, a 600-dpi PNG, summary
CSV, and metadata tying the figure to the verified input. Check the statistics
against a known sample and inspect the rendered figure.
