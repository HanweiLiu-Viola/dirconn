## What was observed

The numbers below are from the tables that follow. They are observations in four three-region
simulations at 100 Hz (one network, order chosen by BIC at most 5, BIC order 3 to 5 in practice). They
do not rank the methods, and they do not carry over to other sampling rates, network sizes or real data
without checking.

1. **The forgetting factors of the confirmed grid (0.95 to 0.995) are in the poor region.** Their
   measured memory is 0.01 to 0.02 s (1 to 2 samples), the stable AUROC is 0.65 to 0.80 at 20 dB and
   0.57 to 0.64 at 0 dB (ADTF, mean over the four scenarios), 15 to 56 % of the time points have an unstable
   model (mean over the seeds), and the direction accuracy is 0.60 to 0.74 in `switch_on`. The default `forgetting=0.99` gives 0.75, 0.73 and 0.61 (ADTF
   with the dual recursion, mean over the four scenarios at 20, 10 and 0 dB) and 0.79, 0.77 and 0.63 (TPDC).
   In `switch_on` the "latency" of 0.5 s in those rows is just the first stored time point after the
   switch: the estimate is so noisy that its mean course passes the halfway point at once. This is not
   tracking. `docs/timevarying.md` explains why: the factor sets the process noise and does not make the
   filter remember `1 / (1 - forgetting)` samples.
2. **With a longer measured memory the estimates become good.** From `forgetting = 0.9999` (memory 0.07 s)
   on, the median share of unstable time points is 0 (at most 7 % in any fit), and the stable AUROC
   (mean over the four scenarios) is 0.92 to 1.0. TPDC is at 0.990 to 1.000 for 0.99999 and 0.999999
   (memory 0.2 to 0.6 s) at all three SNR. The ADTF values are 0.95 to 1.00 for the same factors, below
   TPDC's at 20 and 10 dB. The windowed DTF (4 s and 8 s windows) is at 0.96 to 1.0. ADTF and WDTF are
   both DTFs, so the ceiling of about 0.96 to 0.98 at 10 dB is probably the indirect path A → C, which a
   DTF reports but the truth does not list: in `switch_on` at 10 dB (seed 0, `forgetting = 0.99999`) the
   mean ADTF score after the switch is 0.100 for A → C against 0.122 for the true edge B → C, while TPDC
   gives 0.005 and 0.154. This was checked on that one dataset only.
3. **The AUROC does not fall with the noise for the long-memory settings.** For ADTF (dual) with 0.99999 or
   more it is 0.96 at 20 dB and 0.99 at 0 dB. The same indirect path would explain this (noise weakens a
   two-step path more than a direct one), but that was not tested. For the short-memory settings the
   AUROC falls with the noise, as expected.
4. **The latency grows with the memory and with the noise.** In `switch_on` at 10 dB the latency of the
   mean course is 1.0, 1.5, 4.5 and 8.5 s for `forgetting` 0.9999, 0.99999, 0.999999, 0.9999999 (ADTF dual;
   memory 0.07, 0.19, 0.53, 1.13 s), and 1.0, 2.5, 6.5, 10.5 s at 0 dB: several times the measured
   memory. The windowed methods give 2.0 s for both window lengths. This is the first window centre after
   the switch (their time axis has a step of half a window), so their latency is not resolved below that.
5. **A burst of 10 s is seen by every setting with enough memory** (stable AUROC, mean over the methods and
   factors from 0.9999 on, 0.98 to 1.0 at the three SNR; 0.98 to 1.0 for WDTF). The onset latency of the 8 s window is 3 s and its offset latency
   1 s; the longest-memory ADTF settings (1.1 s) take 3 to 5 s to see the burst begin and 7.5 to 10 s to
   see it end.
6. **A flipped direction is followed, with a lag.** For `forgetting` of 0.99999 and 0.999999 and for both
   windows the direction accuracy in the stable segments is 0.98 to 1.0; the flipping edge takes 1.0 to
   7.5 s to be followed (TPDC, depending on the SNR). With the 0.9999999 setting the direction accuracy
   drops to 0.91 for ADTF (mean over the SNRs).
7. **One value per segment: the two procedures differ where theory predicts.** Segment B
   starts 1 s after the switch. In `switch_on` (10 dB) the continuous ADTF with `forgetting = 0.999999`
   gives a mean score of the switching edge of 0.111 in B (the segment fit gives 0.280) and still ranks
   perfectly (AUROC 1.000); the 5 s fit of segment B ranks slightly worse (0.963, 0.887 at
   20 dB). In `direction_flip` the continuous estimate still carries the old direction in B (AUROC 0.71 to
   0.79; A → B 0.136 against B → A 0.046 at 10 dB, so the old direction is still stronger), while the
   fit of the segment alone has AUROC 1.000 with the right direction (0.137 against 0.013). So averaging
   a continuous estimate over a segment carries the transition into the segment. Fitting the segment
   alone does not, but it has more variance when the segment is short.

## What is not shown

* A comparison at equal measured memory with a window: the grid reaches memories of 0.01 to 1.2 s, and
  the windows are 4 s and 8 s, so the two families are not compared at the same time scale.
* Order: the order is chosen once on the whole recording; a change of order inside the recording is not
  covered.
* Other networks, other sampling rates, real data. The factor that gives a good memory depends on all of
  them, so a factor on its own is not a portable setting (see below).

## The default

The default `forgetting` of `ADTF`, `TPDC` and `fit_tvar` was changed from 0.99
to **0.99999** on the basis of these tables (decided 2026-10-04). The tables were produced with explicit
factors, so they are unaffected. `forgetting` is not portable: the same factor gives a different memory
for another number of nodes, order or sampling rate. The measured memory (`TvarFit.memory_samples`, in the
provenance) is portable, so check it for your data.
