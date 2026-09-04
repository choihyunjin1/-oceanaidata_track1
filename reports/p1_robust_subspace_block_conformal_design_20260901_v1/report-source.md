# P1 robust cross-layer subspace + block-conformal design

## Decision boundary

This is a new information family. It does not reuse the fixed Gaussian CAPA likelihood or its penalties, and it does not fit or rescore the supervised long-event proposal bank. The only signal is an input-only, station-stratified cross-layer innovation after removal of a robust common layer factor.

The first experiment is label-free. It may authorize a separate historical falsification only when all sealed false-alarm, proposal-share, concentration, and block-sensitivity ceilings pass. A failure closes the family without target access or retuning.

## Frozen method

- For each station and layer, center and scale temperature using prefix-fit median and MAD.
- At each station-time vector, estimate the common cross-layer factor by the median standardized value. The absolute deviation from that factor is the innovation statistic. This is a rank-one robust common-mode removal, not a Gaussian segment likelihood.
- Partition calibration histories into contiguous blocks. Convert held-out block maxima to finite-sample conformal p-values against calibration block maxima.
- Use the single e-value transform `e = 0.5 / sqrt(p)` and one e-BH level `q=0.01`; do not sweep alpha, rank, transform, or penalty.
- Require two concurrent layers and a 12-row contiguous run. Total proposal share is capped at 1%; a single station-layer may not contribute more than 70%.
- Primary block length is 144 rows; 72 and 288 rows are sensitivity checks only. Every block-length decision must agree.

## Primary-source basis and limitations

Candès et al., *Robust Principal Component Analysis?* (JACM 2011), DOI `10.1145/1970392.1970395`, supports separating low-rank common structure from sparse innovations. This implementation uses a deliberately simpler robust rank-one median factor and does not claim Principal Component Pursuit guarantees.

Barber et al., *Conformal prediction beyond exchangeability* (Annals of Statistics 2023), DOI `10.1214/23-AOS2276`, motivates explicit treatment of non-exchangeability. Contiguous blocks are used to preserve local temporal dependence; exact exchangeable conformal coverage is not claimed.

Wang and Ramdas, *False discovery rate control with e-values* (JRSS B 2022), DOI `10.1111/rssb.12489`, is the primary basis for the e-BH aggregation rule. The audit measures empirical held-out behavior and does not infer official-test FDR.

Dette, Schüler, and Vetter, *Multiscale change point detection for dependent data* (arXiv:1811.05956), supports retaining temporal dependence in calibration and warns against independent-error calibration.

## Repository duplication audit

- `p1_clean_state_capa_falsification_20260831_v1`: semantically distinct. That family used seasonal/graph residuals plus Gaussian-style segment likelihood and weighted interval scheduling; this family uses robust cross-layer subspace innovations plus block conformal e-values/e-BH.
- `p1_long_event_segment_proposal_rescore_20260826_v1`: semantically distinct. This family has no supervised LightGBM proposal scorer, inner cell selection, decoder bank, or 72-fit graph.
- `p1_dependence_calibrated_sparse_add_only_falsification_20260901_v1`: semantically distinct. That family thresholded the prior clean-state decoder signal directly; this family changes the representation to cross-layer common-mode innovations and applies multiplicity control through e-values.

Official test, sample submission, hidden values, candidate CSVs, and uploads remain prohibited.
