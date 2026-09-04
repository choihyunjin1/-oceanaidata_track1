# P1 v33b nested I-layer ablation

## Conclusion

`TERMINAL_NO_GO`. Prefix-only layer selection did not generalize: Q3 lost 0.001282506 F1, Q4 gained 0.000217205, and Q3+Q4 lost 0.000680871. No official materialization is authorized.

## Frozen procedure and result

- Q2 abstained. Q3 used layer 2 selected on Q2; Q4 used layers 2, 3, and 4 selected on Q2+Q3.
- The candidate removed only incumbent-negative I-ORS additions from frozen E150; anchor removals were zero.
- Q3 removed 29 rows (21 TP, 8 FP); Q4 removed 2 rows (0 TP, 2 FP).
- Pooled delta was -0.000459053 with day-block CI90 [-0.001943783, 0.000514611] and P(improve)=0.3538.
- Q3+Q4 delta was -0.000680871 with CI90 [-0.002964658, 0.000901612] and P(improve)=0.3754.
- Expected score delta was -0.0180963 points. Changed fraction was 0.000073629; maximum layer concentration was 0.935484.
- Fits: 0; runtime: 9.1975 seconds; official/hidden/CSV/upload operations: 0.

The full historical deployment layer set would be [2, 3, 4], but preparation remains non-executing because the strict gates failed.
