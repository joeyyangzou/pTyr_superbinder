# Fixed-test Hamming-buffer result

The frozen independent test set was identical in the primary and
Hamming-buffer analyses (n = 3,384; 1,692 positive and 1,692 negative).
Excluding every non-test sequence within Hamming distance 1 of a test sequence
removed 10,084 development candidates and retained 3,452 sequences. The
minimum development-test Hamming distance was 2.

The ten-seed Hamming-buffer ensemble achieved AUROC = 0.950 (95% bootstrap CI,
0.942-0.957), AUPRC = 0.963 (0.957-0.968), accuracy = 0.900 (0.889-0.910),
F1 = 0.901 (0.891-0.911), and MCC = 0.799 (0.779-0.820). The corresponding
primary random-split ensemble AUROC was 0.976.

Because Hamming filtering removed 74.5% of the original development pool, the
performance difference reflects both removal of near-neighbor information and
the substantial reduction in training-set coverage. The primary random split
therefore measures within-library interpolation, whereas this fixed-test
analysis is a more stringent assessment of local sequence extrapolation.
