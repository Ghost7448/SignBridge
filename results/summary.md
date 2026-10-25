# SignBridge - RAW vs RELATIVE results

Samples: 1500 | sessions: 15 | seeds: [42, 43, 44]

| Held-out position | RAW accuracy | RELATIVE accuracy |
|---|---|---|
| CENTER | 93.44% (+/-2.27) | 100.00% (+/-0.00) |
| LEFT | 33.33% (+/-0.00) | 100.00% (+/-0.00) |
| RIGHT | 67.00% (+/-0.47) | 99.89% (+/-0.16) |
| TOP | 40.11% (+/-8.65) | 100.00% (+/-0.00) |
| BOTTOM | 74.22% (+/-1.97) | 100.00% (+/-0.00) |

```
RAW       mean accuracy  61.62% | error  38.38% | worst position  33.33% | macro-F1  52.49% | latency 9.74 ms/sample
RELATIVE  mean accuracy  99.98% | error   0.02% | worst position  99.89% | macro-F1  99.98% | latency 9.70 ms/sample
```

Difference (RELATIVE - RAW): +38.36 percentage points

Random 80/20 split (leaky, for comparison only): RAW 100.00% | RELATIVE 100.00%

Limitations: one signer, one camera/lighting setup; results are preliminary.
