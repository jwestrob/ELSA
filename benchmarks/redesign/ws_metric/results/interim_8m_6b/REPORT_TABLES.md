# ws_metric report tables

## esm2_8m

Retrieval: exact top-60 per held-out query. vis = fraction sequence-visible (E <= 1e-3); qany = queries with >= 1 visible hit; strepinv = share of retrieved pairs to the Streptomyces reference genome that are invisible; inv TM = fraction TM>=0.5 among a hash sample of invisible retrieved pairs. AUROCs on held-out pairs; TM = sequence-invisible in-block pairs, label TM>=0.5.

| arm | round | vis all | vis other | qany other | ident vis all | other-phylum share (all) | strepinv | SD recall all | inv TM all | inv TM other | TM AUROC in | TM AUROC other | popA in | popA other | xphy other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | untrained | 0.466 | 0.330 | 0.660 | 0.405 | 0.296 | 0.0070 | 0.088 | - | - | 0.825 | 0.768 | 0.689 | 0.641 | 0.760 |
| gcenter | untrained | 0.476 | 0.334 | 0.661 | 0.405 | 0.268 | 0.0070 | 0.089 | - | - | 0.868 | 0.814 | 0.759 | 0.776 | 0.796 |
| genomecenter | untrained | 0.485 | 0.370 | 0.679 | 0.402 | 0.375 | 0.0055 | 0.090 | - | - | 0.895 | 0.840 | 0.810 | 0.801 | 0.710 |
| csls2 | untrained | 0.463 | 0.334 | 0.665 | 0.400 | 0.358 | 0.0050 | 0.086 | - | - | 0.839 | 0.789 | 0.672 | 0.629 | 0.772 |
| abtt2 | untrained | 0.489 | 0.372 | 0.677 | 0.406 | 0.335 | 0.0064 | 0.092 | - | - | 0.905 | 0.842 | 0.811 | 0.823 | 0.808 |
| abtt10 | untrained | 0.496 | 0.380 | 0.678 | 0.408 | 0.333 | 0.0065 | 0.094 | - | - | 0.903 | 0.869 | 0.817 | 0.851 | 0.838 |
| abtt30 | untrained | 0.538 | 0.420 | 0.701 | 0.420 | 0.373 | 0.0047 | 0.105 | - | - | 0.906 | 0.877 | 0.843 | 0.848 | 0.798 |
| zca10 | untrained | 0.558 | 0.433 | 0.709 | 0.428 | 0.314 | 0.0054 | 0.111 | - | - | 0.907 | 0.878 | 0.870 | 0.873 | 0.833 |
| pca256 | untrained | 0.473 | 0.332 | 0.659 | 0.404 | 0.269 | 0.0070 | 0.089 | - | - | 0.868 | 0.814 | 0.756 | 0.773 | 0.794 |
| csls_q | untrained | - | - | - | - | - | - | - | - | - | 0.834 | 0.775 | 0.558 | 0.470 | 0.719 |
| z_q | untrained | - | - | - | - | - | - | - | - | - | 0.817 | 0.772 | 0.588 | 0.494 | 0.649 |
| rank_q | untrained | - | - | - | - | - | - | - | - | - | - | - | 0.585 | 0.495 | - |
| lin256_t05_ge6_inb | r1 | 0.568 | 0.424 | 0.717 | 0.428 | 0.348 | 0.0043 | 0.114 | - | - | 0.869 | 0.838 | 0.884 | 0.848 | 0.807 |
| lin64_t05_ge6_inb | r1 | 0.548 | 0.405 | 0.706 | 0.426 | 0.351 | 0.0044 | 0.111 | - | - | 0.867 | 0.832 | 0.874 | 0.836 | 0.802 |
| lin128_t05_ge6_inb | r1 | 0.564 | 0.420 | 0.715 | 0.428 | 0.347 | 0.0043 | 0.114 | - | - | 0.868 | 0.837 | 0.882 | 0.845 | 0.805 |
| lin512_t05_ge6_inb | r1 | 0.569 | 0.426 | 0.718 | 0.429 | 0.348 | 0.0043 | 0.115 | - | - | 0.869 | 0.838 | 0.885 | 0.848 | 0.809 |
| mlp256_t05_ge6_inb | r1 | 0.560 | 0.412 | 0.711 | 0.427 | 0.340 | 0.0045 | 0.113 | - | - | 0.869 | 0.835 | 0.883 | 0.845 | 0.804 |
| lin256_t03_ge6_inb | r1 | 0.557 | 0.395 | 0.704 | 0.428 | 0.300 | 0.0050 | 0.113 | - | - | 0.846 | 0.797 | 0.880 | 0.829 | 0.721 |
| lin256_t10_ge6_inb | r1 | 0.576 | 0.444 | 0.720 | 0.426 | 0.379 | 0.0037 | 0.116 | - | - | 0.900 | 0.862 | 0.880 | 0.850 | 0.810 |
| lin256_t05_acc05_inb | r1 | 0.557 | 0.407 | 0.709 | 0.429 | 0.319 | 0.0048 | 0.113 | - | - | 0.864 | 0.821 | 0.873 | 0.827 | 0.792 |
| lin256_t05_vis_inb | r1 | 0.577 | 0.442 | 0.722 | 0.428 | 0.392 | 0.0035 | 0.117 | - | - | 0.884 | 0.856 | 0.866 | 0.830 | 0.817 |
| lin256_t05_ge6_hard | r1 | 0.568 | 0.424 | 0.717 | 0.428 | 0.348 | 0.0043 | 0.114 | - | - | 0.869 | 0.838 | 0.884 | 0.848 | 0.807 |
| lin256_t10_vis_inb | r2 | 0.579 | 0.448 | 0.722 | 0.427 | 0.392 | 0.0034 | 0.117 | - | - | 0.904 | 0.867 | 0.874 | 0.843 | 0.812 |
| zlin256_t10_ge6_inb | r2 | 0.578 | 0.446 | 0.720 | 0.427 | 0.382 | 0.0037 | 0.117 | - | - | 0.900 | 0.863 | 0.883 | 0.852 | 0.814 |
| zlin256_t10_vis_inb | r2 | 0.580 | 0.449 | 0.722 | 0.427 | 0.393 | 0.0034 | 0.118 | - | - | 0.904 | 0.868 | 0.876 | 0.844 | 0.815 |
| lin256_t20_ge6_inb | r2 | 0.563 | 0.435 | 0.714 | 0.423 | 0.391 | 0.0037 | 0.112 | - | - | 0.919 | 0.873 | 0.862 | 0.856 | 0.799 |
| lin256_t10_ge6_inb_s18 | r2 | 0.576 | 0.444 | 0.720 | 0.427 | 0.381 | 0.0037 | 0.116 | - | - | 0.900 | 0.861 | 0.881 | 0.851 | 0.813 |

Predeclared comparison: best learned minus best untrained (each picked on that metric).

| metric | rounds | best learned | best untrained | difference | margin | met |
|---|---|---|---|---|---|---|
| vis_all | r1 | lin256_t05_vis_inb | zca10 | +0.0183 | 0.05 | no |
| vis_all | r1+r2 | zlin256_t10_vis_inb | zca10 | +0.0217 | 0.05 | no |
| vis_other | r1 | lin256_t10_ge6_inb | zca10 | +0.0116 | 0.05 | no |
| vis_other | r1+r2 | zlin256_t10_vis_inb | zca10 | +0.0162 | 0.05 | no |
| tm_in_phylum | r1 | lin256_t10_ge6_inb | zca10 | -0.0065 | 0.03 | no |
| tm_in_phylum | r1+r2 | lin256_t20_ge6_inb | zca10 | +0.0120 | 0.03 | no |
| tm_other_phylum | r1 | lin256_t10_ge6_inb | zca10 | -0.0162 | 0.03 | no |
| tm_other_phylum | r1+r2 | lin256_t20_ge6_inb | zca10 | -0.0049 | 0.03 | no |

- r1: held-out genomes criterion not met; held-out phyla criterion not met; gain = NO.
- r1+r2: held-out genomes criterion not met; held-out phyla criterion not met; gain = NO.

## esmc6b

Retrieval: exact top-60 per held-out query. vis = fraction sequence-visible (E <= 1e-3); qany = queries with >= 1 visible hit; strepinv = share of retrieved pairs to the Streptomyces reference genome that are invisible; inv TM = fraction TM>=0.5 among a hash sample of invisible retrieved pairs. AUROCs on held-out pairs; TM = sequence-invisible in-block pairs, label TM>=0.5.

| arm | round | vis all | vis other | qany other | ident vis all | other-phylum share (all) | strepinv | SD recall all | inv TM all | inv TM other | TM AUROC in | TM AUROC other | popA in | popA other | xphy other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | untrained | 0.320 | 0.137 | 0.544 | 0.501 | 0.059 | 0.0706 | 0.082 | - | - | 0.629 | 0.570 | 0.763 | 0.454 | 0.778 |
| gcenter | untrained | 0.317 | 0.132 | 0.531 | 0.503 | 0.054 | 0.0841 | 0.082 | - | - | 0.686 | 0.588 | 0.720 | 0.376 | 0.777 |
| genomecenter | untrained | 0.374 | 0.216 | 0.614 | 0.480 | 0.251 | 0.0289 | 0.092 | - | - | 0.755 | 0.629 | 0.728 | 0.385 | 0.786 |
| csls2 | untrained | 0.342 | 0.172 | 0.585 | 0.496 | 0.122 | 0.0501 | 0.088 | - | - | 0.700 | 0.640 | 0.755 | 0.572 | 0.774 |
| abtt2 | untrained | 0.367 | 0.184 | 0.583 | 0.497 | 0.076 | 0.0909 | 0.093 | - | - | 0.800 | 0.722 | 0.772 | 0.486 | 0.741 |
| abtt10 | untrained | 0.395 | 0.223 | 0.615 | 0.496 | 0.073 | 0.0984 | 0.098 | - | - | 0.770 | 0.742 | 0.841 | 0.614 | 0.832 |
| abtt30 | untrained | 0.454 | 0.332 | 0.684 | 0.488 | 0.108 | 0.1123 | 0.109 | - | - | 0.804 | 0.805 | 0.864 | 0.743 | 0.865 |
| zca10 | untrained | 0.609 | 0.519 | 0.763 | 0.466 | 0.153 | 0.0897 | 0.138 | - | - | 0.857 | 0.849 | 0.925 | 0.835 | 0.930 |
| pca256 | untrained | 0.273 | 0.106 | 0.493 | 0.506 | 0.054 | 0.0814 | 0.071 | - | - | 0.683 | 0.583 | 0.677 | 0.340 | 0.766 |
| csls_q | untrained | - | - | - | - | - | - | - | - | - | 0.687 | 0.619 | 0.560 | 0.279 | 0.722 |
| z_q | untrained | - | - | - | - | - | - | - | - | - | 0.712 | 0.684 | 0.641 | 0.515 | 0.627 |
| rank_q | untrained | - | - | - | - | - | - | - | - | - | - | - | 0.638 | 0.526 | - |
| lin256_t05_ge6_inb | r1 | 0.577 | 0.304 | 0.679 | 0.472 | 0.209 | 0.0123 | 0.140 | - | - | 0.823 | 0.740 | 0.894 | 0.647 | 0.856 |
| lin64_t05_ge6_inb | r1 | 0.476 | 0.199 | 0.599 | 0.495 | 0.273 | 0.0101 | 0.130 | - | - | 0.808 | 0.720 | 0.885 | 0.619 | 0.844 |
| lin128_t05_ge6_inb | r1 | 0.546 | 0.267 | 0.657 | 0.479 | 0.228 | 0.0117 | 0.137 | - | - | 0.818 | 0.736 | 0.892 | 0.637 | 0.850 |
| lin512_t05_ge6_inb | r1 | 0.587 | 0.318 | 0.689 | 0.470 | 0.201 | 0.0126 | 0.141 | - | - | 0.824 | 0.745 | 0.895 | 0.651 | 0.857 |
| mlp256_t05_ge6_inb | r1 | 0.512 | 0.228 | 0.635 | 0.487 | 0.195 | 0.0135 | 0.133 | - | - | 0.813 | 0.716 | 0.889 | 0.612 | 0.832 |
| lin256_t03_ge6_inb | r1 | 0.524 | 0.237 | 0.641 | 0.484 | 0.234 | 0.0120 | 0.135 | - | - | 0.808 | 0.736 | 0.887 | 0.677 | 0.844 |
| lin256_t10_ge6_inb | r1 | 0.667 | 0.471 | 0.756 | 0.454 | 0.184 | 0.0086 | 0.149 | - | - | 0.848 | 0.756 | 0.920 | 0.621 | 0.879 |
| lin256_t05_acc05_inb | r1 | 0.566 | 0.320 | 0.693 | 0.471 | 0.160 | 0.0188 | 0.136 | - | - | 0.815 | 0.741 | 0.917 | 0.677 | 0.852 |
| lin256_t05_vis_inb | r1 | 0.668 | 0.448 | 0.757 | 0.453 | 0.209 | 0.0078 | 0.148 | - | - | 0.871 | 0.807 | 0.938 | 0.767 | 0.864 |
| lin256_t05_ge6_hard | r1 | 0.577 | 0.304 | 0.679 | 0.472 | 0.209 | 0.0122 | 0.140 | - | - | 0.823 | 0.742 | 0.895 | 0.653 | 0.857 |

Predeclared comparison: best learned minus best untrained (each picked on that metric).

| metric | rounds | best learned | best untrained | difference | margin | met |
|---|---|---|---|---|---|---|
| vis_all | r1 | lin256_t05_vis_inb | zca10 | +0.0585 | 0.05 | yes |
| vis_all | r1+r2 | lin256_t05_vis_inb | zca10 | +0.0585 | 0.05 | yes |
| vis_other | r1 | lin256_t10_ge6_inb | zca10 | -0.0480 | 0.05 | no |
| vis_other | r1+r2 | lin256_t10_ge6_inb | zca10 | -0.0480 | 0.05 | no |
| tm_in_phylum | r1 | lin256_t05_vis_inb | zca10 | +0.0139 | 0.03 | no |
| tm_in_phylum | r1+r2 | lin256_t05_vis_inb | zca10 | +0.0139 | 0.03 | no |
| tm_other_phylum | r1 | lin256_t05_vis_inb | zca10 | -0.0414 | 0.03 | no |
| tm_other_phylum | r1+r2 | lin256_t05_vis_inb | zca10 | -0.0414 | 0.03 | no |

- r1: held-out genomes criterion met; held-out phyla criterion not met; gain = NO.
- r1+r2: held-out genomes criterion met; held-out phyla criterion not met; gain = NO.

