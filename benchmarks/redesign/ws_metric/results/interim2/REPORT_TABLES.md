# ws_metric report tables

## esm2_8m

Retrieval: exact top-60 per held-out query. vis = fraction sequence-visible (E <= 1e-3); qany = queries with >= 1 visible hit; strepinv = share of retrieved pairs to the Streptomyces reference genome that are invisible; inv TM = fraction TM>=0.5 among a hash sample of invisible retrieved pairs. AUROCs on held-out pairs; TM = sequence-invisible in-block pairs, label TM>=0.5.

| arm | round | vis all | vis other | qany other | ident vis all | other-phylum share (all) | strepinv | SD recall all | inv TM all | inv TM other | TM AUROC in | TM AUROC other | popA in | popA other | xphy other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | untrained | 0.466 | 0.330 | 0.660 | 0.405 | 0.296 | 0.0070 | 0.088 | - | - | 0.826 | 0.768 | 0.689 | 0.641 | 0.761 |
| gcenter | untrained | 0.476 | 0.334 | 0.661 | 0.405 | 0.268 | 0.0070 | 0.089 | - | - | 0.868 | 0.814 | 0.759 | 0.776 | 0.794 |
| genomecenter | untrained | 0.485 | 0.370 | 0.679 | 0.402 | 0.375 | 0.0055 | 0.090 | - | - | 0.895 | 0.840 | 0.810 | 0.801 | 0.709 |
| csls2 | untrained | 0.463 | 0.334 | 0.665 | 0.400 | 0.358 | 0.0050 | 0.086 | - | - | 0.840 | 0.790 | 0.672 | 0.629 | 0.771 |
| abtt2 | untrained | 0.489 | 0.372 | 0.677 | 0.406 | 0.335 | 0.0064 | 0.092 | - | - | 0.905 | 0.842 | 0.811 | 0.823 | 0.807 |
| abtt10 | untrained | 0.496 | 0.380 | 0.678 | 0.408 | 0.333 | 0.0065 | 0.094 | - | - | 0.904 | 0.869 | 0.817 | 0.851 | 0.837 |
| abtt30 | untrained | 0.538 | 0.420 | 0.701 | 0.420 | 0.373 | 0.0047 | 0.105 | - | - | 0.906 | 0.877 | 0.843 | 0.848 | 0.796 |
| zca10 | untrained | 0.558 | 0.433 | 0.709 | 0.428 | 0.314 | 0.0054 | 0.111 | - | - | 0.907 | 0.878 | 0.870 | 0.873 | 0.831 |
| pca256 | untrained | 0.473 | 0.332 | 0.659 | 0.404 | 0.269 | 0.0070 | 0.089 | - | - | 0.868 | 0.814 | 0.756 | 0.773 | 0.792 |
| csls_q | untrained | - | - | - | - | - | - | - | - | - | 0.835 | 0.775 | 0.558 | 0.470 | 0.719 |
| z_q | untrained | - | - | - | - | - | - | - | - | - | 0.818 | 0.772 | 0.588 | 0.494 | 0.649 |
| rank_q | untrained | - | - | - | - | - | - | - | - | - | - | - | 0.585 | 0.495 | - |
| lin256_t05_ge6_inb | r1 | 0.568 | 0.424 | 0.717 | 0.428 | 0.348 | 0.0043 | 0.114 | - | - | 0.869 | 0.837 | 0.884 | 0.848 | 0.806 |
| lin64_t05_ge6_inb | r1 | 0.548 | 0.405 | 0.706 | 0.426 | 0.351 | 0.0044 | 0.111 | - | - | 0.867 | 0.832 | 0.874 | 0.836 | 0.800 |
| lin128_t05_ge6_inb | r1 | 0.564 | 0.420 | 0.715 | 0.428 | 0.347 | 0.0043 | 0.114 | - | - | 0.869 | 0.836 | 0.882 | 0.845 | 0.804 |
| lin512_t05_ge6_inb | r1 | 0.569 | 0.426 | 0.718 | 0.429 | 0.348 | 0.0043 | 0.115 | - | - | 0.869 | 0.838 | 0.885 | 0.848 | 0.807 |
| mlp256_t05_ge6_inb | r1 | 0.560 | 0.412 | 0.711 | 0.427 | 0.340 | 0.0045 | 0.113 | - | - | 0.869 | 0.835 | 0.883 | 0.845 | 0.803 |
| lin256_t03_ge6_inb | r1 | 0.557 | 0.395 | 0.704 | 0.428 | 0.300 | 0.0050 | 0.113 | - | - | 0.846 | 0.797 | 0.880 | 0.829 | 0.720 |
| lin256_t10_ge6_inb | r1 | 0.576 | 0.444 | 0.720 | 0.426 | 0.379 | 0.0037 | 0.116 | - | - | 0.901 | 0.862 | 0.880 | 0.850 | 0.808 |
| lin256_t05_acc05_inb | r1 | 0.557 | 0.407 | 0.709 | 0.429 | 0.319 | 0.0048 | 0.113 | - | - | 0.864 | 0.820 | 0.873 | 0.827 | 0.791 |
| lin256_t05_vis_inb | r1 | 0.577 | 0.442 | 0.722 | 0.428 | 0.392 | 0.0035 | 0.117 | - | - | 0.884 | 0.855 | 0.866 | 0.830 | 0.815 |
| lin256_t05_ge6_hard | r1 | 0.568 | 0.424 | 0.717 | 0.428 | 0.348 | 0.0043 | 0.114 | - | - | 0.869 | 0.838 | 0.884 | 0.848 | 0.806 |
| lin256_t10_vis_inb | r2 | 0.579 | 0.448 | 0.722 | 0.427 | 0.392 | 0.0034 | 0.117 | - | - | 0.905 | 0.867 | 0.874 | 0.843 | 0.810 |
| zlin256_t10_ge6_inb | r2 | 0.578 | 0.446 | 0.720 | 0.427 | 0.382 | 0.0037 | 0.117 | - | - | 0.900 | 0.862 | 0.883 | 0.852 | 0.813 |
| zlin256_t10_vis_inb | r2 | 0.580 | 0.449 | 0.722 | 0.427 | 0.393 | 0.0034 | 0.118 | - | - | 0.905 | 0.868 | 0.876 | 0.844 | 0.813 |
| lin256_t20_ge6_inb | r2 | 0.563 | 0.435 | 0.714 | 0.423 | 0.391 | 0.0037 | 0.112 | - | - | 0.919 | 0.873 | 0.862 | 0.856 | 0.797 |
| lin256_t10_ge6_inb_s18 | r2 | 0.576 | 0.444 | 0.720 | 0.427 | 0.381 | 0.0037 | 0.116 | - | - | 0.900 | 0.860 | 0.881 | 0.851 | 0.811 |

Predeclared comparison: best learned minus best untrained (each picked on that metric).

| metric | rounds | best learned | best untrained | difference | margin | met |
|---|---|---|---|---|---|---|
| vis_all | r1 | lin256_t05_vis_inb | zca10 | +0.0183 | 0.05 | no |
| vis_all | r1+r2 | zlin256_t10_vis_inb | zca10 | +0.0217 | 0.05 | no |
| vis_other | r1 | lin256_t10_ge6_inb | zca10 | +0.0116 | 0.05 | no |
| vis_other | r1+r2 | zlin256_t10_vis_inb | zca10 | +0.0162 | 0.05 | no |
| tm_in_phylum | r1 | lin256_t10_ge6_inb | zca10 | -0.0065 | 0.03 | no |
| tm_in_phylum | r1+r2 | lin256_t20_ge6_inb | zca10 | +0.0119 | 0.03 | no |
| tm_other_phylum | r1 | lin256_t10_ge6_inb | zca10 | -0.0164 | 0.03 | no |
| tm_other_phylum | r1+r2 | lin256_t20_ge6_inb | zca10 | -0.0050 | 0.03 | no |

- r1: held-out genomes criterion not met; held-out phyla criterion not met; gain = NO.
- r1+r2: held-out genomes criterion not met; held-out phyla criterion not met; gain = NO.

## esm2_650m

Retrieval: exact top-60 per held-out query. vis = fraction sequence-visible (E <= 1e-3); qany = queries with >= 1 visible hit; strepinv = share of retrieved pairs to the Streptomyces reference genome that are invisible; inv TM = fraction TM>=0.5 among a hash sample of invisible retrieved pairs. AUROCs on held-out pairs; TM = sequence-invisible in-block pairs, label TM>=0.5.

| arm | round | vis all | vis other | qany other | ident vis all | other-phylum share (all) | strepinv | SD recall all | inv TM all | inv TM other | TM AUROC in | TM AUROC other | popA in | popA other | xphy other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | untrained | 0.612 | 0.409 | 0.732 | 0.436 | 0.178 | 0.0116 | 0.128 | - | - | 0.783 | 0.709 | 0.804 | 0.663 | 0.850 |
| gcenter | untrained | 0.613 | 0.405 | 0.724 | 0.439 | 0.161 | 0.0153 | 0.128 | - | - | 0.831 | 0.692 | 0.805 | 0.564 | 0.888 |
| genomecenter | untrained | 0.642 | 0.488 | 0.742 | 0.431 | 0.257 | 0.0089 | 0.132 | - | - | 0.839 | 0.705 | 0.781 | 0.581 | 0.883 |
| csls2 | untrained | 0.614 | 0.433 | 0.739 | 0.432 | 0.244 | 0.0086 | 0.127 | - | - | 0.805 | 0.769 | 0.750 | 0.649 | 0.849 |
| abtt2 | untrained | 0.640 | 0.463 | 0.746 | 0.436 | 0.189 | 0.0160 | 0.132 | - | - | 0.886 | 0.849 | 0.888 | 0.780 | 0.891 |
| abtt10 | untrained | 0.670 | 0.520 | 0.764 | 0.431 | 0.259 | 0.0118 | 0.136 | - | - | 0.896 | 0.889 | 0.907 | 0.861 | 0.919 |
| abtt30 | untrained | 0.703 | 0.561 | 0.777 | 0.429 | 0.307 | 0.0066 | 0.142 | - | - | 0.919 | 0.913 | 0.938 | 0.912 | 0.932 |
| zca10 | untrained | 0.726 | 0.588 | 0.787 | 0.434 | 0.290 | 0.0053 | 0.149 | - | - | 0.912 | 0.914 | 0.948 | 0.933 | 0.938 |
| pca256 | untrained | 0.573 | 0.350 | 0.698 | 0.443 | 0.151 | 0.0172 | 0.122 | - | - | 0.819 | 0.670 | 0.758 | 0.512 | 0.875 |
| csls_q | untrained | - | - | - | - | - | - | - | - | - | 0.798 | 0.723 | 0.651 | 0.438 | 0.836 |
| z_q | untrained | - | - | - | - | - | - | - | - | - | 0.796 | 0.800 | 0.620 | 0.485 | 0.661 |
| rank_q | untrained | - | - | - | - | - | - | - | - | - | - | - | 0.611 | 0.491 | - |
| lin256_t05_ge6_inb | r1 | 0.703 | 0.518 | 0.770 | 0.436 | 0.265 | 0.0043 | 0.145 | - | - | 0.840 | 0.840 | 0.911 | 0.869 | 0.898 |
| lin64_t05_ge6_inb | r1 | 0.678 | 0.469 | 0.754 | 0.438 | 0.255 | 0.0046 | 0.142 | - | - | 0.828 | 0.814 | 0.907 | 0.849 | 0.887 |
| lin128_t05_ge6_inb | r1 | 0.696 | 0.505 | 0.766 | 0.436 | 0.260 | 0.0044 | 0.145 | - | - | 0.838 | 0.834 | 0.910 | 0.864 | 0.896 |
| lin512_t05_ge6_inb | r1 | 0.705 | 0.522 | 0.771 | 0.436 | 0.267 | 0.0043 | 0.146 | - | - | 0.841 | 0.842 | 0.911 | 0.871 | 0.899 |
| mlp256_t05_ge6_inb | r1 | 0.693 | 0.501 | 0.765 | 0.436 | 0.260 | 0.0045 | 0.144 | - | - | 0.837 | 0.832 | 0.911 | 0.862 | 0.894 |
| lin256_t03_ge6_inb | r1 | 0.690 | 0.470 | 0.758 | 0.437 | 0.226 | 0.0049 | 0.143 | - | - | 0.822 | 0.813 | 0.911 | 0.842 | 0.836 |
| lin256_t10_ge6_inb | r1 | 0.720 | 0.566 | 0.784 | 0.435 | 0.293 | 0.0033 | 0.149 | - | - | 0.871 | 0.859 | 0.925 | 0.890 | 0.915 |
| lin256_t05_acc05_inb | r1 | 0.664 | 0.455 | 0.755 | 0.435 | 0.192 | 0.0070 | 0.137 | - | - | 0.840 | 0.820 | 0.924 | 0.845 | 0.842 |
| lin256_t05_vis_inb | r1 | 0.720 | 0.554 | 0.780 | 0.435 | 0.312 | 0.0032 | 0.149 | - | - | 0.875 | 0.864 | 0.910 | 0.871 | 0.899 |
| lin256_t05_ge6_hard | r1 | 0.703 | 0.518 | 0.769 | 0.436 | 0.265 | 0.0043 | 0.145 | - | - | 0.840 | 0.840 | 0.911 | 0.869 | 0.898 |

Predeclared comparison: best learned minus best untrained (each picked on that metric).

| metric | rounds | best learned | best untrained | difference | margin | met |
|---|---|---|---|---|---|---|
| vis_all | r1 | lin256_t10_ge6_inb | zca10 | -0.0061 | 0.05 | no |
| vis_all | r1+r2 | lin256_t10_ge6_inb | zca10 | -0.0061 | 0.05 | no |
| vis_other | r1 | lin256_t10_ge6_inb | zca10 | -0.0219 | 0.05 | no |
| vis_other | r1+r2 | lin256_t10_ge6_inb | zca10 | -0.0219 | 0.05 | no |
| tm_in_phylum | r1 | lin256_t05_vis_inb | abtt30 | -0.0441 | 0.03 | no |
| tm_in_phylum | r1+r2 | lin256_t05_vis_inb | abtt30 | -0.0441 | 0.03 | no |
| tm_other_phylum | r1 | lin256_t05_vis_inb | zca10 | -0.0502 | 0.03 | no |
| tm_other_phylum | r1+r2 | lin256_t05_vis_inb | zca10 | -0.0502 | 0.03 | no |

- r1: held-out genomes criterion not met; held-out phyla criterion not met; gain = NO.
- r1+r2: held-out genomes criterion not met; held-out phyla criterion not met; gain = NO.

## esmc600m

Retrieval: exact top-60 per held-out query. vis = fraction sequence-visible (E <= 1e-3); qany = queries with >= 1 visible hit; strepinv = share of retrieved pairs to the Streptomyces reference genome that are invisible; inv TM = fraction TM>=0.5 among a hash sample of invisible retrieved pairs. AUROCs on held-out pairs; TM = sequence-invisible in-block pairs, label TM>=0.5.

| arm | round | vis all | vis other | qany other | ident vis all | other-phylum share (all) | strepinv | SD recall all | inv TM all | inv TM other | TM AUROC in | TM AUROC other | popA in | popA other | xphy other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | untrained | 0.388 | 0.143 | 0.550 | 0.502 | 0.101 | 0.0379 | 0.100 | - | - | 0.640 | 0.572 | 0.801 | 0.608 | 0.777 |
| gcenter | untrained | 0.392 | 0.142 | 0.539 | 0.504 | 0.089 | 0.0484 | 0.101 | - | - | 0.642 | 0.526 | 0.764 | 0.488 | 0.818 |
| genomecenter | untrained | 0.432 | 0.244 | 0.635 | 0.487 | 0.230 | 0.0248 | 0.107 | - | - | 0.707 | 0.558 | 0.723 | 0.459 | 0.813 |
| csls2 | untrained | 0.401 | 0.170 | 0.585 | 0.497 | 0.154 | 0.0299 | 0.103 | - | - | 0.677 | 0.618 | 0.770 | 0.643 | 0.775 |
| abtt2 | untrained | 0.465 | 0.218 | 0.600 | 0.490 | 0.116 | 0.0567 | 0.115 | - | - | 0.808 | 0.717 | 0.810 | 0.641 | 0.771 |
| abtt10 | untrained | 0.498 | 0.285 | 0.654 | 0.482 | 0.126 | 0.0601 | 0.121 | - | - | 0.793 | 0.754 | 0.868 | 0.774 | 0.840 |
| abtt30 | untrained | 0.560 | 0.390 | 0.697 | 0.471 | 0.180 | 0.0666 | 0.130 | - | - | 0.815 | 0.805 | 0.900 | 0.862 | 0.882 |
| zca10 | untrained | 0.651 | 0.514 | 0.756 | 0.457 | 0.189 | 0.0501 | 0.144 | - | - | 0.859 | 0.845 | 0.954 | 0.915 | 0.923 |
| pca256 | untrained | 0.357 | 0.121 | 0.510 | 0.511 | 0.087 | 0.0488 | 0.094 | - | - | 0.639 | 0.523 | 0.734 | 0.463 | 0.812 |
| csls_q | untrained | - | - | - | - | - | - | - | - | - | 0.671 | 0.596 | 0.608 | 0.376 | 0.749 |
| z_q | untrained | - | - | - | - | - | - | - | - | - | 0.700 | 0.672 | 0.648 | 0.527 | 0.634 |
| rank_q | untrained | - | - | - | - | - | - | - | - | - | - | - | 0.641 | 0.537 | - |
| lin256_t05_ge6_inb | r1 | 0.649 | 0.410 | 0.733 | 0.456 | 0.221 | 0.0076 | 0.145 | - | - | 0.811 | 0.721 | 0.934 | 0.804 | 0.867 |
| lin64_t05_ge6_inb | r1 | 0.591 | 0.321 | 0.698 | 0.467 | 0.234 | 0.0071 | 0.139 | - | - | 0.792 | 0.696 | 0.925 | 0.765 | 0.845 |
| lin128_t05_ge6_inb | r1 | 0.635 | 0.385 | 0.723 | 0.459 | 0.223 | 0.0075 | 0.144 | - | - | 0.807 | 0.714 | 0.932 | 0.795 | 0.861 |
| lin512_t05_ge6_inb | r1 | 0.654 | 0.418 | 0.737 | 0.455 | 0.221 | 0.0077 | 0.146 | - | - | 0.812 | 0.724 | 0.935 | 0.808 | 0.870 |
| mlp256_t05_ge6_inb | r1 | 0.619 | 0.357 | 0.717 | 0.462 | 0.210 | 0.0079 | 0.142 | - | - | 0.802 | 0.696 | 0.930 | 0.772 | 0.850 |
| lin256_t03_ge6_inb | r1 | 0.631 | 0.358 | 0.718 | 0.460 | 0.186 | 0.0090 | 0.143 | - | - | 0.793 | 0.711 | 0.937 | 0.828 | 0.850 |
| lin256_t10_ge6_inb | r1 | 0.700 | 0.525 | 0.771 | 0.447 | 0.240 | 0.0049 | 0.151 | - | - | 0.845 | 0.745 | 0.944 | 0.779 | 0.888 |
| lin256_t05_acc05_inb | r1 | 0.612 | 0.360 | 0.719 | 0.461 | 0.159 | 0.0127 | 0.138 | - | - | 0.784 | 0.704 | 0.958 | 0.826 | 0.861 |
| lin256_t05_vis_inb | r1 | 0.697 | 0.494 | 0.767 | 0.448 | 0.253 | 0.0049 | 0.151 | - | - | 0.869 | 0.802 | 0.957 | 0.863 | 0.875 |
| lin256_t05_ge6_hard | r1 | 0.649 | 0.410 | 0.733 | 0.456 | 0.221 | 0.0076 | 0.145 | - | - | 0.811 | 0.722 | 0.935 | 0.806 | 0.868 |
| lin256_t10_vis_inb | r2 | 0.730 | 0.570 | 0.789 | 0.443 | 0.272 | 0.0036 | 0.154 | - | - | 0.893 | 0.823 | 0.971 | 0.890 | 0.906 |
| zlin256_t10_ge6_inb | r2 | 0.705 | 0.533 | 0.774 | 0.447 | 0.247 | 0.0045 | 0.152 | - | - | 0.845 | 0.756 | 0.945 | 0.814 | 0.901 |
| zlin256_t10_vis_inb | r2 | 0.735 | 0.578 | 0.790 | 0.442 | 0.278 | 0.0035 | 0.155 | - | - | 0.896 | 0.842 | 0.974 | 0.922 | 0.912 |
| lin256_t20_ge6_inb | r2 | 0.699 | 0.532 | 0.773 | 0.446 | 0.252 | 0.0045 | 0.150 | - | - | 0.858 | 0.734 | 0.938 | 0.724 | 0.888 |
| lin256_t10_ge6_inb_s18 | r2 | 0.700 | 0.524 | 0.771 | 0.447 | 0.240 | 0.0049 | 0.151 | - | - | 0.846 | 0.744 | 0.944 | 0.780 | 0.891 |

Predeclared comparison: best learned minus best untrained (each picked on that metric).

| metric | rounds | best learned | best untrained | difference | margin | met |
|---|---|---|---|---|---|---|
| vis_all | r1 | lin256_t10_ge6_inb | zca10 | +0.0489 | 0.05 | no |
| vis_all | r1+r2 | zlin256_t10_vis_inb | zca10 | +0.0843 | 0.05 | yes |
| vis_other | r1 | lin256_t10_ge6_inb | zca10 | +0.0111 | 0.05 | no |
| vis_other | r1+r2 | zlin256_t10_vis_inb | zca10 | +0.0646 | 0.05 | yes |
| tm_in_phylum | r1 | lin256_t05_vis_inb | zca10 | +0.0100 | 0.03 | no |
| tm_in_phylum | r1+r2 | zlin256_t10_vis_inb | zca10 | +0.0370 | 0.03 | yes |
| tm_other_phylum | r1 | lin256_t05_vis_inb | zca10 | -0.0428 | 0.03 | no |
| tm_other_phylum | r1+r2 | zlin256_t10_vis_inb | zca10 | -0.0027 | 0.03 | no |

- r1: held-out genomes criterion not met; held-out phyla criterion not met; gain = NO.
- r1+r2: held-out genomes criterion met; held-out phyla criterion met; gain = YES.

## esmc6b

Retrieval: exact top-60 per held-out query. vis = fraction sequence-visible (E <= 1e-3); qany = queries with >= 1 visible hit; strepinv = share of retrieved pairs to the Streptomyces reference genome that are invisible; inv TM = fraction TM>=0.5 among a hash sample of invisible retrieved pairs. AUROCs on held-out pairs; TM = sequence-invisible in-block pairs, label TM>=0.5.

| arm | round | vis all | vis other | qany other | ident vis all | other-phylum share (all) | strepinv | SD recall all | inv TM all | inv TM other | TM AUROC in | TM AUROC other | popA in | popA other | xphy other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | untrained | 0.320 | 0.137 | 0.544 | 0.501 | 0.059 | 0.0706 | 0.082 | - | - | 0.629 | 0.570 | 0.763 | 0.454 | 0.779 |
| gcenter | untrained | 0.317 | 0.132 | 0.531 | 0.503 | 0.054 | 0.0841 | 0.082 | - | - | 0.686 | 0.587 | 0.720 | 0.376 | 0.776 |
| genomecenter | untrained | 0.374 | 0.216 | 0.614 | 0.480 | 0.251 | 0.0289 | 0.092 | - | - | 0.756 | 0.629 | 0.728 | 0.385 | 0.785 |
| csls2 | untrained | 0.342 | 0.172 | 0.585 | 0.496 | 0.122 | 0.0501 | 0.088 | - | - | 0.701 | 0.640 | 0.755 | 0.572 | 0.773 |
| abtt2 | untrained | 0.367 | 0.184 | 0.583 | 0.497 | 0.076 | 0.0909 | 0.093 | - | - | 0.800 | 0.722 | 0.772 | 0.486 | 0.739 |
| abtt10 | untrained | 0.395 | 0.223 | 0.615 | 0.496 | 0.073 | 0.0984 | 0.098 | - | - | 0.771 | 0.742 | 0.841 | 0.614 | 0.832 |
| abtt30 | untrained | 0.454 | 0.332 | 0.684 | 0.488 | 0.108 | 0.1123 | 0.109 | - | - | 0.805 | 0.804 | 0.864 | 0.743 | 0.865 |
| zca10 | untrained | 0.609 | 0.519 | 0.763 | 0.466 | 0.153 | 0.0897 | 0.138 | - | - | 0.858 | 0.848 | 0.925 | 0.835 | 0.930 |
| pca256 | untrained | 0.273 | 0.106 | 0.493 | 0.506 | 0.054 | 0.0814 | 0.071 | - | - | 0.683 | 0.582 | 0.677 | 0.340 | 0.765 |
| csls_q | untrained | - | - | - | - | - | - | - | - | - | 0.688 | 0.618 | 0.560 | 0.279 | 0.722 |
| z_q | untrained | - | - | - | - | - | - | - | - | - | 0.713 | 0.684 | 0.641 | 0.515 | 0.626 |
| rank_q | untrained | - | - | - | - | - | - | - | - | - | - | - | 0.638 | 0.526 | - |
| lin256_t05_ge6_inb | r1 | 0.577 | 0.304 | 0.679 | 0.472 | 0.209 | 0.0123 | 0.140 | - | - | 0.823 | 0.740 | 0.894 | 0.647 | 0.855 |
| lin64_t05_ge6_inb | r1 | 0.476 | 0.199 | 0.599 | 0.495 | 0.273 | 0.0101 | 0.130 | - | - | 0.808 | 0.719 | 0.885 | 0.619 | 0.844 |
| lin128_t05_ge6_inb | r1 | 0.546 | 0.267 | 0.657 | 0.479 | 0.228 | 0.0117 | 0.137 | - | - | 0.818 | 0.735 | 0.892 | 0.637 | 0.849 |
| lin512_t05_ge6_inb | r1 | 0.587 | 0.318 | 0.689 | 0.470 | 0.201 | 0.0126 | 0.141 | - | - | 0.825 | 0.744 | 0.895 | 0.651 | 0.856 |
| mlp256_t05_ge6_inb | r1 | 0.512 | 0.228 | 0.635 | 0.487 | 0.195 | 0.0135 | 0.133 | - | - | 0.814 | 0.715 | 0.889 | 0.612 | 0.832 |
| lin256_t03_ge6_inb | r1 | 0.524 | 0.237 | 0.641 | 0.484 | 0.234 | 0.0120 | 0.135 | - | - | 0.809 | 0.736 | 0.887 | 0.677 | 0.843 |
| lin256_t10_ge6_inb | r1 | 0.667 | 0.471 | 0.756 | 0.454 | 0.184 | 0.0086 | 0.149 | - | - | 0.848 | 0.755 | 0.920 | 0.621 | 0.878 |
| lin256_t05_acc05_inb | r1 | 0.566 | 0.320 | 0.693 | 0.471 | 0.160 | 0.0188 | 0.136 | - | - | 0.816 | 0.741 | 0.917 | 0.677 | 0.851 |
| lin256_t05_vis_inb | r1 | 0.668 | 0.448 | 0.757 | 0.453 | 0.209 | 0.0078 | 0.148 | - | - | 0.871 | 0.807 | 0.938 | 0.767 | 0.863 |
| lin256_t05_ge6_hard | r1 | 0.577 | 0.304 | 0.679 | 0.472 | 0.209 | 0.0122 | 0.140 | - | - | 0.824 | 0.741 | 0.895 | 0.653 | 0.856 |
| lin256_t10_vis_inb | r2 | 0.716 | 0.546 | 0.787 | 0.446 | 0.226 | 0.0053 | 0.153 | - | - | 0.899 | 0.837 | 0.963 | 0.796 | 0.899 |
| zlin256_t10_ge6_inb | r2 | 0.671 | 0.478 | 0.757 | 0.454 | 0.182 | 0.0090 | 0.150 | - | - | 0.846 | 0.774 | 0.918 | 0.676 | 0.898 |
| zlin256_t10_vis_inb | r2 | 0.726 | 0.563 | 0.790 | 0.444 | 0.231 | 0.0052 | 0.155 | - | - | 0.903 | 0.862 | 0.966 | 0.870 | 0.911 |
| lin256_t20_ge6_inb | r2 | 0.673 | 0.488 | 0.765 | 0.451 | 0.191 | 0.0063 | 0.149 | - | - | 0.867 | 0.766 | 0.936 | 0.634 | 0.878 |
| lin256_t10_ge6_inb_s18 | r2 | 0.668 | 0.473 | 0.757 | 0.454 | 0.184 | 0.0085 | 0.149 | - | - | 0.848 | 0.754 | 0.921 | 0.618 | 0.877 |

Predeclared comparison: best learned minus best untrained (each picked on that metric).

| metric | rounds | best learned | best untrained | difference | margin | met |
|---|---|---|---|---|---|---|
| vis_all | r1 | lin256_t05_vis_inb | zca10 | +0.0585 | 0.05 | yes |
| vis_all | r1+r2 | zlin256_t10_vis_inb | zca10 | +0.1171 | 0.05 | yes |
| vis_other | r1 | lin256_t10_ge6_inb | zca10 | -0.0480 | 0.05 | no |
| vis_other | r1+r2 | zlin256_t10_vis_inb | zca10 | +0.0438 | 0.05 | no |
| tm_in_phylum | r1 | lin256_t05_vis_inb | zca10 | +0.0134 | 0.03 | no |
| tm_in_phylum | r1+r2 | zlin256_t10_vis_inb | zca10 | +0.0457 | 0.03 | yes |
| tm_other_phylum | r1 | lin256_t05_vis_inb | zca10 | -0.0415 | 0.03 | no |
| tm_other_phylum | r1+r2 | zlin256_t10_vis_inb | zca10 | +0.0136 | 0.03 | no |

- r1: held-out genomes criterion met; held-out phyla criterion not met; gain = NO.
- r1+r2: held-out genomes criterion met; held-out phyla criterion not met; gain = NO.

