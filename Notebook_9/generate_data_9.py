"""
generate_data_9.py
------------------
Generates a synthetic pulsar candidate classification dataset for Challenge 9.

Binary classification:
  0 = Non-pulsar  (91%): RFI and noise candidates
  1 = Pulsar       (9%): genuine rotating neutron star signals

Total samples: 10,000
Dataset saved to: data/radio.csv

Real-world context
------------------
Pulsar candidate classification is an imbalanced binary problem. In the
Parkes High Time Resolution Universe (HTRU) survey, approximately 1% of
processed candidates are genuine pulsars; the rest are RFI or noise.
The HTRU-2 benchmark dataset (Lyon et al. 2016) contains 17,898 candidates
with 1,639 genuine pulsars (~9%), and is one of the most widely used
benchmarks for evaluating ML methods in radio astronomy.

Each candidate is described by statistics of its folded pulse profile
and its DM-SNR curve. The challenge is that many RFI signals produce
folded profiles that superficially resemble real pulsars. The 8 statistical
features used here are the exact features used in the real HTRU-2 dataset.

Feature design: 20 features
----------------------------
8 SIGNAL features (direct equivalents of HTRU-2 features):
  prof_mean:         Mean of the integrated folded profile
  prof_std:          Standard deviation of the integrated profile
  prof_skewness:     Excess kurtosis of the integrated profile
                     (named skewness for historical reasons in the dataset)
  prof_kurtosis:     Skewness of the integrated profile
  dm_snr_mean:       Mean of the DM-SNR curve
  dm_snr_std:        Standard deviation of the DM-SNR curve
  dm_snr_skewness:   Excess kurtosis of the DM-SNR curve
  dm_snr_kurtosis:   Skewness of the DM-SNR curve

12 ADDITIONAL features (correlated noisy extensions of the 8 signal features,
simulating the additional pipeline scores computed by modern survey software):
  subband_01 to subband_04: sub-band profile statistics
                             (correlated copies of profile features + noise)
  subband_05 to subband_08: sub-band DM statistics
                             (correlated copies of DM-SNR features + noise)
  pipeline_01 to pipeline_04: higher-order pipeline diagnostic scores
                              (weak linear combinations of all features + noise)

This creates a realistic correlation structure where:
  - The 8 signal features are moderately correlated (by class structure)
  - The 12 additional features are correlated with the signal features
    but add mostly noise
  - PCA finds that 2-3 components explain ~60-70% of variance and that
    the first principal component nearly separates the two classes
  - This motivates the PCA approach: work in the reduced space

Pedagogical goals:
  1. The scree plot shows a clear elbow at 2-3 components
  2. A 2D PCA scatter plot clearly shows the pulsar/non-pulsar separation
  3. MLP on PCA 2 components achieves ~90-92% accuracy with very fast training
  4. MLP on all 20 raw features achieves ~93-95% accuracy but trains much
     slower and is harder to interpret
  5. The key lesson: PCA enables visualisation and interpretation; the
     accuracy tradeoff between 2 PCA components and 20 raw features is small,
     and choosing the number of components is a principled tradeoff

Class distribution reflects the real HTRU-2 survey:
  Non-pulsar: 9,100 samples (91%)
  Pulsar:       900 samples ( 9%)

Label noise: 2% (a small fraction of RFI signals truly mimic pulsars;
some genuine pulsars have degraded data quality).
"""

import numpy as np
import pandas as pd
import os

RNG = np.random.default_rng(seed=55)

N_TOTAL      = 10_000
N_NON_PULSAR = 9_100
N_PULSAR     =   900

SIGNAL_FEATURES = [
    'prof_mean', 'prof_std', 'prof_skewness', 'prof_kurtosis',
    'dm_snr_mean', 'dm_snr_std', 'dm_snr_skewness', 'dm_snr_kurtosis'
]
EXTRA_FEATURES = (
    [f'subband_{i:02d}' for i in range(1, 9)] +
    [f'pipeline_{i:02d}' for i in range(1, 5)]
)
FEATURE_NAMES = SIGNAL_FEATURES + EXTRA_FEATURES

CLASS_NAMES = ['non_pulsar', 'pulsar']


def make_non_pulsar(n, rng):
    """
    RFI and noise candidates. These dominate the dataset. They are
    heterogeneous: some are narrowband RFI that partially mimics pulsars
    in profile shape, others are broadband or noise-like. The class has
    wide, overlapping distributions with genuine pulsars in some features.
    """
    # Profile statistics: more variable, often broader
    prof_mean     = rng.normal(0.0,  1.0, n)
    prof_std      = rng.normal(0.0,  0.9, n)
    prof_skewness = rng.normal(0.0,  1.0, n)
    prof_kurtosis = rng.normal(0.0,  1.0, n)
    # DM-SNR statistics: flat or noisy curves, low kurtosis
    dm_snr_mean     = rng.normal(0.0,  0.8, n)
    dm_snr_std      = rng.normal(0.0,  0.9, n)
    dm_snr_skewness = rng.normal(0.0,  0.9, n)
    dm_snr_kurtosis = rng.normal(0.0,  0.9, n)
    return np.column_stack([prof_mean, prof_std, prof_skewness, prof_kurtosis,
                             dm_snr_mean, dm_snr_std, dm_snr_skewness, dm_snr_kurtosis])


def make_pulsar(n, rng):
    """
    Genuine pulsars. Shifted in latent score space: higher dm_snr_kurtosis
    (sharp DM peak), higher prof_kurtosis (narrow pulse), lower dm_snr_std
    (concentrated DM-SNR curve). The distributions overlap with non-pulsars
    at their edges, reflecting the genuine difficulty of the problem.
    """
    prof_mean     = rng.normal(0.6,  0.9, n)   # slightly elevated
    prof_std      = rng.normal(0.3,  0.8, n)
    prof_skewness = rng.normal(1.2,  0.9, n)   # more skewed profiles
    prof_kurtosis = rng.normal(2.5,  1.2, n)   # high: narrow pulse
    # DM-SNR: sharp peak at true DM
    dm_snr_mean     = rng.normal( 0.8, 0.8, n)
    dm_snr_std      = rng.normal(-0.8, 0.8, n)  # lower std: concentrated
    dm_snr_skewness = rng.normal( 1.0, 0.9, n)  # positive: asymmetric peak
    dm_snr_kurtosis = rng.normal( 3.5, 1.4, n)  # high: sharp kurtosis
    return np.column_stack([prof_mean, prof_std, prof_skewness, prof_kurtosis,
                             dm_snr_mean, dm_snr_std, dm_snr_skewness, dm_snr_kurtosis])


def add_extra_features(signal_arr, rng):
    """
    12 additional features: correlated noisy copies of the signal features.

    subband_01-04: noisy copies of profile features (profile shape in
                   different frequency sub-bands of the receiver)
    subband_05-08: noisy copies of DM-SNR features (DM curve in sub-bands)
    pipeline_01-04: weak linear combinations of all 8 features + noise,
                    representing higher-order diagnostic scores
    """
    n = signal_arr.shape[0]
    extras = []

    # Sub-band profile scores: profile features + noise
    for i in range(4):
        feat_idx = i % 4   # cycle through prof_mean, prof_std, prof_skew, prof_kurt
        extras.append(signal_arr[:, feat_idx] + rng.normal(0, 0.8, n))

    # Sub-band DM scores: DM-SNR features + noise
    for i in range(4):
        feat_idx = 4 + (i % 4)  # dm_snr_mean, dm_snr_std, dm_snr_skew, dm_snr_kurt
        extras.append(signal_arr[:, feat_idx] + rng.normal(0, 0.8, n))

    # Pipeline scores: weak random combinations + noise
    for _ in range(4):
        weights = rng.normal(0, 0.3, 8)
        pipeline_score = signal_arr @ weights + rng.normal(0, 1.2, n)
        extras.append(pipeline_score)

    return np.column_stack(extras)


def build_dataframe():
    # Non-pulsar
    signal_np = make_non_pulsar(N_NON_PULSAR, RNG)
    extra_np  = add_extra_features(signal_np, RNG)
    labels_np = np.zeros(N_NON_PULSAR, dtype=int)
    # 2% label noise
    flip = RNG.random(N_NON_PULSAR) < 0.02
    labels_np[flip] = 1

    # Pulsar
    signal_p = make_pulsar(N_PULSAR, RNG)
    extra_p  = add_extra_features(signal_p, RNG)
    labels_p = np.ones(N_PULSAR, dtype=int)
    flip2 = RNG.random(N_PULSAR) < 0.02
    labels_p[flip2] = 0

    X = np.vstack([
        np.hstack([signal_np, extra_np]),
        np.hstack([signal_p, extra_p])
    ])
    y = np.concatenate([labels_np, labels_p])

    perm = RNG.permutation(len(y))
    X, y = X[perm], y[perm]

    df = pd.DataFrame(X, columns=FEATURE_NAMES).round(6)
    df['candidate_type'] = y.astype(int)
    return df


def main():
    os.makedirs('data', exist_ok=True)
    df = build_dataframe()
    df.to_csv('data/radio.csv', index=False)

    print(f'Dataset saved to: data/radio.csv')
    print(f'Total rows:   {len(df)}')
    print(f'Features:     {len(FEATURE_NAMES)} ({len(SIGNAL_FEATURES)} signal + {len(EXTRA_FEATURES)} extra)')

    print('\nClass distribution:')
    for k, v in df['candidate_type'].value_counts().sort_index().items():
        print(f'  Class {k} ({CLASS_NAMES[k]:12s}): {v:5d}  ({100*v/len(df):.1f}%)')

    # PCA analysis
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    import warnings; warnings.filterwarnings('ignore')

    X_vals = df[FEATURE_NAMES].values
    y_vals = df['candidate_type'].values
    Xtr,Xte,ytr,yte = train_test_split(X_vals, y_vals, test_size=0.2,
                                        random_state=42, stratify=y_vals)
    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(Xtr); Xte_sc = scaler.transform(Xte)

    pca_full = PCA(random_state=42).fit(Xtr_sc)
    cumvar   = np.cumsum(pca_full.explained_variance_ratio_)
    print('\nCumulative explained variance by PCA:')
    for i, cv in enumerate(cumvar[:10], 1):
        print(f'  PC{i:2d}: {cv:.4f}  ({pca_full.explained_variance_ratio_[i-1]:.4f})')

    print('\nMLP (64,32) accuracy and macro-F1 sweep:')
    arch = (64, 32)

    mlp = MLPClassifier(hidden_layer_sizes=arch, max_iter=500,
                         random_state=42, early_stopping=True)
    mlp.fit(Xtr_sc, ytr); yp = mlp.predict(Xte_sc)
    print(f'  Raw 20 features: acc={accuracy_score(yte,yp):.4f}  F1={f1_score(yte,yp,average="macro"):.4f}  iters={mlp.n_iter_}')
    print(classification_report(yte, yp, target_names=CLASS_NAMES))

    for nc in [2, 3, 4, 5, 6, 8]:
        pca_n = PCA(n_components=nc, random_state=42)
        Xp = pca_n.fit_transform(Xtr_sc); Xpt = pca_n.transform(Xte_sc)
        mlp_p = MLPClassifier(hidden_layer_sizes=arch, max_iter=500,
                               random_state=42, early_stopping=True)
        mlp_p.fit(Xp, ytr); yp = mlp_p.predict(Xpt)
        print(f'  PCA n={nc:2d}: acc={accuracy_score(yte,yp):.4f}  F1={f1_score(yte,yp,average="macro"):.4f}  '
              f'iters={mlp_p.n_iter_}  var={cumvar[nc-1]:.3f}')


if __name__ == '__main__':
    main()
