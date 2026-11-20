"""Patient-separated, fixed-landmark readmission prediction. See README.md.

Requires Python >=3.10 and scikit-learn >=1.3. All timestamps are converted
to UTC; localize naive local timestamps before calling if necessary.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MEASUREMENTS = {
    'weight [kg]': 'weight', 'spo2 [%]': 'spo2',
    'spo2 hr [bpm]': 'hr_spo2', 'syst bp [mmHg]': 'sbp',
    'diast bp [mmHg]': 'dbp', 'map [mmHg]': 'map',
    'bpt hr [bpm]': 'hr_bp',
}
HISTORY = ['n_admissions_to_date', 'last_length_of_stay_days']
METADATA = ['pseudopaz', 'prediction_time', 'outcome_end', 'readmitted']


def _timestamps(values):
    return pd.to_datetime(values, errors='raise', utc=True)


def build_dataset(df_hospit, df_medical, *, followup_end=None,
                  df_followup=None, landmark_days=7, horizon_days=30,
                  lookback_days=7):
    """Return one row per eligible discharge, using ONLY past measurements.

    Supply EITHER an administrative followup_end (same ascertainment for all)
    OR df_followup with pseudopaz, observation_end, optional death_date.
    observation_end means reliable hospitalization ascertainment, NOT the last
    measurement. Require complete administrative coverage of the full horizon
    for ALL rows, even if readmission occurs early, to avoid preferentially
    retaining positive outcomes with short follow-up.

    Deaths before/on the landmark are excluded. If a known death occurs after
    the landmark but before readmission, the binary outcome is zero: the target
    is readmission before death within the horizon, not readmission-or-death.
    With no death dates, reliable survival/ascertainment must be assumed.

    landmark_days=0 uses inpatient measurements through discharge. Positive
    landmarks use outpatient measurements strictly after discharge only.
    """
    if landmark_days < 0 or horizon_days <= 0 or lookback_days <= 0:
        raise ValueError('Require landmark_days >= 0 and positive windows.')
    if (followup_end is None) == (df_followup is None):
        raise ValueError('Supply exactly one of followup_end or df_followup.')
    required_h = {'pseudopaz', 'data_apertura', 'data_chiusura'}
    required_m = {'pseudopaz', 'datetime', *MEASUREMENTS}
    if not required_h.issubset(df_hospit) or not required_m.issubset(df_medical):
        raise ValueError('Missing required dataframe columns; see README.')
    h = df_hospit[list(required_h)].copy()
    m = df_medical[['pseudopaz', 'datetime', *MEASUREMENTS]].copy()
    if h.pseudopaz.isna().any() or m.pseudopaz.isna().any():
        raise ValueError('Missing patient identifiers.')
    h['data_apertura'] = _timestamps(h.data_apertura)
    h['data_chiusura'] = _timestamps(h.data_chiusura)
    m['datetime'] = _timestamps(m.datetime)
    if h.data_apertura.isna().any() or m.datetime.isna().any():
        raise ValueError('Missing admission or measurement timestamps.')
    if (h.data_chiusura < h.data_apertura).any():
        raise ValueError('Discharge before admission.')
    h = h.drop_duplicates().sort_values(['pseudopaz', 'data_apertura'])
    # Reject, rather than silently decide whether overlaps are transfers.
    for _, g in h.groupby('pseudopaz', sort=False):
        starts, ends = g.data_apertura.tolist(), g.data_chiusura.tolist()
        for j in range(len(g) - 1):
            if pd.isna(ends[j]) or starts[j + 1] <= ends[j]:
                raise ValueError('Overlapping/touching episodes or nonfinal open episode. '
                                 'Resolve transfers and overlaps first.')
    for col in MEASUREMENTS:
        m[col] = pd.to_numeric(m[col], errors='raise').replace([np.inf, -np.inf], np.nan)
    # Only logically impossible values are removed here. Clinical plausibility
    # limits must be reviewed with the data owner; no arbitrary winsorization.
    m.loc[(m['spo2 [%]'] < 0) | (m['spo2 [%]'] > 100), 'spo2 [%]'] = np.nan
    for col in set(MEASUREMENTS) - {'spo2 [%]'}:
        m.loc[m[col] <= 0, col] = np.nan
    m = m.drop_duplicates().sort_values('datetime')
    if df_followup is None:
        end = _timestamps(followup_end)
        if pd.isna(end) or not isinstance(end, pd.Timestamp):
            raise ValueError('followup_end must be one valid timestamp.')
        f = pd.DataFrame({'pseudopaz': h.pseudopaz.unique(), 'observation_end': end})
    else:
        f = df_followup.copy()
        if not {'pseudopaz', 'observation_end'}.issubset(f):
            raise ValueError('df_followup requires pseudopaz and observation_end.')
        if f.pseudopaz.isna().any() or f.pseudopaz.duplicated().any():
            raise ValueError('Follow-up requires one nonmissing ID per patient.')
        f['observation_end'] = _timestamps(f.observation_end)
    if f.observation_end.isna().any():
        raise ValueError('Missing observation_end.')
    if 'death_date' not in f:
        f['death_date'] = pd.NaT
    f['death_date'] = _timestamps(f.death_date)
    f = f.set_index('pseudopaz')
    if not h.pseudopaz.isin(f.index).all():
        raise ValueError('Follow-up information is missing for some patients.')
    medical_groups = dict(tuple(m.groupby('pseudopaz', sort=False)))
    rows = []
    for pid, g in h.groupby('pseudopaz', sort=False):
        g = g.reset_index(drop=True)
        obs_end, death = f.loc[pid, ['observation_end', 'death_date']]
        # Do not use information recorded after ascertainment ends.
        if pd.notna(death) and death > obs_end:
            death = pd.NaT
        if pd.notna(death) and (g.data_apertura > death).any():
            raise ValueError('Admission after recorded death; check timestamps.')
        pm = medical_groups.get(pid, m.iloc[:0])
        for i, episode in g.iterrows():
            discharge = episode.data_chiusura
            if pd.isna(discharge):
                continue  # An open stay can be an outcome, never an index discharge.
            t = discharge + pd.Timedelta(days=landmark_days)
            outcome_end = t + pd.Timedelta(days=horizon_days)
            next_admission = g.iloc[i + 1].data_apertura if i + 1 < len(g) else pd.NaT
            if (pd.notna(next_admission) and next_admission <= t):
                continue  # Already readmitted by the landmark.
            if pd.notna(death) and death <= t:
                continue
            if obs_end < outcome_end:
                continue  # Exclude ALL incompletely observed horizons.
            event = (pd.notna(next_admission) and next_admission <= outcome_end
                     and (pd.isna(death) or next_admission <= death))
            start = t - pd.Timedelta(days=lookback_days)
            mask = (pm.datetime > start) & (pm.datetime <= t)
            if landmark_days > 0:
                mask &= pm.datetime > discharge
            else:
                mask &= pm.datetime >= episode.data_apertura
            window = pm.loc[mask].copy()
            row = {'pseudopaz': pid, 'prediction_time': t,
                   'outcome_end': outcome_end, 'readmitted': int(event),
                   'n_admissions_to_date': i + 1,
                   'last_length_of_stay_days': (discharge - episode.data_apertura)
                   / pd.Timedelta(days=1)}
            # Daily medians reduce domination by high-frequency measurement days.
            # Each daily timestamp below is the last nonmissing time for that
            # specific variable, not a timestamp from a different measurement.
            for col, name in MEASUREMENTS.items():
                s = window[['datetime', col]].dropna().copy()
                s['day'] = s.datetime.dt.floor('D')
                daily = s.groupby('day').agg(value=(col, 'median'), time=('datetime', 'max'))
                n = len(daily)
                row[name + '__n_days'] = n
                row[name + '__last'] = daily.value.iloc[-1] if n else np.nan
                row[name + '__mean'] = daily.value.mean() if n else np.nan
                row[name + '__age_days'] = ((t - daily.time.iloc[-1])
                    / pd.Timedelta(days=1)) if n else np.nan
                if n >= 2:
                    x = ((daily.time - daily.time.iloc[0]) / pd.Timedelta(days=1)).to_numpy()
                    row[name + '__slope_per_day'] = np.polyfit(x, daily.value, 1)[0]
                else:
                    row[name + '__slope_per_day'] = np.nan
            rows.append(row)
    if not rows:
        raise ValueError('No eligible discharges. Check coverage and landmark settings.')
    data = pd.DataFrame(rows).reset_index(drop=True)
    data.attrs['design'] = dict(landmark_days=landmark_days, horizon_days=horizon_days,
                                lookback_days=lookback_days)
    return data


def _metrics(y, p):
    return {'auroc': roc_auc_score(y, p),
            'average_precision': average_precision_score(y, p),
            'brier': brier_score_loss(y, p)}


def fit_models(data, *, test_size=0.25, cv_splits=3, random_state=42,
               bootstrap_repeats=200):
    """Tune on development patients only; evaluate once on held-out patients.

    Model selection uses CV negative Brier score (probability quality). No
    class reweighting/SMOTE: these can distort raw risk probabilities.
    Returns models, performance, predictions, exploratory importance/coefficients,
    training-derived-bin risk summaries, and patient bootstrap metric intervals.
    """
    X = data.drop(columns=METADATA)
    y = data.readmitted.astype(int)
    groups = data.pseudopaz
    if y.nunique() != 2:
        raise ValueError('Need both readmitted and non-readmitted outcomes.')
    # Stratify patients by whether any of their eligible episodes has an event.
    patient_y = data.groupby('pseudopaz').readmitted.max()
    train_ids, test_ids = train_test_split(patient_y.index.to_numpy(),
        test_size=test_size, stratify=patient_y.to_numpy(), random_state=random_state)
    train = np.flatnonzero(groups.isin(train_ids))
    test = np.flatnonzero(groups.isin(test_ids))
    Xtr, Xte, ytr, yte = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
    gtr, gte = groups.iloc[train], groups.iloc[test]
    assert set(gtr).isdisjoint(set(gte))
    if yte.nunique() != 2:
        raise ValueError('Test set lacks a class; more patients/events are needed.')
    cv = list(StratifiedGroupKFold(n_splits=cv_splits, shuffle=True,
              random_state=random_state).split(Xtr, ytr, gtr))
    for a, b in cv:
        if ytr.iloc[a].nunique() < 2 or ytr.iloc[b].nunique() < 2:
            raise ValueError('A CV fold lacks a class. Reduce cv_splits or collect more events.')
    if ytr.sum() < 100:
        warnings.warn('Fewer than 100 training events: use fewer predictors and '
                      'interpret performance/importance cautiously. This is not a sample-size rule.')
    clinical = list(X.columns)
    # All preprocessing is fitted inside each CV training fold.
    logistic = make_pipeline(SimpleImputer(strategy='median', add_indicator=True,
                                           keep_empty_features=True),
                             StandardScaler(),
                             LogisticRegression(max_iter=3000, random_state=random_state))
    boosting = make_pipeline(SimpleImputer(strategy='median', add_indicator=True,
                                           keep_empty_features=True),
                             HistGradientBoostingClassifier(early_stopping=False,
                                 max_iter=150, random_state=random_state))
    # early_stopping=False prevents an internal row-wise validation split.
    specifications = {
        'history_logistic': (logistic, HISTORY, {'logisticregression__C': [0.01, 0.1, 1.0]}),
        'clinical_logistic': (logistic, clinical, {'logisticregression__C': [0.01, 0.1, 1.0]}),
        'clinical_boosting': (boosting, clinical,
            {'histgradientboostingclassifier__max_leaf_nodes': [3, 7],
             'histgradientboostingclassifier__l2_regularization': [1.0, 10.0]}),
    }
    fitted, scores, probabilities = {}, [], {}
    for name, (estimator, columns, grid) in specifications.items():
        search = GridSearchCV(clone(estimator), grid, cv=cv, scoring='neg_brier_score',
                              n_jobs=1, error_score='raise')
        search.fit(Xtr[columns], ytr)
        fitted[name] = {'model': search.best_estimator_, 'columns': columns}
        probabilities[name] = search.predict_proba(Xte[columns])[:, 1]
        scores.append({'model': name, 'cv_brier': -search.best_score_,
                       **_metrics(yte, probabilities[name])})
    # Choose BEFORE examining test performance.
    best_name = min(scores, key=lambda s: s['cv_brier'])['model']
    selected = fitted[best_name]
    scores.append({'model': 'constant_training_prevalence', 'cv_brier': np.nan,
                   **_metrics(yte, np.repeat(ytr.mean(), len(yte)))})
    prediction = data.iloc[test][METADATA].copy()
    for name, p in probabilities.items():
        prediction[name] = p
    columns = selected['columns']
    perm = permutation_importance(selected['model'], Xte[columns], yte,
        scoring='neg_brier_score', n_repeats=10, random_state=random_state, n_jobs=1)
    importance = pd.DataFrame({'feature': columns,
        'brier_increase_when_shuffled': perm.importances_mean,
        'shuffle_sd': perm.importances_std}).sort_values(
            'brier_increase_when_shuffled', ascending=False)
    lr = fitted['clinical_logistic']['model']
    names = lr[:-1].get_feature_names_out(clinical)
    coefficients = pd.DataFrame({'feature': names,
        'log_odds_per_training_SD': lr[-1].coef_[0]})
    coefficients['odds_ratio_per_training_SD'] = np.exp(coefficients.log_odds_per_training_SD)
    coefficients = coefficients.sort_values('log_odds_per_training_SD', ascending=False)
    # Descriptive value-risk tables: bin boundaries come ONLY from training.
    # These are unadjusted observed rates, not causal or validated cutoffs.
    bins = []
    for col in clinical:
        if not (col.endswith('__last') or col.endswith('__slope_per_day')):
            continue
        nonmissing = Xtr[col].dropna()
        if nonmissing.nunique() < 4:
            continue
        cuts = np.unique(nonmissing.quantile([0.25, 0.5, 0.75]).to_numpy())
        labels = pd.cut(Xte[col], [-np.inf, *cuts, np.inf]).astype(str)
        labels[Xte[col].isna()] = 'missing'
        tab = pd.DataFrame({'bin': labels, 'outcome': yte,
                            'risk': probabilities[best_name], 'patient': gte})
        summary = tab.groupby('bin').agg(n_episodes=('outcome', 'size'),
            n_patients=('patient', 'nunique'), n_readmissions=('outcome', 'sum'),
            observed_rate=('outcome', 'mean'), mean_predicted_risk=('risk', 'mean')).reset_index()
        summary.insert(0, 'feature', col)
        bins.append(summary)
    # Bootstrap whole held-out patients; intervals conditional on fitted model.
    # Duplicated sampled patients contribute all their episodes each time.
    rng = np.random.default_rng(random_state)
    by_patient = [np.flatnonzero(gte.to_numpy() == p) for p in pd.unique(gte)]
    bootstrap = []
    for _ in range(bootstrap_repeats):
        idx = np.concatenate([by_patient[j] for j in rng.integers(0, len(by_patient), len(by_patient))])
        if yte.iloc[idx].nunique() == 2:
            bootstrap.append(_metrics(yte.iloc[idx], probabilities[best_name][idx]))
    intervals = (pd.DataFrame(bootstrap).quantile([0.025, 0.975]).T
                 if bootstrap else pd.DataFrame())
    intervals = intervals.rename(columns={0.025: 'lower_95', 0.975: 'upper_95'})
    return {'models': fitted, 'selected_model': best_name,
        'performance': pd.DataFrame(scores).set_index('model'),
        'predictions': prediction, 'importance': importance,
        'logistic_coefficients': coefficients,
        'value_risk_table': pd.concat(bins, ignore_index=True) if bins else pd.DataFrame(),
        'metric_intervals': intervals,
        'split_summary': {'training_patients': len(train_ids), 'test_patients': len(test_ids),
            'training_episodes': len(train), 'test_episodes': len(test),
            'training_events': int(ytr.sum()), 'test_events': int(yte.sum()),
            'test_prevalence': float(yte.mean())}}


def save_results(results, output_dir):
    """Write local CSV reports and calibration plot. Contains patient IDs."""
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for key in ['performance', 'predictions', 'importance', 'logistic_coefficients',
                'value_risk_table', 'metric_intervals']:
        results[key].to_csv(out / (key + '.csv'), index=key in ['performance', 'metric_intervals'])
    (out / 'summary.json').write_text(json.dumps({
        'selected_model': results['selected_model'], **results['split_summary']}, indent=2))
    fig, ax = plt.subplots(figsize=(6, 5))
    p = results['predictions']
    for name in results['models']:
        CalibrationDisplay.from_predictions(p.readmitted, p[name], n_bins=5,
                                            strategy='quantile', name=name, ax=ax)
    ax.set_title('Calibration on held-out patients')
    fig.tight_layout()
    fig.savefig(out / 'calibration.png', dpi=160)
    plt.close(fig)
