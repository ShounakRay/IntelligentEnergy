# @Author: Shounak Ray <Ray>
# @Date:   30-Mar-2021 11:03:34:348  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: defs.py
# @Last modified by:   Ray
# @Last modified time: 20-Apr-2021 11:04:73:735  GMT-0600
# @License: [Private IP]

import sys
import warnings

import numpy as np


def _theo_fluid(df, producer):
    producer_df = df[df['PRO_Well'] == producer].reset_index(drop=True)
    producer_df.loc[producer_df['PRO_Engineering_Approved'] == False, 'PRO_Water'] = np.nan
    producer_df = producer_df.sort_values('Date').interpolate('linear')
    producer_df['PRO_Theo_Fluid'] = (producer_df['PRO_Fluid'] / producer_df['PRO_Pump_Efficiency']
                                     ) * (100 / producer_df['PRO_Adj_Pump_Speed'])
    producer_df = producer_df.replace([np.inf, -np.inf], np.nan)
    producer_df['PRO_Theo_Fluid'] = producer_df['PRO_Theo_Fluid'].dropna().median() * \
        producer_df['PRO_Adj_Pump_Speed']

    return producer_df


def process_local_anomalydetection(df, pwell, cont_col, BASE_UPSCALED=50, UPSCALAR=100):
    try:
        import Anomaly_Detection_PKG
    except Exception:
        sys.path.append('/Users/Ray/Documents/Github/AnomalyDetection')
        import Anomaly_Detection_PKG
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ft, _ = Anomaly_Detection_PKG.anomaly_detection(data=df.copy(),
                                                        well=pwell,
                                                        feature=cont_col,
                                                        ALL_FEATURES=list(df.select_dtypes(float).columns),
                                                        method=['Offline Outlier'],
                                                        mode=['overall'],
                                                        contamination=['0.10'],
                                                        GROUPBY_COL='pro_well',
                                                        TIME_COL='date',
                                                        plot=df)
        # print(f'Anomaly Detection for: {cont_col} and {pwell}')
        # ft.drop(['detection_iter', 'changepoint_status', 'regular_y'], axis=1, inplace=True)
        min_date = min(ft['date'])
        range = (max(ft['date']) - min_date).days
        adjusted_anomscores = []
        transf_time = ft['date'].apply(lambda x: (x - min_date).days / range)**1.0
        for score, time_factor in zip(ft['scores'], transf_time):
            if score >= 0:
                adjusted_anomscores.append(time_factor)
            else:
                adjusted_anomscores.append(time_factor * abs(score**0.10))
        ft['updated_score'] = [BASE_UPSCALED + (orig * UPSCALAR) for orig in adjusted_anomscores]
        ft['group'] = pwell
        ft['feature'] = cont_col

        return ft