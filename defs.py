# @Author: Shounak Ray <Ray>
# @Date:   30-Mar-2021 11:03:34:348  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: defs.py
# @Last modified by:   Ray
# @Last modified time: 30-Mar-2021 14:03:18:184  GMT-0600
# @License: [Private IP]

import sys
import warnings

BASE_UPSCALED = 0
UPSCALAR = 100


def process_local_anomalydetection(df, pwell, cont_col):
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
        transf_time = ft['date'].apply(lambda x: (x - min_date).days / range)**2
        ft.to_html('reference_only.html')
        for score, time_factor in zip(ft['scores'], transf_time):
            if score >= 0:
                adjusted_anomscores.append(time_factor)
            else:
                adjusted_anomscores.append(time_factor * abs(score**0.10))
        ft['updated_score'] = [BASE_UPSCALED + (orig * UPSCALAR) for orig in adjusted_anomscores]
        ft['group'] = pwell
        ft['feature'] = cont_col

        return ft
