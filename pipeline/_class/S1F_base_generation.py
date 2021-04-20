# @Author: Shounak Ray <Ray>
# @Date:   16-Apr-2021 08:04:73:734  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: S1F_base_generation.py
# @Last modified by:   Ray
# @Last modified time: 17-Apr-2021 00:04:61:610  GMT-0600
# @License: [Private IP]

import datetime
import os
from itertools import chain
from typing import Final

import pandas as pd
from acessories import get_default_args


class Ingestion():

    FORMAT_COLUMNS: Final = {'INJECTION': ['Date', 'INJ_Pad', 'INJ_Well', 'INJ_UWI', 'INJ_Time_On', 'INJ_Alloc_Steam',
                                           'INJ_Meter_Steam', 'INJ_Casing_BHP', 'INJ_Tubing_Pressure', 'INJ_Reason',
                                           'INJ_Comment'],
                             'PRODUCTION': ['Date', 'PRO_Pad', 'PRO_Well', 'PRO_UWI', 'PRO_Time_On',
                                            'PRO_Downtime_Code', 'PRO_Alloc_Oil', 'PRO_Alloc_Water', 'PRO_Alloc_Gas',
                                            'PRO_Alloc_Steam', 'PRO_Alloc_Steam_To_Producer', 'PRO_Hourly_Meter_Steam',
                                            'PRO_Daily_Meter_Steam', 'PRO_Pump_Speed', 'PRO_Tubing_Pressure',
                                            'PRO_Casing_Pressure', 'PRO_Heel_Pressure',  'PRO_Toe_Pressure',
                                            'PRO_Heel_Temp', 'PRO_Toe_Temp', 'PRO_Last_Test_Date', 'PRO_Reason',
                                            'PRO_Comment'],
                             'PRODUCTION_TEST': ['PRO_Pad', 'PRO_Well', 'PRO_Start_Time', 'PRO_End_Time',
                                                 'PRO_Duration', 'PRO_Effective_Date', 'PRO_24_Fluid', 'PRO_24_Oil',
                                                 'PRO_24_Water', 'PRO_Oil', 'PRO_Water', 'PRO_Gas', 'PRO_Fluid',
                                                 'PRO_BSW', 'PRO_Chlorides', 'PRO_Pump_Speed', 'PRO_Pump_Efficiency',
                                                 'PRO_Pump_Size', 'PRO_Operator_Approved', 'PRO_Operator_Rejected',
                                                 'PRO_Operator_Comment', 'PRO_Engineering_Approved',
                                                 'PRO_Engineering_Rejected', 'PRO_Engineering_Comment']}
    CHOICE_COLUMNS: Final = {'INJECTION': ['Date', 'INJ_Pad', 'INJ_Well', 'INJ_Meter_Steam', 'INJ_Casing_BHP',
                                           'INJ_Tubing_Pressure'],
                             'PRODUCTION': ['Date', 'PRO_UWI', 'PRO_Well', 'PRO_Pump_Speed', 'PRO_Time_On',
                                            'PRO_Casing_Pressure', 'PRO_Heel_Pressure', 'PRO_Toe_Pressure',
                                            'PRO_Heel_Temp', 'PRO_Toe_Temp', 'PRO_Alloc_Oil', 'PRO_Alloc_Water'],
                             'PRODUCTION_TEST': ['PRO_Pad', 'PRO_Well', 'PRO_Start_Time', 'PRO_Duration', 'PRO_Oil',
                                                 'PRO_Water', 'PRO_Gas', 'PRO_Fluid', 'PRO_Chlorides',
                                                 'PRO_Pump_Efficiency', 'PRO_Engineering_Approved']}

    __purpose__: Final = 'The purpose of {} is to accept: \n1. Injection\n2. Production \n3. Well Test\n4. Fiber\n' \
        'data.\nUpon accceptance, this class has methods which can transform this data\nso it\'s ready for the next ' \
        'step in the analytical pipeline.\nThis ensures compartmentalization.'

    def __init__(self):
        self._name = self.__class__.__name__
        self.id = id(self)
        self.init_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.datasets = {}
        self.flow_history = []

    @property
    def __str__(self):
        formatted = f'{self._name} imports two '
        return formatted

    @property
    def __repr__(self):
        formatted = f'{self._name}:\n' \
            'Flow Completed:\n' + [f'> {path}' for path in self.datasets]
        return formatted

    @property
    def _purpose(self):
        return print(Ingestion.__purpose__.format(self._name))

    def helper_find_uniques(self, group_colname, data_group):
        source_data = self.access(data_group)
        if(group_colname not in source_data.columns):
            raise KeyError(f'{group_colname} isn\'t in the specified dataset.')
        return list(source_data[group_colname].unique())

    def producer_wells(self, colname='PRO_Well', data_group='JOINED'):
        return self.helper_find_uniques(*[v for k, v in locals().items() if k != 'self'])

    def producer_pads(self, colname='PRO_Pad', data_group='JOINED'):
        return self.helper_find_uniques(*[v for k, v in locals().items() if k != 'self'])

    def injector_wells(self, colname='INJ_Well', data_group='INJECTOR'):
        return self.helper_find_uniques(*[v for k, v in locals().items() if k != 'self'])

    def injector_pads(self, colname='INJ_Pad', data_group='INJECTOR'):
        return self.helper_find_uniques(*[v for k, v in locals().items() if k != 'self'])

    def ingest(self, sep=',', encoding='utf-8', error_bad_lines=False, **kw_paths):
        def fiber_aggregation(combined):
            combined = combined.iloc[:, 9:]
            combined = combined.apply(pd.to_numeric)
            combined = combined.reset_index()
            combined = combined.sort_values('index')
            combined['index'] = combined['index'].str.split(" ").str[0]

            max_length = int(max(combined.columns[1:]))
            segment_length = int(max_length / bins)

            condensed = []
            for i, s in enumerate(range(0, max_length + 1, segment_length)):
                condensed.append(pd.DataFrame(
                    {'Bin_' + str(i + 1): list(combined.iloc[:, s:s + segment_length].mean(axis=1))}))

            condensed = pd.concat(condensed, axis=1)
            condensed['Date'] = combined['index']
            condensed = condensed.sort_values('Date').set_index('Date')

        import_configs = get_default_args(self.ingest)
        batch_1 = {k: [v] if (k in ['INJECTION', 'PRODUCTION', 'PRODUCTION_TEST', 'FIBER'] and type(v) != list)
                   else v for k, v in kw_paths.items()}
        # batch_2 = {k: v for k, v in kw_paths.items() if k in ['FIBER']}

        if (not batch_1):
            raise ValueError('Nothing to import. `kwargs` are either improperly entered or ' +
                             'ingestion of unexpected files through `kwargs` was attempted.')

        if len(batch_1) != 0:
            fiber_tracker = []
            pwell_fiber = {}
            last_pwell = None
            for kw, path_s in batch_1.items():
                for path in path_s:
                    if(path.endswith('.csv')):
                        try:
                            print(f'> Importing {kw} – ' + path.split('/')[-1] + '...')
                            imported_file = pd.read_csv(path, **import_configs)
                        except Exception:
                            raise ImportError(
                                f'Fatally failed to ingest {path}. Check import configs and data source.')
                    elif(path.endswith(('.xlsx', '.xls', '.xlsm', '.xlsb', '.odf', '.ods', '.odt'))):
                        import_configs = {}
                        try:
                            print(f'> Importing {kw} – ' + path.split('/')[-1] + '...')
                            imported_file = pd.read_excel(path, **import_configs)
                        except Exception:
                            raise ImportError(
                                f'Fatally failed to ingest {path}. Check import configs and data source.')
                    else:
                        raise ImportError(f'Incompatible file extention. Cannot ingest {path}.')

                    if(kw == 'FIBER'):
                        pwell = path.split('/')[2]
                        fiber_tracker.append(imported_file.T.iloc[1:])
                        if(pwell != last_pwell):
                            pwell_fiber_data = fiber_aggregation(pd.concat(fiber_tracker).reset_index(drop=True))
                            pwell_fiber_data['PRO_Well'] = pwell
                            pwell_fiber[pwell] = pwell_fiber_data
                            fiber_tracker.clear()
                        last_pwell = pwell
                        continue

                    imported_file = imported_file.infer_objects()
                    imported_file.columns = Ingestion.FORMAT_COLUMNS[kw]

                    self.healthy_data(imported_file)
                    self.datasets[kw] = imported_file
            if('FIBER' in batch_1.keys()):
                fiber_aggregate = pd.concat(list(pwell_fiber.values())).dropna(how='all', axis=1).infer_objects()
                self.healthy_data(fiber_aggregate)
                self.datasets['FIBER'] = fiber_aggregate

    def cleanup(self, data_group='ALL'):
        data_group = list(self.datasets.keys()) if data_group == 'ALL' else data_group
        for group, data in {k: v for k, v in self.datasets.items() if k in data_group}:
            self.datasets[group] = self.datasets[group][Ingestion.CHOICE_COLUMNS[group]]

    def healthy_data(self, df):
        healthy = True
        if(not isinstance(df, pd.DataFrame) or df.empty):
            healthy = False

        if(not healthy):
            raise RuntimeError('The attributed dataset is not healthy!')

        return

    def access(self, data_group):
        if(data_group not in self.datasets.keys()):
            raise KeyError(f"Column named \"{data_group}\" doesn't exist in this instance of {self._name}.")
        data = self.datasets[data_group].infer_objects()

        self.healthy_data(data)
        return self.datasets[data_group].infer_objects()

    def _config(self, verbose=False):
        return {attr: getattr(self, attr) for attr in dir(self)} if verbose else self.__dict__


def core_ingestion():
    ingestion = Ingestion()

    FIBER_PRODUCERS = sorted([p for p in os.listdir('Data/DTS/') if p[0] != '.'])
    FIBER_PATHS = list(chain.from_iterable([['Data/DTS/' + pwell + '/' + x for x in os.listdir('Data/DTS/' + pwell)]
                                            for pwell in FIBER_PRODUCERS]))

    ingestion.ingest(INJECTION='Data/Isolated/OLT injection data.xlsx',
                     PRODUCTION='Data/Isolated/OLT production data (rev 1).xlsx',
                     PRODUCTION_TEST='Data/Isolated/OLT well test data.xlsx',
                     FIBER=FIBER_PATHS)

    ingestion.ingest()


# EOF
