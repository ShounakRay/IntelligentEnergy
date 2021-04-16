# @Author: Shounak Ray <Ray>
# @Date:   16-Apr-2021 08:04:73:734  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: S1F_base_generation.py
# @Last modified by:   Ray
# @Last modified time: 16-Apr-2021 13:04:68:680  GMT-0600
# @License: [Private IP]

import datetime
from typing import Final

import pandas as pd
from acessories import get_default_args


class Ingestion():

    __purpose__: Final = 'The purpose of {} is to accept: \n1. Injection\n2. Production \n3. Well Test\n4. Fiber\n' \
        'data.\nUpon accceptance, this class has methods which can transform this data\nso it\'s ready for the next ' \
        'step in the analytical pipeline.\nThis ensures compartmentalization.'

    def __init__(self):
        self._name = self.__class__.__name__
        self.id = id(self)
        self.init_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.datasets = {}
        self.flow_history = []
        self.joined_data = None

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
        if(data_group is None):
            source_data = self.joined_data
        else:
            source_data = self.datasets['']
        if(group_colname not in source_data.columns):
            raise KeyError(f'{group_colname} isn\'t in the specified dataset.')
        return list(self.source_data[group_colname].unique())

    def producer_wells(pwell_colname='PRO_Well', data_group=None):
        return helper_find_uniques(**locals())

    def producer_pads(ppad_colname='PRO_Pad', data_group=None):
        return helper_find_uniques(**locals())

    def injector_wells(iwell_colname='INJ_Well', data_group=None):
        return helper_find_uniques(**locals())

    def injector_pads(ipad_colname='INJ_Pad', data_group=None):
        return helper_find_uniques(**locals())

    def ingest(self, sep=',', encoding='utf-8', error_bad_lines=False, **kw_paths):
        import_configs = get_default_args(self.ingest)
        batch_1 = {k: v for k, v in kw_paths.items() if k in ['INJECTION', 'PRODUCTION', 'PRODUCTION_TEST']}
        batch_2 = {k: v for k, v in kw_paths.items() if k in ['FIBER']}

        if (not batch_1) and (not batch_2):
            raise ValueError('Nothing to import. `kwargs` are either improperly entered or ' +
                             'ingestion of unexpected files through `kwargs` was attempted.')

        if len(batch_1) != 0:
            for kw, path in batch_1.items():
                if(path.endswith('.csv')):
                    try:
                        print('> Importing {kw}...')
                        imported_file = pd.read_csv(path, **import_configs)
                    except Exception:
                        raise ImportError(f'Fatally failed to ingest {path}. Check import configs and data source.')
                elif(path.endswith(('.xlsx', '.xls', '.xlsm', '.xlsb', '.odf', '.ods', '.odt'))):
                    import_configs = {}
                    try:
                        print('> Importing {kw}...')
                        imported_file = pd.read_excel(path, **import_configs)
                    except Exception:
                        raise ImportError(f'Fatally failed to ingest {path}. Check import configs and data source.')
                else:
                    raise ImportError(f'Incompatible file extention. Cannot ingest {path}.')

                self.datasets[kw] = imported_file.infer_objects()

        if not batch_2:
            for kw, path in batch_2.items():
                if type(path) is not list:
                    raise ValueError('Expected list of FIBER data paths.')
                else:
                    print('> Importing {kw}...')
                    pass

    def healthy_data(self, df):
        healthy = True
        if(not isinstance(df, pd.DataFrame) or df.empty):
            healthy = False

        if(not healthy):
            raise RuntimeError('The attributed dataset is not healthy!')

        return

    def access(self, data_group):
        if(data_group not in self.datasets.keys()):
            raise KeyError(f'Column named "{data_group}" doens\'t exist in this instance of {self._name}.')
        data = self.datasets[data_group].infer_objects()

        self.healthy_data(data)
        return self.datasets[data_group]

    def _config(self, verbose=False):
        return {attr: getattr(self, attr) for attr in dir(self)} if verbose else self.__dict__


def core_ingestion(fiber_dir='Data/DTS/'):
    ingestion = Ingestion()

    ingestion.ingest(INJECTION='Data/Isolated/OLT injection data.xlsx',
                     PRODUCTION='Data/Isolated/OLT production data (rev 1).xlsx',
                     PRODUCTION_TEST='Data/Isolated/OLT well test data.xlsx')

    ingestion.datasets
    ingestion._config()
    ingestion.access('INJECTION')
