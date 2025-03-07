from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from Helpers.utils import Utils

# =============================================================================
# Data Loading and Preprocessing
# =============================================================================
class DataHandler:
    def __init__(self, data_name=None, data_dir=None, utils=None):
        self.dn = data_name
        self.dd = data_dir if data_dir is not None else Path.cwd() / "Data"
        self.dd.mkdir(exist_ok=True, parents=True)
        self.u = Utils() if utils is None else utils
        self.d = None
        self.ds = {"train" : None, "test" : None}
        self.dl = self.ds.copy()

    def dataNames_(self):
        dn = {
            "ecdc" : "https://opendata.ecdc.europa.eu/covid19/casedistribution/csv"
        }

        return dn

    def loadData(self, data_name=None, cache=False):
        if self.d is not None: return self.d

        dn = data_name if data_name is not None else self.dn
        assert dn is not None, "[Error] (DataHandler::downloadData_): data_name is None"

        dn = dn.lower()
        d_file = self.dd / f"{dn}.csv"
        if not d_file.is_file(): self.downloadData_(d_file, dn)

        df = pd.read_csv(d_file)
        if cache: self.d = df
        return df

    def summarize(self, df=None):
        df = df if df is not None else self.d
        self.u.printHeaderText("Info", df.info())
        self.u.printHeaderText("Dtypes", df.dtypes)
        self.u.printHeaderText("Head 10", df.head(15))

    def extendData(self, df=None, cache=False):
        df = df if df is not None else self.d
        df = df.copy()

        # Combine day, month, and year columns into a datetime
        if 'date' not in df.columns:
            df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
            df = df.sort_values('date')
            df = df.reset_index(drop=True)

        # Group by 'countriesAndTerritories' to compute country-specific columns
        if 'cum_cases' not in df.columns:
            df['cum_cases'] = df.groupby('countriesAndTerritories')['cases'].cumsum()
        if 't' not in df.columns:
            df['t'] = df.groupby('countriesAndTerritories')['date'].apply(
                lambda x: (x - x.iloc[0]).dt.days.astype(float)
            ).reset_index(level=0, drop=True)

        if cache: self.d = df

        return df

    def processData(self, df=None, filters=None, cache=False):
        df = df if df is not None else self.d
        df = df.copy()
        if filters is not None:
            for key, val in filters.items():
                if key == 'countriesAndTerritories' and isinstance(val, list):
                    df = df[df[key].isin(val)]
                elif "~" in val:
                    val = val.split("~")[1]
                    df = df[df[key] != val].copy()
                else:
                    df = df[df[key] == val].copy()

        if filters is not None and 'continentExp' in filters:
            assert len(df['continentExp'].unique()) == 1, (
                "[Error] (DataHandler::processData): continentExp filter"
                + f" ({filters['continentExp']}) doesn't give unique entries"
            )

        if cache: self.d = df

        return df

    def tensorData(self, cols, df=None, device=None):
        df = df if df is not None else self.d
        in_datas = []
        for col in cols:
            in_d = df[col].values.reshape(-1, 1)
            in_datas.append(torch.tensor(in_d, dtype=torch.float32, device=device))
        in_data = torch.cat(in_datas, dim=1)
        return in_data

    def mkDataset(self, *t, cache=True, name="train"):
        if self.ds[name] is not None: return self.ds[name]
        dataset = TensorDataset(*t)
        if cache: self.ds[name] = dataset
        return dataset

    def mkDataloader(self, x=None, y=None, dataset=None, batch_size=4, shuffle=True,
                     drop_last=False, cache=True, name="train", use_mode=None, loader_params=None):
        if self.dl[name] is not None: return self.dl[name]

        if name == "test":
            x = torch.cat(list(x.values()), dim=0)
            y = torch.cat(list(y.values()), dim=0)

        if dataset is None:
            dataset = self.mkDataset(x, y, cache=cache, name=name)

        loader_params = loader_params if loader_params is not None else dict(
            worker_init_fn=None,
            generator=None
        )

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            drop_last=drop_last, **loader_params)
        if cache: self.dl[name] = loader
        return loader

    def splitData(self, x, y, ratio=0.8, seed=None, name=""):
        if ratio == 1.0: return x, None, y, None
        elif ratio == 0.0: return None, x, None, y

        n_data = x.shape[0]
        split = int(ratio * n_data)
        if split < 1: return x, None, y, None

        indices = list(range(n_data))
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        train_idx, test_idx = indices[:split], indices[split:]
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]

        name = f" for name {name}" if name else ""
        self.u.printHeaderText(f"No. of total data{name}", x.shape[0])
        self.u.printHeaderText(f"No. of (Train, Test) data{name}", (x_train.shape[0], x_test.shape[0]))
        return x_train, x_test, y_train, y_test

    def normalize_(self, x, y):
        n_data = x.shape[0]
        assert y.shape[0] == n_data, (
            f"[Error] (train): no. of x data ({n_data}) doesn't match y data ({y.shape[0]})"
        )

        y = (y / y[:, -1:]) * 200
        # y[:, 0] = y[:, 0] / y[:, -1]
        # y[:, 0] = y[:, 0] / 1e6
        # y = y / 1e6
        # Normalize time to [0,1] (helps neural network training)
        x[:, 0] = x[:, 0] / x[:, 0].max()

        return x, y

    def scale_(self, x, y, fit=None, key=None):
        x, y = self.normalize_(x, y)
        return x, y

    def mkTrainTest(self, use_mode, in_cols, out_cols,
                    filters_global=None, filters_local=None,
                    df=None, ratio=0.8, data_name=None):
        use_local = use_mode["local"]
        look_back = use_mode["lstm"]
        if look_back:
            return self.mkTrainTestLSTM_(look_back, use_local,
                                         in_cols, out_cols,
                                         filters_global, filters_local,
                                         df, ratio, data_name)
        else:
            return self.mkTrainTestMLP_(use_local, in_cols, out_cols,
                                        filters_global, filters_local,
                                        df, ratio, data_name)

    def mkTrainTestMLP_(self, use_local, in_cols, out_cols,
                        filters_global=None, filters_local=None,
                        df=None, ratio=0.8, data_name=None):
        df = df if df is not None else self.d
        dn = data_name if data_name is not None else self.dn
        filters = filters_local if use_local else filters_global
        df = self.processData(df, filters=filters)
        df = df.sort_values(by=["countriesAndTerritories", "t"])
        df_grp = df.groupby("countriesAndTerritories", sort=False)

        seed = self.u.seed()
        device = self.u.device()

        x_train, y_train = [], []
        x_test, y_test = {}, {}
        country_scalers = {}  # To save scalers for inference

        for country, group in df_grp:
            x = self.tensorData(in_cols, group, device=device)
            y = self.tensorData(out_cols, group, device=device)
            x, y = self.normalize_(x, y)

            train_feats, test_feats, train_targets, test_targets = self.splitData(x, y, ratio, seed=seed)
            x_train.append(train_feats)
            y_train.append(train_targets)
            x_test[country] = test_feats
            y_test[country] = test_targets

        x_train = torch.cat(x_train, dim=0)
        y_train = torch.cat(y_train, dim=0)

        self.u.printHeaderText("No. of Train data and len. of Test data", (x_train.shape[0], len(x_test)))
        if "Italy" in x_test:
            self.u.printHeaderText("No. of Test data for Italy", x_test["Italy"].shape[0])

        return x_train, x_test, y_train, y_test

    def mkSequence_(self, x, y, look_back):
        n_data = x.shape[0]
        seqs, targets = [], []
        for start in range(n_data - look_back):
            end = start + look_back
            seq1 = x[start:end+1]
            seq2 = y[start:end, :-1]
            seq2_last = torch.zeros([1, seq2.shape[1]],
                                    device=seq2.device, dtype=seq2.dtype)
            assert seq2_last.dim() == seq2.dim(), (
                f"[Error] (mkSequence_): invalid dims for seq2 {seq2.dim()} and"
                + f" seq2_last {seq2_last.dim()}"
            )
            seq2 = torch.cat([seq2, seq2_last], dim=0)
            assert seq1.shape[0] == seq2.shape[0], (
                f"[Error] (mkSequence_): invalid size for seq1 {seq1.shape[0]} and"
                + f" seq2 {seq2.shape[0]}"
            )
            seq = torch.cat([seq1, seq2], dim=1)
            seqs.append(seq)
            target = y[end]
            targets.append(target)

        seqs = torch.stack(seqs, dim=0)
        targets = torch.stack(targets, dim=0)
        return seqs, targets

    def mkTrainTestLSTM_(self, look_back, use_local,
                         in_cols, out_cols,
                         filters_global=None, filters_local=None,
                         df=None, ratio=0.8, data_name=None):
        df = df if df is not None else self.d
        dn = data_name if data_name is not None else self.dn
        filters = filters_local if use_local else filters_global
        df = self.processData(df, filters=filters)
        df = df.sort_values(by=["countriesAndTerritories", "t"])
        df_grp = df.groupby("countriesAndTerritories", sort=False)

        seed = self.u.seed()
        device = self.u.device()

        x_train, y_train = [], []
        x_test, y_test = {}, {}
        country_scalers = {}  # To save scalers for inference

        for country, group in df_grp:
            x = self.tensorData(in_cols, group, device=device)
            y = self.tensorData(out_cols, group, device=device)
            x, y = self.scale_(x, y, fit=True, key=country)

            n_data = x.shape[0]
            if n_data < look_back + 1:
                continue
            seqs, targets = self.mkSequence_(x, y, look_back)
            train_seqs, test_seqs, train_targets, test_targets = self.splitData(seqs, targets, ratio, seed=seed, name=country)
            x_train.append(train_seqs)
            y_train.append(train_targets)
            x_test[country] = test_seqs
            y_test[country] = test_targets

        x_train = torch.cat(x_train, dim=0)
        y_train = torch.cat(y_train, dim=0)

        self.u.printHeaderText("No. of Train data and len. of Test data", (x_train.shape[0], len(x_test)))
        if "Italy" in x_test:
            self.u.printHeaderText("No. of Test data for Italy", x_test["Italy"].shape[0])

        return x_train, x_test, y_train, y_test

    def downloadData_(self, data_file, data_name=None):
        d_file = data_file
        dn = data_name if data_name is not None else self.dn
        d_url = self.dataNames_()[dn]
        df = pd.read_csv(d_url)
        df.to_csv(d_file)

    def setData(self, df):
        self.d = df

    def data(self):
        return self.d

    def dataset(self, name=None):
        if name is not None: return self.ds[name]
        return self.ds

    def dataloader(self, name=None):
        if name is not None: return self.dl[name]
        return self.dl
