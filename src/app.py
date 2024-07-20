import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer
from tensorflow_probability import sts
import pickle


class FlowLikeWater:
    path = './Water/acea-water-prediction/'
    watertables = {'Aquifer_': ['Auser', 'Doganella', 'Luco', 'Petrignano'],
                   'Water_Spring_': ['Amiata', 'Lupa', 'Madonna_di_Canneto'],
                   'Lake_': ['Bilancino'],
                   'River_': ['Arno']}
    data = {}
    targets = {}
    kni = KNNImputer(weights='distance', copy=False)

    def __init__(self):
        try:
            with open('./Water/acea-water-prediction/water.dat', 'rb') as file:
                self.data, self.targets = pickle.load(file)
        except FileNotFoundError:
            for k, v in self.watertables.items():
                for i in v:
                    self.data[f'{k}{i}'] = pd.read_csv(f'{self.path + k + i}.csv')
            for k, v in self.data.items():
                relevant = []
                for i in v.columns:
                    if k.startswith('Aquifer'):
                        if i.startswith('Depth'):
                            relevant.append(i)
                    elif k.startswith('Water_Spring'):
                        if i.startswith('Flow_Rate'):
                            relevant.append(i)
                    elif k.startswith('River'):
                        if i.startswith('Hydrometry'):
                            relevant.append(i)
                    else:
                        relevant = ['Lake_Level', 'Flow_Rate']
                        break
                winner = relevant.pop(relevant.index(v.count()[relevant].idxmax()))
                self.targets[k] = winner
                self.data[k] = v.drop(columns=relevant)
                self.data[k] = self.data[k].dropna(subset=winner)
                self.data[k]['Date'] = self.data[k]['Date'].apply(self.date_stripper)
                self.data[k] = self.data[k].set_index('Date')
                self.data[k] = self.rain_correct(self.data[k], winner)
            for _, i in self.data.items():
                cols = [j for j in i.columns if j != 'Date']
                self.data[_][cols] = self.kni.fit_transform(i)
        finally:
            with open('./Water/acea-water-prediction/water.dat', 'wb') as file:
                pickle.dump([self.data, self.targets], file)
            self.model_mayhem()

    def model_mayhem(self):
        for k, v in self.data.items():
            if k.startswith('Lake'):
                still = v.diff().dropna()
                train, test = TimeSeriesSplit(n_splits=2).split(still)
                train, test = v.iloc[test[0]], v.iloc[test[1]]
                trend = sts.Autoregressive(order=2, observed_time_series=train, name='autoregressive')
                seasonal = sts.Seasonal(
                    num_seasons=13, num_steps_per_season=28, observed_time_series=train)
                model = sts.Sum([trend, seasonal], observed_time_series=train)
                variational_posteriors = sts.build_factored_surrogate_posterior(
                    model=model)
                q_samp = variational_posteriors.sample(50)
                predictions = sts.forecast(
                    model,
                    observed_time_series=train,
                    parameter_samples=q_samp,
                    num_steps_forecast=len(test))
                mean = predictions.mean().numpy()[..., 0]
                mean = pd.DataFrame(pd.DataFrame(mean).transpose()[6])
                mean = mean.set_index(test.index)
                print(
                    f'Mean Squared Error of predictions: {mean_squared_error(list(test[self.targets[k]]), mean)}')
                sns.lineplot(data=test[self.targets[k]])
                sns.lineplot(data=mean, color='red')
                with open('./Water/acea-water-prediction/rivermodel.dat', 'wb') as file:
                    pickle.dump(model, file)
                plt.show()

    def date_stripper(self, d):
        return dt.datetime.strptime(d, '%d/%m/%Y').date()

    def rain_correct(self, df, target):
        cols = [i for i in df.columns if i.startswith('Rainfall')]
        prop = df.drop(columns=[i for i in df.columns if i not in cols and i != target])
        index = list(prop.index)
        relations = {i: {j: [] for j in cols if j != i} for i in cols}
        for i in relations.keys():
            for j in cols:
                if j != i:
                    if len(relations[j][i]) != 0:
                        running_total = 1 / relations[j][i][0]
                        counter_total = prop.loc[(prop[i] == 0) & (prop[j] != 0)][j].mean()
                        relations[i][j] = [running_total, counter_total]
                    else:
                        running_total = []
                        counter_total = []
                        for k in range(len(index)):
                            if not pd.isna(prop.loc[index[k], i]) and not pd.isna(prop.loc[index[k], j]):
                                if prop.loc[index[k], j] != 0:
                                    running_total.append(prop.loc[index[k], i] / prop.loc[index[k], j])
                                else:
                                    if prop.loc[index[k], i] == 0:
                                        running_total.append(0)
                                    else:
                                        counter_total.append(prop.loc[index[k], i])
                        if len(counter_total) != 0:
                            relations[i][j] = [np.mean(running_total), np.mean(counter_total)]
                        else:
                            relations[i][j] = [np.mean(running_total)]
        for i in relations.keys():
            for j in range(len(index)):
                if pd.isna(prop.loc[index[j], i]):
                    results = []
                    for k in relations[i].keys():
                        if not pd.isna(prop.loc[index[j], k]) and prop.loc[index[j], k] != 0:
                            results.append(prop.loc[index[j], k] * relations[i][k][0])
                        elif prop.loc[index[j], k] == 0:
                            results.append(relations[i][k][1])
                    if len(results) != 0:
                        prop.loc[index[j], i] = np.mean(results)
        results = {k: {0: prop.corr(method='spearman')[target].loc[k]} for k in cols}
        for offset in range(1, 181):
            prop = self.popper(prop)
            for i in cols:
                results[i][offset] = prop.corr(method='spearman')[target].loc[i]
        results = pd.DataFrame(results)
        # results = results.apply(abs)
        res = {k: results[k].idxmax() for k in results.columns}
        df = self.hopper(df, res)
        return df

    def hopper(self, df, res):
        for k, v in res.items():
            for i in range(v):
                df[k] = self.popper(df[k])
        return df

    def popper(self, df):
        if isinstance(df, pd.DataFrame):
            for i in df.columns:
                if i.startswith('Rainfall'):
                    lisp = list(df[i])
                    pop = lisp.pop(-1)
                    lisp.insert(0, pop)
                    df[i] = lisp
                    df.drop(axis=0, index=df.index[0])
        else:
            df = list(df)
            pop = df.pop(-1)
            df.insert(0, pop)
        return df


if __name__ == '__main__':
    FlowLikeWater()
