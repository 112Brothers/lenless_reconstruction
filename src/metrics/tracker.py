import pandas as pd


class MetricTracker:

    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(
            index=keys,
            columns=["total", "counts", "average"],
            dtype=float,
        )
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col] = 0.0

    def update(self, key, value, n=1):
        self._data.loc[key, "total"] = float(self._data.loc[key, "total"]) + float(value) * n
        self._data.loc[key, "counts"] = float(self._data.loc[key, "counts"]) + n
        self._data.loc[key, "average"] = self._data.loc[key, "total"] / self._data.loc[key, "counts"]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def keys(self):
        return self._data.total.keys()