
import numpy as np
import pandas as pd
import polars as pl
import re


class W2VEmbed:
    """
    Use word2vec word embedding as a crude document embedding (bag of word2vec).
    https://en.wikipedia.org/wiki/Word2vec 
    """
    def __init__(self, d) -> None:
        """
        :param d: Pandas or Polars data frame with word2vec vectors. first column should be named "word", all other columns should contain the numeric embedding values.
        """
        if not isinstance(d, pl.DataFrame):
            d = pl.DataFrame(d)
        assert d.columns[0] == "word"
        # copy data into a map
        v_cols = [c for c in d.columns if c != "word"]
        self.n_dim = len(v_cols)
        self.mp = dict()
        for i in range(d.shape[0]):
            row = d.row(i)
            self.mp[row[0]] = np.array(row[1:(1 + self.n_dim)])
    
    def transform_str(self, doc: str) -> pd.DataFrame:
        """
        Crude word2vec based document embedder. Embeds documents to mean word2vec word encoding. For demonstration only.
        
        :param X: string to be transformed
        :return: single row data frame representing embedding of document
        """
        assert isinstance(doc, str)
        n_toks = 0
        enc = np.zeros(self.n_dim)
        doc = re.sub('[^0-9a-zA-Z]+', ' ', doc).strip()
        toks = doc.split()
        for tok in toks:
            try:
                v = self.mp[tok]
                n_toks = n_toks + 1
                enc = enc + v
            except KeyError:
                pass
        if n_toks > 1:
            enc = enc / n_toks
        return pd.DataFrame(enc).transpose()
    
    def transform(self, X) -> pd.DataFrame:
        """
        Transform a data column.

        :param X: iterable series (not data frame)
        :return: pd.DataFrame
        """
        # if a data frame or ndarray, get the first column
        if isinstance(X, pl.DataFrame):
            X = X[:, 0]
        elif isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0].values
        elif isinstance(X, np.ndarray):
            assert len(X.shape) >= 1
            assert len(X.shape) <= 2
            if len(X.shape) > 1:
                X = X[:, 0]
        assert not isinstance(X, str)  # common error
        frames = [
            self.transform_str(text) for text in X
        ]
        return pd.concat(frames, ignore_index=True)
