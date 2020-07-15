import numpy as np
import pandas as pd
import logging
from numpy.linalg import inv, multi_dot
from copy import deepcopy
from scipy.stats import chi2


logger = logging.getLogger(__name__)


def load_dataset(csv_path):
    return pd.read_csv(csv_path)


def create_matrix_from_dataset(dataset_df):
    k = dataset_df['k'].to_numpy()
    z = dataset_df['z'].to_numpy()
    h = np.stack([np.ones(k.size), k, k*k], axis=1)
    r = np.where(k % 2 == 0, 4, 1)

    logger.info(f'K: \n{k}\n')
    logger.info(f'Z: \n{z}\n')
    logger.info(f'H: \n{h}\n')
    logger.info(f'R: \n{r}\n')
    return k, z, h, r


def print_console_and_log(string):
    logger.info(string)
    print(string)


class KalmanFilterWithDataAssociation:
    """データアソシエーション有りのカルマンフィルター.

    Attributes:
        k (array_like, 1-D): 時刻リスト.
        z (array_like, 1-D): 観測値リスト.
        h (array_like, 2-D): 変換行列.
        r (array_like, 1-D): 観測雑音の共分散リスト.
        q (array_like, 1-D): プラント雑音の共分散リスト.
        init_x (array_like, 1-D): 推定値xの初期値.
        init_p (array_like, 2-D): 推定誤差共分散pの初期値.
        alpha (float): カイ２乗検定の危険率.
        estimated_x (float): 最終的な推定値x.
        estimated_x_list (array_like, 2-D): 各時刻における推定値xのリスト.
    """
    def __init__(self, k, z, h, r, q, init_x, init_p, alpha=0.1):
        """初期化.

        Args:
            Attributes参照.
        """
        self.k = k
        self.z = z
        self.h = h
        self.r = r
        self.q = q
        self.init_x = init_x
        self.init_p = init_p
        self.alpha = alpha

        self.estimated_x_list = []
        self.estimated_x = None
        self.estimated_p = None


    def run(self):
        """推定値を計算する."""
        predicted_xi = None
        estimated_xi = self.init_x.reshape(-1, 1)
        predicted_pi = None
        estimated_pi = self.init_p

        self.estimated_x_list = []

        for i in range(self.z.size):
            # 予測アルゴリズム
            qi = self.q[i]  # scalar
            hi = self.h[i].reshape(1, -1)  # shape (1, 3)
            predicted_xi = estimated_xi  # shape (3, 1)
            predicted_pi = estimated_pi + qi  # shape (3, 3)
            predicted_zi = np.dot(hi, predicted_xi)  # shape (1, 1)

            # データアソシエーション
            ri = self.r[i]  # scalar
            zi = self.z[i]  # scalar
            z_error = zi - predicted_zi  # shape (1, 1)
            si = multi_dot([hi, predicted_pi, hi.T]) + ri  # shape (1, 1)
            # ε
            e = multi_dot([z_error.T, inv(si), z_error])  # shape (1, 1)
            # 次元数
            n = self.init_x.size  # scalar

            # カイ２乗分布を取得
            r = chi2.ppf(q=1-self.alpha, df=n)
            z_is_correct: bool = e <= r  # 観測値zが正しいかどうかの真理値

            if z_is_correct:
                # 推定アルゴリズム
                wi = multi_dot([predicted_pi, hi.T, inv(si)])  # shape (3, 1)
                estimated_xi = predicted_xi + np.dot(wi, z_error)  # shape (3, 1)
                estimated_pi = predicted_pi - multi_dot([wi, si, wi.T])  # shape (3, 3)

            else:
                print_console_and_log(f'異常値を検出: k={self.k[i]}, z={zi}, ε={e}')

            self.estimated_x_list.append(estimated_xi.T[0])

        self.estimated_x_list = np.array(self.estimated_x_list)
        self.estimated_x = estimated_xi
        self.estimated_p = estimated_pi


def main():
    # logger設定
    logging.basicConfig(
        level=logging.INFO,
        filemode='w',
        filename='log/data_association.log'
    )

    # データセットの読み込み
    dataset_path = 'dataset.csv'
    dataset_df = load_dataset(dataset_path)

    s = \
    """
    ************
    --- （1) --- 
    ************
    """
    print_console_and_log(s)

    # 行列生成
    k, z, h, r = create_matrix_from_dataset(dataset_df)

    q = np.zeros(k.size)
    init_x = np.zeros(3)
    init_p = np.identity(3) * 10**3
    alpha = 0.1
    kf = KalmanFilterWithDataAssociation(k, z, h, r, q, init_x, init_p, alpha)
    kf.run()
    x = kf.estimated_x
    logger.info(f'Q: \n{q}\n')
    logger.info(f'init X: \n{init_x}\n')
    logger.info(f'init P: \n{init_p}\n')
    result = f'推定値x（通常）: \n{x}\n'
    print_console_and_log(result)

    s = \
    """
    ************
    --- （2）---
    ************
    """
    print_console_and_log(s)

    # k=0で異常値を与えて実験
    fixed_df = deepcopy(dataset_df)
    fixed_df.at[fixed_df['k'] == 0, 'z'] = -100  # k=0に-100を代入
    k, z, h, r = create_matrix_from_dataset(fixed_df)
    kf = KalmanFilterWithDataAssociation(k, z, h, r, q, init_x, init_p, alpha)
    kf.run()
    x = kf.estimated_x
    result = f'推定値x（k=0で異常値）: \n{x}\n'
    print_console_and_log(result)

    s = \
    """
    ************
    --- （3）---
    ************
    """
    print_console_and_log(s)

    # k=-13で異常値を与えて実験
    fixed_df = deepcopy(dataset_df)
    fixed_df.at[fixed_df['k'] == -13, 'z'] = -100  # k=-13に-100を代入
    k, z, h, r = create_matrix_from_dataset(fixed_df)
    kf = KalmanFilterWithDataAssociation(k, z, h, r, q, init_x, init_p, alpha)
    kf.run()
    x = kf.estimated_x
    result = f'推定値x（k=-13で異常値）: \n{x}\n'
    print_console_and_log(result)


if __name__ == '__main__':
    main()
