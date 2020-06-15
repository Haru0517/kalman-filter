import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from numpy.linalg import inv, multi_dot

logger = logging.getLogger(__name__)


def load_dataset(csv_path):
    return pd.read_csv(csv_path)


class BatchLeastSquares:
    """バッチ型最小２乗法.

    Attributes:
        estimated_x (float): 推定値x.
    """
    def __init__(self, z, h, r):
        """
        Args:
            z (array_like, 1-D): 観測値リスト.
            h (array_like, 2-D): 変換行列.
            r (array_like, 1-D): 共分散リスト.
        """
        z = z.reshape(-1, 1)
        r = np.diag(r)
        self.estimated_x = multi_dot([inv(multi_dot([h.T, inv(r), h])), h.T, inv(r), z])


class IterLeastSquares:
    """逐次型最小２乗法.

    Attributes:
        estimated_x (float): 最終的な推定値x.
        p (array_like, 2-D): 最終的な推定誤差共分散p.
        x_list (array_like, 2-D): 各時刻における推定値xのリスト.
    """
    def __init__(self, z, h, r, init_x, init_p):
        """
        Args:
            z (array_like, 1-D): 観測値リスト.
            h (array_like, 2-D): 変換行列.
            r (array_like, 1-D): 共分散リスト.
            init_x (array_like, 1-D): 推定値xの初期値.
            init_p (array_like, 2-D): 推定誤差共分散pの初期値.
        """
        xi = init_x.reshape(-1, 1)
        pi = init_p

        self.x_list = []

        for i in range(z.size):
            hi = h[i].reshape(1, -1)  # shape (1, 3)
            ri = r[i]  # scalar
            zi = z[i]  # scalar
            si = multi_dot([hi, pi, hi.T]) + ri  # shape (1, 1)
            wi = multi_dot([pi, hi.T, inv(si)])  # shape (3, 1)
            xi = xi + np.dot(wi, (zi - np.dot(hi, xi)))  # shape (3, 1)
            pi = pi - multi_dot([wi, si, wi.T])  # shape (3, 3)

            self.x_list.append(xi.T[0])

        self.x_list = np.array(self.x_list)
        self.estimated_x = xi
        self.p = pi


def main():
    # logger設定
    logging.basicConfig(
        level=logging.INFO,
        filemode='w',
        filename='log/least_squares.log'
    )
    # データセットの読み込み
    dataset_path = 'dataset.csv'
    dataset_df = load_dataset(dataset_path)

    # 行列生成
    k = dataset_df['k'].to_numpy()
    z = dataset_df['z'].to_numpy()
    h = np.stack([np.ones(k.size), k, k*k], axis=1)
    r = np.where(k % 2 == 0, 4, 1)
    logger.info(f'K: \n{k}\n')
    logger.info(f'Z: \n{z}\n')
    logger.info(f'H: \n{h}\n')
    logger.info(f'R: \n{r}\n')

    """
    ************
    --- （1) --- 
    ************
    """
    # バッチ型最小２乗法
    x = BatchLeastSquares(z, h, r).estimated_x
    print(f'推定値x（バッチ型）: \n{x}\n')
    logger.info(f'推定値x（バッチ型）: \n{x}\n')

    # 逐次型最小２乗法
    init_x = np.zeros(3)
    init_p = np.identity(3) * 10**6
    x = IterLeastSquares(z, h, r, init_x, init_p).estimated_x
    print(f'推定値x（逐次型）: \n{x}\n')
    logger.info(f'init X: \n{init_x}\n')
    logger.info(f'init P: \n{init_p}\n')
    logger.info(f'推定値x（逐次型）: \n{x}\n')

    """
    ************
    --- （2）---
    ************
    """
    # 推定誤差共分散pの初期値を変えて，逐次型最小２乗法を適用
    label_list = ['10**3', '10**0',  '10**-3']
    init_p_list = [np.identity(3) * 10**3, np.identity(3) * 10**0, np.identity(3) * 10**-3]
    x_list = []
    for init_p in init_p_list:
        x_list.append(IterLeastSquares(z, h, r, init_x, init_p).x_list)

    # グラフ作成
    for i in range(3):
        fig, ax = plt.subplots()
        ax.set_xlabel('k')  # x軸ラベル
        ax.set_ylabel(f'x{i}')  # y軸ラベル
        ax.set_title('iter least squares')

        for j, x in enumerate(x_list):
            ax.plot(k, x[:, i], label=label_list[j])

        ax.legend(loc=0)
        plt.savefig(f'graph/iter_least_squares_graph_x{i}.png')


if __name__ == '__main__':
    main()



