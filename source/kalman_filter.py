import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from numpy.linalg import inv, multi_dot

logger = logging.getLogger(__name__)


def load_dataset(csv_path):
    return pd.read_csv(csv_path)


class KalmanFilterSteadyState:
    """定常モデルのカルマンフィルター.

    Attributes:
        estimated_x (float): 最終的な推定値x.
        estimated_x_list (array_like, 2-D): 各時刻における推定値xのリスト.
    """
    def __init__(self, z, h, r, q, init_x, init_p):
        """
        Args:
            z (array_like, 1-D): 観測値リスト.
            h (array_like, 2-D): 変換行列.
            r (array_like, 1-D): 観測雑音の共分散リスト.
            q (array_like, 1-D): プラント雑音の共分散リスト.
            init_x (array_like, 1-D): 推定値xの初期値.
            init_p (array_like, 2-D): 推定誤差共分散pの初期値.
        """
        predicted_xi = None
        estimated_xi = init_x.reshape(-1, 1)
        predicted_pi = None
        estimated_pi = init_p

        self.estimated_x_list = []

        for i in range(z.size):
            # 予測アルゴリズム
            qi = q[i]  # scalar
            hi = h[i].reshape(1, -1)  # shape (1, 3)
            predicted_xi = estimated_xi  # shape (3, 1)
            predicted_pi = estimated_pi + qi  # shape (3, 3)
            predicted_zi = np.dot(hi, predicted_xi)  # shape (1, 1)

            # 推定アルゴリズム
            ri = r[i]  # scalar
            zi = z[i]  # scalar
            z_error = zi - predicted_zi  # shape (1, 1)
            si = multi_dot([hi, predicted_pi, hi.T]) + ri  # shape (1, 1)
            wi = multi_dot([predicted_pi, hi.T, inv(si)])  # shape (3, 1)
            estimated_xi = predicted_xi + np.dot(wi, z_error)  # shape (3, 1)
            estimated_pi = predicted_pi - multi_dot([wi, si, wi.T])  # shape (3, 3)

            self.estimated_x_list.append(estimated_xi.T[0])

        self.estimated_x_list = np.array(self.estimated_x_list)
        self.estimated_x = estimated_xi
        self.estimated_p = estimated_pi


def main():
    # logger設定
    logging.basicConfig(
        level=logging.INFO,
        filemode='w',
        filename='log/kalman_filter.log'
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
    q = np.zeros(k.size)
    init_x = np.zeros(3)
    init_p = np.identity(3) * 10**3
    x = KalmanFilterSteadyState(z, h, r, q, init_x, init_p).estimated_x
    print(f'推定値x（カルマンフィルター）: \n{x}\n')
    logger.info(f'Q: \n{q}\n')
    logger.info(f'init X: \n{init_x}\n')
    logger.info(f'init P: \n{init_p}\n')
    logger.info(f'推定値x（カルマンフィルター）: \n{x}\n')

    """
    ************
    --- （2）---
    ************
    """
    # 推定誤差共分散pの初期値とプラント雑音の共分散qを変えて，カルマンフィルターを適用
    p_label_list = ['10**3', '10**0', '10**-3']
    init_p_list = [np.identity(3) * 10**3, np.identity(3) * 10**0, np.identity(3) * 10**-3]
    q_label_list = ['10**6', '10**3', '10**1']
    q_list = [np.ones(k.size) * 10**6, np.ones(k.size) * 10**3, np.ones(k.size) * 10**1]
    x_stack = []

    for init_p in init_p_list:
        # 各時刻における推定値xのリストを取得
        q = np.zeros(k.size)
        estimated_x_list = KalmanFilterSteadyState(z, h, r, q, init_x, init_p).estimated_x_list
        x_stack.append(estimated_x_list)

    for i in range(3):
        # グラフ作成
        fig, ax = plt.subplots()
        ax.set_xlabel('k')  # x軸ラベル
        ax.set_ylabel(f'x{i}')  # y軸ラベル
        ax.set_title('kalman_filter_change_p')

        for j, x in enumerate(x_stack):
            ax.plot(k, x[:, i], label=p_label_list[j])

        ax.legend(loc=0)
        plt.savefig(f'graph/kalman_filter_change_p_graph_x{i}.png')

    x_stack = []
    for q in q_list:
        # 各時刻における推定値xのリストを取得
        init_p = np.identity(3) * 10 ** 6
        estimated_x_list = KalmanFilterSteadyState(z, h, r, q, init_x, init_p).estimated_x_list
        x_stack.append(estimated_x_list)

    for i in range(3):
        # グラフ作成
        fig, ax = plt.subplots()
        ax.set_xlabel('k')  # x軸ラベル
        ax.set_ylabel(f'x{i}')  # y軸ラベル
        ax.set_title('kalman_filter_change_q')

        for j, x in enumerate(x_stack):
            ax.plot(k, x[:, i], label=q_label_list[j])

        ax.legend(loc=0)
        plt.savefig(f'graph/kalman_filter_change_q_graph_x{i}.png')


if __name__ == '__main__':
    main()
