import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from numpy.linalg import inv, multi_dot
from copy import deepcopy

logger = logging.getLogger(__name__)


def load_dataset(csv_path):
    return pd.read_csv(csv_path)


class KalmanFilterSteadyState:
    """定常モデルのカルマンフィルター.

    Attributes:
        k (array_like, 1-D): 時刻リスト.
        z (array_like, 1-D): 観測値リスト.
        h (array_like, 2-D): 変換行列.
        r (array_like, 1-D): 観測雑音の共分散リスト.
        q (array_like, 1-D): プラント雑音の共分散リスト.
        init_x (array_like, 1-D): 推定値xの初期値.
        init_p (array_like, 2-D): 推定誤差共分散pの初期値.
        estimated_x (float): 最終的な推定値x.
        estimated_x_list (array_like, 2-D): 各時刻における推定値xのリスト.
    """
    def __init__(self, k, z, h, r, q, init_x, init_p):
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

        self.estimated_x_list = []
        self.estimated_x = None
        self.estimated_p = None

    def __setitem__(self, key, value):
        """メンバ変数を辞書形式でsetできるようにする.

        Args:
            key (str): メンバ変数の文字列.
            value (str): 代入する値.
        """
        self.__dict__[key] = value

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

            # 推定アルゴリズム
            ri = self.r[i]  # scalar
            zi = self.z[i]  # scalar
            z_error = zi - predicted_zi  # shape (1, 1)
            si = multi_dot([hi, predicted_pi, hi.T]) + ri  # shape (1, 1)
            wi = multi_dot([predicted_pi, hi.T, inv(si)])  # shape (3, 1)
            estimated_xi = predicted_xi + np.dot(wi, z_error)  # shape (3, 1)
            estimated_pi = predicted_pi - multi_dot([wi, si, wi.T])  # shape (3, 3)

            self.estimated_x_list.append(estimated_xi.T[0])

        self.estimated_x_list = np.array(self.estimated_x_list)
        self.estimated_x = estimated_xi
        self.estimated_p = estimated_pi


def output_multi_graphs(x, y_list, annotations, x_label, y_label, title, file_name):
    """複数のグラフを1つの画像に出力する.

    Args:
        x (array_like, 1-D): x軸の値.
        y_list (array_like, 2-D): 全グラフのy軸値のリスト. （1つ目のリストの要素は, 各グラフのy値）
        annotations (array_like, 1-D): 全グラフの注釈リスト. （リストの要素は, 各グラフの注釈）
        x_label (str): x軸のラベル.
        y_label (str): y軸のラベル.
        title (str): グラフのタイトル.
        file_name (str): 出力画像のファイル名.
    """
    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    for i, y in enumerate(y_list):
        ax.plot(x, y, label=annotations[i])

    ax.legend(loc=0)
    plt.savefig(file_name)


def output_with_grid_search(kalman_filter, key, grid_params, annotations):
    """パラメータをグリッドサーチして, xiごとに1つの図に出力.

    Args:
        kalman_filter (KalmanFilterSteadyState): kalman filterクラス.
        key (str): グリッドサーチ対象の(kalman_filterクラスの)メンバ変数.
        grid_params (list): グリッドサーチに使う値リスト.
        annotations (list): grid_paramsごとにグラフに表示する注釈.
    """
    k = kalman_filter.k
    x_stack = []

    for param in grid_params:
        kf = deepcopy(kalman_filter)
        kf[key] = param
        kf.run()
        estimated_x_list = kf.estimated_x_list
        x_stack.append(estimated_x_list)

    x_stack = np.array(x_stack)
    x_stack = x_stack.transpose((2, 0, 1))  # x_stackの軸を変更. 軸のindexを(0, 1, 2)→(2, 0, 1)に並び替え.

    for i, xi_stack in enumerate(x_stack):
        output_multi_graphs(x=k, y_list=xi_stack, x_label='k', y_label=f'x{i}', annotations=annotations,
                            title=f'kalman_filter_change_{key}',
                            file_name=f'graph/kalman_filter_change_{key}_graph_x{i}.png')


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
    kf = KalmanFilterSteadyState(k, z, h, r, q, init_x, init_p)
    kf.run()
    x = kf.estimated_x
    logger.info(f'Q: \n{q}\n')
    logger.info(f'init X: \n{init_x}\n')
    logger.info(f'init P: \n{init_p}\n')
    result = f'推定値x（カルマンフィルター）: \n{x}\n'
    logger.info(result)
    print(result)

    """
    ************
    --- （2）---
    ************
    """
    # 推定誤差共分散pの初期値を変えて, グラフを出力.
    key = 'init_p'
    labels = [10**3, 10**2, 10**0, 10**-2, 10**-3]
    grid_params = [np.identity(3) * i for i in labels]
    default_kf = KalmanFilterSteadyState(k, z, h, r, q, init_x, init_p)
    output_with_grid_search(default_kf, key, grid_params, labels)

    # プラント雑音の共分散qを変えて, グラフを出力.
    key = 'q'
    labels = [10**4, 10**2, 10**0, 10**-2, 10**-4]
    grid_params = [np.ones(k.size) * i for i in labels]
    default_kf = KalmanFilterSteadyState(k, z, h, r, q, init_x, init_p)
    output_with_grid_search(default_kf, key, grid_params, labels)


if __name__ == '__main__':
    main()
