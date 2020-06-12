import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from numpy.linalg import inv, multi_dot

logger = logging.getLogger(__name__)


def load_dataset(csv_path):
    return pd.read_csv(csv_path)


def batch_least_squares(_z, _h, _r):
    # バッチ型最小２乗法
    estimated_x = multi_dot([inv(multi_dot([_h.T, inv(_r), _h])), _h.T, inv(_r), _z])
    return estimated_x


def iter_least_squares(_z, _h, _r, _init_x, _init_p):
    # 逐次型最小２乗法
    xi = _init_x
    pi = _init_p

    for i in range(len(_z)):
        hi = _h[i].reshape(1, -1)                   # shape (1, 3)
        ri = _r[i][i]                               # shape (1, 1)
        zi = _z[i]                                  # shape (1, 1)
        si = multi_dot([hi, pi, hi.T]) + ri         # shape (1, 1)
        wi = multi_dot([pi, hi.T, inv(si)])         # shape (3, 1)
        xi = xi + np.dot(wi, (zi - np.dot(hi, xi))) # shape (3, 1)
        pi = pi - multi_dot([wi, si, wi.T])         # shape (3, 3)

    return xi, pi


if __name__ == '__main__':
    # logger設定
    logging.basicConfig(
        level=logging.INFO,
        # filename='log/least_squares.log'
    )

    # データセットの読み込み
    dataset_path = 'dataset.csv'
    dataset_df = load_dataset(dataset_path)

    # 行列生成
    k = dataset_df['k'].to_numpy()
    z = dataset_df['z'].to_numpy().reshape(-1, 1)
    h = np.stack([np.ones(k.size), k, k*k], axis=1)
    r = np.diag(np.where(k % 2 == 0, 4, 1))
    # logger.info(f'H: \n{h}\n')
    # logger.info(f'R: \n{r}\n')

    """ --- バッチ型最小２乗法 --- """
    x = batch_least_squares(z, h, r)
    print(f'推定値x（バッチ型）: \n{x}\n')

    """ --- 逐次型最小２乗法 --- """
    init_x = np.zeros((3, 1))
    init_p = np.identity(3) * 10**6
    x, p = iter_least_squares(z, h, r, init_x, init_p)
    print(f'推定値x（逐次型）: \n{x}\n')

    plt.scatter(k, z)
    plt.show()
    plt.savefig('dataset_graph.png')



