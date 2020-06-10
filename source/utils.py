import datetime
import os


def make_log_path(file):
    """ログファイルのパスを生成

    Args:
        file (str): 呼び出し元の__file__

    Returns:
        str: ログファイルのパス
    """
    log_path = 'log/' + datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S') \
               + '_' + os.path.basename(file).rstrip('.py') + '.log'
    return log_path
