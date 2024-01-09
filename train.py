import argparse
import os
from importlib import import_module

import datetime
from pytz import timezone
import time

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=666, help="random seed (default: 666)"
    )
    parser.add_argument("--name", type=str, default="test", help="experiments name")
    parser.add_argument(
        "--receiver", type=str, default="all", help="random seed (default: 666)"
    )
    args = parser.parse_args()
    print(args)

    start_time = datetime.datetime.now(timezone("Asia/Seoul"))
    print(f"학습 시작 : {str(start_time)[:19]}")

    time.sleep(10)

    end_time = datetime.datetime.now(timezone("Asia/Seoul"))
    print(f"학습 끝 : {str(end_time)[:19]}")

    # 학습 소요 시간 계산 및 출력
    elapsed_time = end_time - start_time
    print(f"학습 소요 시간: {elapsed_time}")
