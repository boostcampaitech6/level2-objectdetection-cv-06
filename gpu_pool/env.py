import os

CWD = os.getcwd()

# MySQL 서버 관련
MYSQL_SERVER_IP = 
MYSQL_SERVER_PORT = 
MYSQL_SERVER_USER = "CV-06"
MYSQL_SERVER_PASSWORD = "cv06"

# 이름
CAMPER_NAME = "bkh"

# 큐에 푸시하려는 메시지를 작성한 파일의 주소
MESSAGE_FILE_PATH = "./message.json"

# id, branch, train_path, receiver, pushed의 위치를 수정하지 마세요.
MESSAGE_COLUMNS = ["id", "branch", "train_path", "receiver", "name", "seed", "pushed"]

# 이하 수정 x
ARGS_EXCEPTION = ["id", "branch", "train_path", "receiver", "pushed"]
BRANCH_INDEX = 1
TRAIN_INDEX = 2
RECEIVER_INDEX = 3
