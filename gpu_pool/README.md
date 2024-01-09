# Message-queue-based-learning-server

Message-queue-based-learning-server는 분산되어 있는 gpu 서버들이 메시지 큐를 통해 자동으로 학습할 수 있는 환경을 제공합니다.<br>
based by Naver Boostcamp 조창희 camper<br>
modified by 백광현

## Project Structure

```
${PROJECT}
├── gpu_pool          # env, server, client, ...
├── ai 관련 폴더
|   ├── train.py
|   └── ... 등
├── ai 관련 파일
├── ... 등
├── README.md
├── message.json      # 메시지 큐에 보낼 메시지
└── requirements.txt
```

## Getting Started

### Requirements

Server 경우:
- MySQL Server (MySQL >= 8.0) 사전 설치 필요
  - ```apt-get install mysql-server```


Client 경우:
- ```pip install pymysql```

### Usage

- **사용 전에 env.py 수정이 필요합니다.**
  - `CAMPER_NAME`, `MESSAGE_COLUMNS` 아래 예시를 참고하세요.
- **server 는 동일한 ip의 서버 하나만 열려 있으면 되고, Client 들은 env.py 만 수정 후 `python gpu_pool client` 로 메시지를 받을 수 있습니다.**

#### Server

아래의 코드를 실행하기전 env.py에 비어 있는 항목을 모두 기입한 후 실행해야합니다.
ai_stages에서 서버를 할당받을 때 주어진 포트 번호 중 한 개를 선택하여 사용하시기 바랍니다.
```
python gpu_pool server
```

#### Client

```
# foreground
python gpu_pool client

# background
nohup python gpu_pool client &
```

#### Message

```
# Push message to queue
python gpu_pool push


#Check message list
python gpu_pool message
# or
python gpu_pool message -id 2


#Check error message list
python gpu_pool error
# or
python gpu_pool error -id 3
```

### Example

#### env

```
# MySQL Server 관련
MYSQL_SERVER_IP = '10.28.xxx.xxx'
MYSQL_SERVER_PORT = 30034
MYSQL_SERVER_USER = 'hee'
MYSQL_SERVER_PASSWORD = 'password'

# 큐에 푸시하려는 메시지를 작성한 파일의 주소
MESSAGE_FILE_PATH = './message.json'

# 이름
CAMPER_NAME = "bkh"

# 메시지 큐에 담을 내용
# id, branch, train_path, receiver, pushed의 위치를 수정하지 마세요.
MESSAGE_COLUMNS = ['id', 'branch', 'train_path', 'receiver',

                  'exp_name', 'seed',
                  ...,

                  'pushed']

# 이하 수정 x
ARGS_EXCEPTION = ['id', 'branch', 'train_path', 'receiver', 'pushed']
BRANCH_INDEX = 1
TRAIN_INDEX = 2
RECEIVER_INDEX = 3
```

#### message

```
# ./message.json
# branch, train_path 항목 필수 // id, pushed 항목은 x
# receiver 에 메시지를 처리할 사람을 특정(CAMPER_NAME) 하거나 'any'로 누구나 처리할 수 있도록 지정
# env.py의 MESSAGE_COLUMNS을 바탕으로 작성

{
  "train_path": "./train.py",
  "receiver": "any",

  "exp_name": "test",
  "seed": "666",
  ...
}

```
