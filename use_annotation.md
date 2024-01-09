## LabelImg 를 활용한 annotation & data cleaning

작성자: 백광현 <br>
공식 repo:
[LabelImg](https://github.com/HumanSignal/labelImg)

### 공유사항
- coco format 과 pascal format 의 bbox 좌표 기준이 다릅니다. coco 는 `xmin, ymin, w, h` 인 반면 pascal 은 `xmin, ymin, xmax, ymax` 입니다.
- 설치 혹은 실행 중 에러가 발생하면 바로 말씀해주세요.


### 설치

1) 가상환경 생성. 저의 경우 conda create -n 'name' python=3.9 로 했더니 잘 됩니다!
2) 적당한 곳에 labelImg 를 git clone.
3) labelImg를 제외한 `requirements` 는 딱히 없고 `tqdm`, `argparse` 설치해주시면 됩니다.
4) 먼저 train.json 을 xml 파일로 변경 (coco -> pascal) (labelImg 에서 xml을 자동으로 읽어주기 때문)
`--xml_output_folder` 는 `/dataset/train` 이 아니라 `/dataset` 까지만 주셔야 합니다. 잘 돌아가면 `/dataset/train` 폴더 안에 이미지 마다 하나의 `.xml` 을 가지게 됩니다.
```bash
python annotation.py --function json_to_xml --xml_output_folder "/dataset"
```

1) labelImg 를 위한 필요 라이브러리 설치<br>

Mac
```bash
conda activate name
pip3 install pyqt5 lxml
make qt5py3
cd ./labelImg
python3 labelImg.py [IMAGE_PATH]
```
  <br>

Window

```bash
conda install pyqt=5
conda install -c anaconda lxml
pyrcc5 -o libs/resources.py resources.qrc
cd ./labelImg
python labelImg.py
python labelImg.py [IMAGE_PATH]
```

<br>

5) labelImg 에 `IMAGE_PATH` 로 `dataset/train` 을 주면 알아서 annotation 이 들어간 이미지가 보입니다.
6) labelImg GUI 가 뜨면 상단에 View 에서 `Display Labels` 를 체크해주시면 class 이름도 bbox 위에 표시됩니다.
7) bbox 는 드래그로 수정하시면 되고, cls 는 좌측 `Box Labels` 에 체크박스가 되어 있는 것들을 더블클릭 하면 하나씩 변경하실 수 있습니다.
8) 클래스를 수정할 때 10개의 클래스가 없다면 `labelImg/data/predefined_classes.txt` 내부에 클래스들을 한 줄에 하나씩 적으시면 됩니다.
9) 데이터를 보다가 bbox 혹은 cls 를 수정하게 된다면 바로바로 `ctrl+s` 를 눌러 저장하시면 `.xml` 에 반영됩니다.
10) 이후 `.xml` 을 하나의 `.json` 으로 만들 때. `--json_output_path` 는 확장자까지 적어주셔야 합니다.
```bash
python annotation.py --function xml_to_json --xml_folder './dataset/train' --json_output_path './modified.json'
```
