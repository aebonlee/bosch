Python 스크립트(노트북/코드스페이스/로컬 공통)
import zipfile, glob, os
os.makedirs("data", exist_ok=True)
for z in glob.glob("data/*.zip"):
    with zipfile.ZipFile(z) as f:
        f.extractall("data")
print("Unpacked into ./data")

터미널(Windows 7-Zip)


# 7-Zip 설치 후
cd data
7z x train_numeric.csv.zip
7z x test_numeric.csv.zip
7z x train_date.csv.zip
7z x test_date.csv.zip
7z x train_categorical.csv.zip
7z x test_categorical.csv.zip


중요: 깃허브 웹에서 “Download ZIP”으로 받으면 LFS 포인터만 포함될 수 있어요.
반드시 git lfs install 후 git clone(또는 git lfs pull)로 받도록 안내하세요.
