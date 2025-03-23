set -e
set -x

pip install gdown
mkdir -p .temp/
# URL: https://drive.google.com/file/d/1ox0T940B_9_pZquKFu7oTqWJPeiYXaoA/view?usp=drive_link
gdown "https://drive.google.com/uc?id=1ox0T940B_9_pZquKFu7oTqWJPeiYXaoA" -O .temp/data.zip
unzip  -o .temp/data.zip -x "*.DS_Store" "__MACOSX/*" -d ./

rm -rf .temp/