conda create --name chrisbase python=3.10 -y; conda activate chrisbase;
rm -rf build dist src/*.egg-info;
pip install build; python3 -m build;
pip install twine; python3 -m twine upload dist/*;
rm -rf build dist src/*.egg-info;

sleep 3s; clear;
conda create --name chrisbase python=3.10 -y; conda activate chrisbase;
sleep 5s; clear;
pip install --upgrade chrisbase; pip list;
