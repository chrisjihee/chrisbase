mamba create --name chrisbase python=3.9 -y; mamba activate chrisbase;
rm -rf build dist src/*.egg-info;
pip install build; python3 -m build;
pip install twine; python3 -m twine upload dist/*;
rm -rf build dist src/*.egg-info;

sleep 3; clear;
mamba create --name chrisbase python=3.9 -y; mamba activate chrisbase;
sleep 5; clear;
pip install --upgrade chrisbase; pip list;
