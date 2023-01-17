conda create --name chrisbase python=3.10 -y; conda activate chrisbase;
rm -rf dist src/*.egg-info;
pip install build; python3 -m build;
pip install twine; python3 -m twine upload dist/*;
rm -rf dist src/*.egg-info;

conda create --name chrisbase python=3.10 -y; conda activate chrisbase;
pip install --upgrade chrisbase; pip list;
