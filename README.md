# NCKH_LR
NCKH - logistic regression

### 1. Set enviroment:

- python minimum version: ***>=3.6***
- install virtual enviroment: `pip3 install virtualenv`
- create virtual enviroment: `virtualenv $ENV_NAME`
- activate vitualenv (linux): `source $ENV_NAME/bin/activate`
- install list package : `pip3 install -r requirements.txt`
> ENV_NAME=envLR

**options**
- add virtual enviroment to jupyter notebook: ` python3 -m ipykernel install --user --name=$ENV_NAME`
- deactivate (linux): `deactivate`
### 2. Run:
`python3 index.py`

- all model will be saved in **result/model**
- accuracy result will be wrote in file *log*
- ROC training result saved in **result/ROC**
