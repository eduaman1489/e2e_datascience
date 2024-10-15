echo [$(date)]: "Start building the environment"
echo [$(date)]: "creating env with python 3.8 version" 
conda create --prefix ./E2E_DATASCIENCE python=3.10 -y
echo [$(date)]: "Activating the environment" 
source activate ./E2E_DATASCIENCE
echo [$(date)]: "Installing the requirements file" 
pip install -r requirements_dev.txt
echo [$(date)]: "Environment build !!!" 

## bash build_env.sh