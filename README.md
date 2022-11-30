# Webapp access and deployment

## Accessing the webapp

To access the webapp, visit the following [webapp link](http://10.195.8.77:30401)

## Webapp deployment
The kubernetes deployment is separated into three folders with the following content:
- deployment: deployment for both server and frontend (PVC deployment is included in the server deployment)
- samba_mount: PV, PVC, and credentials for samba mount
- service: service definition for server and frontend

If the aforementioned files are not deployed, this can be done by running:
`kubectl apply -f kubernetes`

Checking status:
- deployment: `kubectl get pods -o wide` and `kubectl get deploy -o wide`
- samba_mount: `kubectl get pv -o wide` and `kubectl get pvc -o wide`
- service: `kubectl get svc -o wide`

Restarting:
- deployment: `kubectl rollout restart deploy server` and `kubectl rollout restart deploy frontend-kompose`

Redeploying:
- deployment: `kubectl apply -f kubernetes/deployment`
- samba_mount: `kubectl apply -f kubernetes/samba_mount`
- service: `kubectl apply -f kubernetes/service`


## Testing locally

Download the models [here](https://tumde-my.sharepoint.com/:f:/g/personal/gohdennis_tum_de/ElhWm3KCmuBBnYNWSS8i5IkBvcn6BypUdo9c1-xquNI6VQ?e=reHLLs)

Unzip to root to populate ```./models```, ```./data``` and ```./etc.``` (```os.getenv("MODEL")``` should point to ```./models```)

Retraining will modify this ```restructured_w_original_labels.json``` which points to the dataset being used by the transfer model.

The datast is also in the link above.

(Fix me if I'm doing it wrong)

Getting the backend running:
```
source etc/setup.sh
``` 
then restart VSCode if you haven't sourced yet.

Run the backend : 
```
python -m flask --app server.flaskr run
```
or :
```
cd server
python -m flask --app flaskr run
```
Getting the frontend running:
```
cd frontend
npm start
```

You can try curling an image to test the backend:
```
curl -F "file=@path/to/your/image.png" http://127.0.0.1:5000/api/predict
```

You can set the model to be usde by POSTing
```
curl -F "mode=<mode>" http://127.0.0.1:5000/api/mode
```

With mode being either transfer or selfsup.

# Transfer Learning Hyperparameter Opt

To run a transfer learning hyperparameter optimization, make sure your dependencies are installed correctly. After this, run the file notebooks/transfer_learning/ilsvrc_optuna_trial.py. You can modify the search space and optimization configuration as described in ilsvrc_optuna_demo.ipynb. 

Information on how to configure the search space can be found on the optuna website https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html.

# Contributing

## Confluence

To learn how to contribute to our common git repository, visit the following [confluence link](https://tum-ami-wiki.atlassian.net/l/cp/SPK0vB8E). You will learn:

- Where to locate files
- Where to add files
- How to name files
- Style conventions for python and java script
- Using the CI/CD
- Using docker

## Filesystem

|              |                                                                                                                             |
| ------------ | --------------------------------------------------------------------------------------------------------------------------- |
| /.vscode     | Common settings for the VSCode editor                                                                                       |
| /data        | Data Folder: Folder in which the data is stored locally. The folder is excluded from version control to keep our git light. |
| /etc         | Config folder: for project settings.json and environment setup files.                                                       |
| /frontend    | Frontend source folder.                                                                                                     |
| /notebooks   | Notebook folder: Store your research, demonstration and tutorial noteboooks here.                                           |
| /src         | Source folder: contains project python modules.                                                                             |
| changelog.md | Changelog: for keeping track of changes to the project                                                                      |

## Naming Conventions

|               |                                                                                                                                         |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Folder Names: | snake_case                                                                                                                              |
| File Names:   | snake_case                                                                                                                              |
| Branch Names: | feature_snake_case\_{jiraid} <br> fix_snake_case <br> docker_snake_case <br> experimental_snake_case\_{jiraid} <br> develop <br> master |

## Contribution Standards

### Pyhton

- If you are adding a pyhton script or module, which is reusing code from external sources, add a reference at the beginning of the file in the script/module docstring.
- All docstring must be generate with the autoDocstring VSCode extension to garantee a uniform look.
- All function and methods need a docstring except for nested functions.
- All files need to be formated with yapf.

### Notebooks

- A notebook either researches or showcases a functionality, which is or is to be implemented in the source folder
- A notebook needs to begin with an abstract, explaining what is being showcased
- It must be apparent what is being executed by the subsequent cells at any moment, without the reader needing to understand the implementation
- Additional files used by notebooks are to be stored in a /resources folder
- If a setup is required for the notebook, all locations at which the setup is to be performed need to be included in the description
- Avoid redundant notebooks and use descriptive naming
- Group related notebooks into subfolders

### Merge Requests

- Describe clearly which functionalities are being added
- Request review by at least one additional member
- Remove all unnecessary files
- Add test coverage for any features
- Add your changes to changelog.md
- Present your changes in the weekly meetings

# Installation and Setup

## Getting Python Environment Ready 

To prepare your environment, you need to install the dependencies locally.

```
pip install "python-dotenv[cli]"
pip install pipenv
pipenv install
```

Note, that we are using python3.8. If your system is running a different version, create an environment first. This will also prevent version conflict for the dependencies.

```
python3.8 -m venv ami
```

Next, run `source etc/setup.sh` to finalize the setup and restart your vscode instance.<br>
**Note**:  In **Windows**, please use the git bash to run the `setup.sh` file. 

The setup will only be workable on your local system (not kubernetes or docker), and only if you are using VSCode.


## Getting Web Development Environment Ready

### web dev environment setup
make sure you have the following packages installed:
- nodejs
```
sudo apt install nodejs
```
- after install nodejs, use npm to install yarn
```
npm install --global yarn
```

### Start the backend:
Open a terminal and change to the server folder. Type then:
```
export FLASK_APP=flaskr
export FLASK_ENV=development
flask run
```

### Start the frontend and reload automatically:
Open a terminal and change to the frontend folder. The first line will download all the dependencies that are needed for the frontend UI. The second line will start the frontend adn reload it automatically. Type then: 
    yarn 
    yarn run start
Open a second terminal and change to the frontend folder. Type then:
```
yarn install
yarn run start
```

## Verifying the installation

Once you have finished the setup, you will be able to import source modules in the integrated terminal interpreter. To check if it is working open your interpreter by running `python`. Import the preprocessing module. The import should not throw errors and your output should look like:

```
user@user-machine-id:~Group04$ python3
Python 3.8.10 (default, Jun 22 2022, 20:18:18)
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import preprocessing
```

Make sure, that the same line works inside of a jupyter notebook as well.

Finally, we need to make sure, that the environment variables are set. To do so run the following command and make sure, that the output point to your project root:

```
user@user-machine-id:~Group04$ echo $WORKINGDIR
/your/root/path/Group04
```

# Addresses and Imports

## Adresses

The workspace provides you with three environment variables, allowing you to address different location of the workspace

```
WORKINGDIR=Group04/
DATA=Group04/data
CONFIG=Group04/etc
```

Lets consider the case, in which I want access to the folder cropped_images in `/data` to obtain the path to the folder inside of a python script, I can run:

```
WORKINGDIR=Group04/
path = Path(os.getenv("DATA"), "cropped_images")
print(path)
```

The output will be:

```
/your/root/path/Group04/data/cropped_images
```

## Imports

To import a module from src, import it using the name beginning from the child directory of the src/ location. For the preprocessing module this would be:

```
import preprocessing
```

For a submodule this would be:

```
import preprocessing.data_loader
```
