# machine-learning

sentiment calculator ML project for Task 2 SE-2026. 

## initialisation
```bash
uv sync
```

this sets up the uv environment. 

## training

start the jupyter server with this:
```bash
uv run --with jupyter jupyter lab
```

then look at the terminal and copy the jupyter server address (probably something like `localhost`). input
it into vscode. 

if you want, just run it, then quit when safe. it really just needs the .venv to be ready.

The data has already been pretrained in [Model_Development/csv](Model_Development/csv). 

> [!WARNING]
> You are going to have to deal with 8 million rows of data. This will be extremely resource and time intensive. 
> 
> To give you an estimate, 32gb ram + NVIDIA 1070 Ti trains in about 10 minutes, but on a Surface Laptop 4, it trains
> in about 1.5 hours (dependent on the GPU). This is due to the SBERT model feature (more in notebook 2). 
> 
> Using a codespace is not recommended, as training will likely fail (because it is a simple VM). It is fine to evaluate 
> code and run the web server with a trained model, however training is not feasible. 
> 
> looking at you *cough* *cough*

## deployment

the deployment runs as a flask app. 

to run it, run this command:
```bash
uv run python Deployment/main.py
```

it will assume that the training is done, and fail if the models don't exist in their correct folder.

### training then deployment in one go
there is also a command line arg for training, then deploying without touching the jupyter notebooks:
```bash
uv run python Deployment/main.py train-for-me=true
```
again, this will take some time, but it will save the model permanently locally. 
