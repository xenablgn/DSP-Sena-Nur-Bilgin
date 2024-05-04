summary: PW2 - Practical work 2 id: pw2 categories: industrialisation tags: industrialisation status: Published authors:
Alaa BAKHTI Feedback Link: https://github.com/EPITA-courses/dsp_practical_work/issues/new

# Data Science in production assignment 2

The goal of this practical work is to industrialize the ML model you have created in the previous session. To do that you will be going over the following steps:

### Step :zero: : Creating a test to make sure, the refactoring step does not change the previous code behavior

In this practical work, you will be refactoring the code you have done in `pw1`. As we saw in the class, when refactoring your code you should not change the code behavior but only the code internal structure. To make sure of that, you need to test your code by doing the following:

1. Before refactoring, save the processed dataframe (encoded dataframe for example) in a parquet file (parquet is used instead of csv to preserve the types of each column).

```python
processed_df.to_parquet('/my/filapth/processed_df.parquet', index=False)
```

2. After refactoring, load the `processed_df.parquet` dataframe

```python
import pandas as pd

expected_processed_df = pd.read_parquet('/my/filapth/processed_df.parquet')
```

3. Check that the processed dataframe after refactoring is the same as the one saved saved locally

```python
pd.testing.assert_frame_equal(actual_processed_df, expected_processed_df)
```

This test can be executed after each refactoring of a chunk of code to make sure, the previous code behavior did not change

### Step :one: : notebook reorganization

In this step, you are asked to:

1. :octocat: Create a new branch `pw2` from `main` (make sure to merge the `pw1` branch into `main` before)
2. Create a copy of the notebook `house-prices-modeling.ipynb` and name it `model-industrialization-1.ipynb`. All the following steps should be done in this notebook
3. Use a persistent pre-processing object in the different data preparation steps (encoders, etc) for example use `OneHotEncoder` instead of `pandas.get_dummies`
4. Remove the [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html) if you are using it and make the required changes
5. For each of the transformers you are using (encoder, scaler, etc), replace the `fit_transform` method
   by the 2 methods `fit` for fitting the transformer and `transform` for transforming the data
6. Instead of splitting your dataset after pre-processing, split it just after loading the data from the csv file (to avoid data leakage) and make the required changes in the notebook code.
7. Add a `Model building` section composed of `Model training` and `Model evaluation` sub-sections. Here the `train.csv` file should be used
    - `Model training` sub-section should contain the following code:
        - Dataset loading and splitting into train and test
        - Preprocessing and feature engineering of the train set
        - Model training
    - `Model evaluation` sub-section should contain the following code:
        - Preprocessing and feature engineering of the test set
        - Model predictions on the test set
        - Model evaluation
8. Add a new section `Model inference` after the `Model building` section. In this section you need to write the code for:
    - Reading data from a given file (`test.csv`  file in your case)
    - Preprocessing and feature engineering of this data
    - Predicting the house prices of this data

### Step :two: : object persistance

Now that you seperated your notebook in different sections, let's start making some changes in the code to make it more "production ready"

1. :file_folder: Create a `models` folder in the root of your project (dsp-firstname-lastname). All the objects that you will be persisting in the following steps should be saved in this folder.

2. **Persist the trained model**

   As we saw during the course, to use the trained model in production, we need to save it locally after the training phase. You will be using [joblib](https://joblib.readthedocs.io/en/latest/persistence.html) for this
    1. Save the model under the name `model.joblib` in the `models` folder with the [joblib.dump()](https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html#joblib.dump) function
    2. In the `Model inference` section, instead of using the model instance that you already have in memory, load it from the `models` folder with the [joblib.load()](https://joblib.readthedocs.io/en/latest/generated/joblib.load.html#joblib.load) function

1. **Persist the encoders and scalers**

   When making model predictions during the inference phase for example, we need to apply the same preprocessing and
   feature engineering as we did during the training phase. Therefore, we need to persist the different objects we used
   for the data preparation (encoders, scalers, etc)

    1. Save the different encoder and scaler objects in the `models` folder
    2. In the `Model inference` section, instead of using the encoder and scaler instances that you already have in
       memory, load them from the `models` folder

### Step :three: : code refactoring

Now that you saved the different objects (encoders, model, etc), you need to apply the refactoring techniques we saw during the class. Please refer to the class slides for more information.

Here are some of these techniques

- Renaming variables to descriptive names
- Removing mutability if possible
- Making sure to respect the [PEP8](https://peps.python.org/pep-0008/) coding style
- Extracting code in functions (you should not use classes !)

At the end of this step you need to have the following functions along with others (up to you to decide how to design your code):

- **build_model**: this function will be responsible for the orchestration of the different steps for the model building phase

```python
import pandas as pd


def build_model(data: pd.DataFrame) -> dict[str, str]:
    # Returns a dictionary with the model performances (for example {'rmse': 0.18})
    pass
```

- **make_predictions**: this function will be responsible for the orchestration of the different steps of the inference phase

```python
import pandas as pd
import numpy as np


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    # the model and all the data preparation objects (encoder, etc) should be loaded from the models folder
    pass
```

### Step :four: : code extraction in python modules

In this step, you will be extracting the different functions you wrote from the notebook to Python modules

1. :file_folder: Create a folder `house_prices` in the root of your project

2. :open_file_folder: Create 3 Python modules in this folder: `preprocess.py`, `train.py`, `inference.py`

3. Extract the different notebook functions to the Python modules
    - `preprocess.py`: for cleansing and feature engineering functions
    - `train.py`: for the functions used by the `build_model` function and `build_model` itself
    - `inference.py`: for the functions used by the `make_predictions` function and `make_predictions` itself

   :warning: You should not have any dependency between the `train.py` and `inference.py` modules
   (no `from house_prices.train import my_function` in the inference module)

4. Create a new notebook `model-industrialization-final.ipynb` in the notebooks folder with 2 sections `Model building` and `Model inference` and copy the following code in each section
    - `Model building`

   ```python
   import pandas as pd
   from house_prices.train import build_model
   
   training_data_df = pd.read_csv('../data/train.csv')
   model_performance_dict = build_model(training_data_df)
   print(model_performance_dict)
   ```
    - `Model inference`:

   ```python
   import pandas as pd
   from house_prices.inference import make_predictions
   
   user_data_df = pd.read_csv('../data/test.csv')
   predictions = make_predictions(user_data_df)
   predictions
   ```

### Step :five: : type hinting, docstring and code linting

1. Add type hinting in the different functions (function parameters and return values). Check the
   [official documentation](https://docs.python.org/fr/3/library/typing.html) for more information.
2. Add docstring for each of the function you've created (guideline => [website](https://note.nkmk.me/en/python-docstring/))
3. Lint your code with *flake8* and make sure you do not have any errors
    1. Install `flake8`
    2. execute it in the house_prices folder
   ```python
   $ flake8 house_prices
   ```
   If you have any errors or warnings, make sure to correct them

    3. After correcting the different lint problems, take a screenshot of your terminal, name it to `flake8_report.jpg`
       and submit it in the *Teams* assignment tab

### Step :six: (**Bonus**): create a python package for your model :muscle:

This step is not mandatory, if you do it you will get extra points :heart_eyes_cat:

Instead of importing the functions from the source code in the `house_prices` folder, import them from a Python package

- Create a Python package called `house_prices`, you can
  follow [this tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- Install the package in your environment and make sure to update the `requirements.txt` file

:raised_back_of_hand: if you want to install the package and be able to make changes in the code at the same time, you
can install it in edit mode using `pip install -e path/to/the/package`. However, this should never be done in production

### Step :seven: : merge the pw2 to main and make submission

Now that you completed the different industrialization steps, merge the `pw2` branch to `main` :octocat: and make your submission in teams. Refer to the next section for the assignment submission instructions

## Assignment submission instructions

In the Teams assignment tab, submit:

- the **URL of the `model-industrialization-final.ipynb` notebook** in the main branch (not the link to the root of your github repo !)
- the screenshot `flake8_report.jpg` you took in step 5

:raising_hand: Before that, make sure that:

- the `pw2` branch is merged in `main`
- the output of the notebook cells is displayed in GitHub
- you do not have any errors in the `flake8` report

As always, here are some grading criteria:

- For model building and model inference steps:
    - Data is splitted into `X_train` and `X_test` at the start of the training pipeline (before pre-processing)
    - No fit during inference !
    - The same pre-processing and feature engineering functions are used for the 2 steps (you should not have a function only for inference and another one for training)
    - The `build_model` and `make_predictions` functions are displaying the expected output in the `model-industrialization-final.ipynb` notebook
    - The objects (sclaer, encoder and model) are loaded from a file during model inference and saved in a file during model building

- Coding quality
    - The length of the different functions is less than 10 lines of code
    - respect of the PEP8 standards
    - type hinting
    - docstring
    - no linter errors (with Flake8)

- Other
    - No use of classes
    - No use of the `fit_transform` method or the [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html) class
    - up to date requirement.txt file
    - respect of the folder structure
    - respect of Git instructions (`pw2` branch, commit messages, etc)

:muscle: Good luck :woman_technologist:    :technologist:
