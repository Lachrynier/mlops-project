# MLOps Project, group 77
Course project for "02476 Machine Learning Operations" at DTU.

[//]: <> (- Overall goal of the project)

### Goal

The goal is to classify images from the Caltech-256 dataset into their respective categories using DL frameworks, emphasizing reproduciblity, code organization, and leveraging different MLOps tools, rather than achieving a high performance score.

[//]: <> (- What framework are you going to use, and you do you intend to include the framework into your project?)

### Framework

We will apply the framework PyTorch Image Models (TIMM) for the project. TIMM contains a large number of network architectures along with many pre-trained models etc. for computer vision.

[//]: <> (- What data are you going to run on \(initially, may change\))

### Data

The dataset we will work with is the [Caltech-256 Image Dataset](https://www.kaggle.com/datasets/jessicali9530/caltech256). The dataset consists of 30607 images spanning 257 different categories - hence an average of 119 images per category. Categories include a wide range of different objects such as animals, vehicles and household objects. Due to the size of the dataset, we may use a subset as a start for faster development.

[//]: <> (- What models do you expect to use)

### Models

We expect to take pretrained models with CNN-architectures as our base building block and finetune or use transfer learning to adapt to the classification task. Potential candidate models from TIMM are ResNet, EfficientNet and MobileNet.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
