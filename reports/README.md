# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [x] Create a git repository (M5)
* [x] Make sure that all team members have write access to the GitHub repository (M5)
* [x] Create a dedicated environment for you project to keep track of your packages (M2)
* [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [x] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [x] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)
* [x] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [x] Do a bit of code typing and remember to document essential parts of your code (M7)
* [x] Setup version control for your data or part of your data (M8)
* [ ] Add command line interfaces and project commands to your code where it makes sense (M9)
* [x] Construct one or multiple docker files for your code (M10)
* [x] Build the docker files locally and make sure they work as intended (M10)
* [x] Write one or multiple configurations files for your experiments (M11)
* [x] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [ ] Use profiling to optimize your code (M12)
* [ ] Use logging to log important events in your code (M14)
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [ ] Consider running a hyperparameter optimization sweep (M14)
* [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [x] Write unit tests related to the data part of your code (M16)
* [x] Write unit tests related to model construction and or model training (M16)
* [x] Calculate the code coverage (M16)
* [x] Get some continuous integration running on the GitHub repository (M17)
* [x] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [ ] Add a linting step to your continuous integration (M17)
* [ ] Add pre-commit hooks to your version control setup (M18)
* [ ] Add a continues workflow that triggers when data changes (M19)
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [x] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [x] Create a trigger workflow for automatically building your docker images (M21)
* [x] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [x] Create a FastAPI application that can do inference using your model (M22)
* [x] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [x] Write API tests for your application and setup continues integration for these (M24)
<!-- perhaps more tests need to be written -->
* [ ] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [x] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [ ] Make sure all group members have an understanding about all parts of the project
* [ ] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

77

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s214727, s214743, s214706, s214681

### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

For this project, we decided to utilize the third-party framework PyTorch Image Models (TIMM). We used TIMM's "resnet50.a1_in1k"
model architecture, which is a 50 layer deep convolutional neural network for image classification that makes use of residual connections. The TIMM API is easy to set up and configure, with flexibility for choosing the number of class numbers, convenient for our classification task. This allowed us to quickly integrate a working state-of-the-art model instead of having to design, refine, and test architectures ourselves from scratch, such that we could focus our efforts on topics more directly related to the learning objectives of this course. There is also an option in TIMM to load pretrained weights, which made it possible to fine-tune an existing model potentially saving a lot of compute.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

All packages for the project are managed using virtual environments, where we use conda for environment management. The project includes multiple requirements files that specify the required package dependencies needed to run our code. A new team member should make sure to have conda or miniconda installed and have ```conda``` accessible as a terminal command (e.g. in the PATH variable or on Windows through the dedicated Anacononda Prompt). Then they should clone the GitHub repository with ```git clone https://github.com/Lachrynier/mlops-project.git```. Afterwards navigate to the root of the repository and execute ```invoke create-environment``` to create the conda environment for the project and then activate the environment using ```conda activate proj```. Then execute ```invoke all-requirements``` which combines all requirements files and packages our project in editable mode (or run a subset of the requirements depending on what the team member would be working on).

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

We organized the project structure using cookiecutter with [this](https://github.com/SkafteNicki/mlops_template) template.
Our structure very closely follows the structure of the original template. The data folder includes data/raw and
data/processed (both of which are excluded from the git repo in .gitignore). The src/proj folder includes the following
files: api.py, data.py, evaluate.py, model.py and train.py. We did not use a visualize.py file. We used all three files
from the tests folder in the original template. We did not use the notebooks folder as we did not utilize any notebooks
for the project. In the project root directory we added a number of new files including cloudbuild.yaml, config_vertex.yaml etc.

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

The code is split over multiple files and folders, organized after functionality. The individual files are modularized such that they can be reused and easily tested. Type hinting is included most places and although there could probably be more documentation, most of the code is self-evident. Tools like linters and formatters help keep the codebase clean and easier to navigate. Code quality and formatting is important for consistency and readability. As the project grows, modularizing code and type hinting makes the code more controlled and predictable, which is crucial when collaborating. Documentation helps save time by not having to spend time constantly revisiting and analyzing source code when you forget how something works.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

We implemented a total of 5 tests. These include testing of api functionality e.g. asserting that we get the expected
status_codes and responses as well as making sure that our application makes valid predictions for a few test images.
We test our custom dataset class to verify that shapes and types are as expected, and lastly, we also test that for a
given valid input, our model provides a valid output.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

The total code coerage is 77% (subject to change), which includes all source code. Ideally this should be 100%, but even then, that is not a guarantee that code is error-free. Code coverage per definition only checks whether a line of code is executed, not whether the logic is flawless or if all edge cases are handled. But not having 100% means that some of the code is completely untested. With full coverage, there could be subtle logical errors that tests fail to catch, as tests might not account for all possible scenarios. It is important to modularize code as much as possible in this regard to reduce the number of input data combinations that need to be tested, making the tests simpler, and making debugging more isolated. The unit tests also do not guarantee that the interaction of the whole system works as expected.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

We used both branches and pull requests for our project. In general, when a group member would start working on a new
feature, they would create a new branch with the name ```feature/<whatever>``` which would then be dedicated to working
on that feature. Changes to the main branch were then made using pull requests. We attempted to make pull requests often
enough that we would only encounter a minimal number of merge conflicts. Additionally, We implemented branch protection
on the main branch such that at least one group member had to approve the request before it could be merged. Tests were
also automated using GitHub actions such that they would be run every time changes were merged to the main branch or pull requests were made. The latter reduced the possibility that merging to main breaks code as we would only merge when the workflow passed flawlessly.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

As our data consists of many files nested in folders, DVC proved to be extremely ineffective. We thus collected all the raw data into a tar file and set up DVC remote on a GCP data bucket. We did not further use DVC. If the dataset of a project changes over time, then it would be useful for reproducibility as we could trace back the version of the dataset that was used to train a specific model. Then we could along with git (and perhaps also a random seed) recreate the exact conditions that were used. DVC also supports working with multiple datasets simultaneously. For example, when new data is introduced, DVC can provide a layer of flexibility to manage different dataset versions, facilitating experimentation and collaboration without overwriting or losing previous data.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

We have a unit testing workflow that tests for multiple operating systems (Windows and Ubuntu) and one could also add another Python version. The workflow can be seen [here](https://github.com/Lachrynier/mlops-project/blob/main/.github/workflows/tests.yaml). We make use of caching with ```cache: 'pip'```. We have an environment set up called ```gcp``` that allows us to authenticate with GCP in the workflow using secrets: ```${{ secrets.GCP_CREDENTIALS }}```. The environment also allows for setting up environment variables. The workflow sets up Google Cloud SDK, sets up Python, installs dependencies from our requirements files and packages our module as editable just like we would do locally. It then runs tests on all test files in our test folder and calculates the code coverage to assess how much of the codebase is being tested. This helps us identify any untested areas and maintain a high level of test coverage over time. The workflow triggers whenever we push to the main branch or make a pull request to the main branch. This ensures that every proposed change is thoroughly tested before being merged into the primary codebase. By automating this process, we eliminate the need for manual testing, which would otherwise be time-consuming and prone to human error.

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We configured experiments using Hydra for managing configuration files and Typer for command line interaction in some places. To run an experiment, we first need to preprocess data which can be done with ```python data.py --num-classes {number of classes}```. Then one needs to set this in ```configs/hydra/model.yaml``` along with specifying other configurations under ```configs/hydra```, after which one can run ```python train.py``` to begin training.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

We made use of Hydra config files. When an experiment is run, we log to wandb and pass all hydra config values to the wandb run config. Model weights are saved as artifacts under the wandb run. ...

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

--- question 14 fill here ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

For our project we made docker files for constructing images for training, inference (backend API), and the frontend application. To run ... Here is a link to ...

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

We used VS Code's Python Debugger extension and logging/printing. Logging allows us to track the progress of the program. The debugger allows to add breakpoints, step through code, inspect variables, and interact with the program through a debugging terminal. The breakpoints can be configured to only trigger when certain conditions are true, which can be quite useful. Ideally all debugging is done locally as it can be extremely time consuming and tedious to wait for docker files to build or GCP apps to deploy. Profiling ?...

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

We used: Cloud Storage, Cloud Run, Artifact Registry, Cloud Build, Vertex AI. Cloud Storage is used for mounting to GCP instances, DVC, and saving models. This is done through a bucket. Cloud Run is used to deploy applications. Here we hosted our backend and frontend. Artifact Registry is used to store our docker images such that they can be used by the other applications/services. Cloud Build is used for continuous deployment based on triggers such as pushing to the main branch or creating a new git tag. Vertex AI lets us create train jobs with specified resources and docker image along with configs.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

We used Vertex AI instead of the Compute engine. ...

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

A screenshot of the GCP bucket can be seen here (subject to change?): [link](figures/bucket.png)

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

A screenshot of the GCP artifact registry can be seen here (subject to change?): [link](figures/registry.png)

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

A screenshot of the GCP Cloud Build history can be seen here (subject to change?): [link](figures/registry.png)

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

Something ... Vertex AI ... Something

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

We made a backend FastAPI API for inference using our trained model stored on GCP bucket or locally. An image in the format of .jpg, .jpeg or .png can be POSTed to the endpoint /predict, and the API will behind the scenes perform a data transform and call the model, returning a json with the predicted class and class probabilities. We also have an endpoint for GETting the class names. We also have a lifespan context manager ensures a clean shutdown process. We also have error handling that takes care of illegal user input and internal errors. We also have a frontend explained in another question.

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

The API was deployed using GCP Run. We have also set it up flexibly so that the application can be run locally as well. The API was initially built locally for faster development and easier debugging, and then when it worked we began deploying it to GCP. We had to build a docker image either locally or through Cloud Build and push it to our artifact registry for the application to use. To invoke inference with the service, one can `import requests` and call `response = requests.post(predict_url, files={"image": image})` filling out `predict_url` with the corresponding URL where it is hosted and concatenating the endpoint, and `image` with the corresponding image file. The response is a json with prediction and probability entries.

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

We performed unit testing on the API. We downloaded some external (not part of our dataset) images found on google in the different image formats that our API supports. We then check for status codes, that the content type is correct (application/json), that the prediction is a valid class integer, that the probabilities sum up to 1. We could probably add some more tests that sends illegal input to test how such is handled. We did not perform load testing. This could be done by simulating user traffic with corresponding randomized input, e.g. using Locust, and see how our application holds up for different numbers of users and spawn rates. Correspondingly, we could scale our application to use better cloud hardware if needed.

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

--- question 26 fill here ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

We ended up using (subject to change) $2.40 in credits. Vertex AI (expensive computational resources) was the most expensive followed by Cloud Run (long uptime) and Cloud Storage (a lot of data transfer). Working in the cloud has both pros and cons. Pros are it is easy scale up and not have to worry about hardware. It is flexible and has convenient services for a lot of things. Cons are that it can be annoying to integrate all things and make different services and external platforms communicate. It is also weird how you cannot set a cap on the billing amount, but only set notifications for when limits are reached. This was not really a concern for us with the free credits. However, if one inserted their credit card then they would have to be extremely cautious and constantly monitor GCP and somehow indirectly set limits as GCP does not allow it.

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

--- question 28 fill here ---

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 29 fill here ---

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- question 30 fill here ---

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

--- question 31 fill here ---
