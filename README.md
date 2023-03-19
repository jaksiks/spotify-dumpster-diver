# cse-6242-project
CSE 6242 Group Project | Spotify Dumpster Diver


# Installing Anaconda

* On Windows Linux (WSL): https://gist.github.com/kauffmanes/5e74916617f9993bc3479f401dfec7da

# Working with the Environment

* [Conda Cheatsheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)

* Install [Anaconda](https://www.anaconda.com/products/distribution) and from a Anaconda prompt:

```bash
conda env create -f environment.yml
```

* Activate the environment (do this every time you open your terminal to load dependencies in virtual environment)

```bash
conda activate cse-6242-project
```

* Update your environment (install dependencies that others added)

```bash
conda env update
```

* Update environment.yml (after installing a dependency you want to share)

```bash
conda env export --from-history>environment.yml
```

# Working with Django

## Start Here

* [Intro Tutorial](https://docs.djangoproject.com/en/4.1/intro/tutorial01/#)

## Start Your Local Server
1. Navigate to ```/cse-6242-project/diver``` in a terminal window >> This is the main Django app directory
2. Run ```python manage.py runserver``` >> This will start a web server locally at http://127.0.0.1:8000/
    * Recommend opening localhost (http://127.0.0.1:8000/) in Chrome Incognito Mode
    * This will "take over" your existing terminal session; if you want to interact with git, etc. while your local server is running, you can open another terminal window

## Django File Structure
<img width="177" alt="image" src="https://user-images.githubusercontent.com/10931549/226191252-8044bc25-2ce0-4778-b9df-e01ff94f9002.png">

* Top-level ```/diver``` >> This is where our "diver" app lives
* ```/diver/diver``` >> App settings and routes
* ```/diver/dumpster_diver``` >> This is where most people will spend their time; pages ("views"), assets (JS), etc.

## Working with Views
<img width="668" alt="image" src="https://user-images.githubusercontent.com/10931549/226192083-c74647f1-e9c7-4af1-9224-6de86899ef7f.png">

