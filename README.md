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

* Activate the environment

```bash
conda activate cse-6242-project
```

* Update your environment

```bash
conda env update
```

* Update environment.yml (after installing a dependency you want to share)

```bash
conda env export --from-history>environment.yml
```

# Working with Django

* [Intro Tutorial](https://docs.djangoproject.com/en/4.1/intro/tutorial01/#)
* 
