# cse-6242-project

```
 _____             _   _  __        ______                           _             ______ _                
/  ___|           | | (_)/ _|       |  _  \                         | |            |  _  (_)               
\ `--. _ __   ___ | |_ _| |_ _   _  | | | |_   _ _ __ ___  _ __  ___| |_ ___ _ __  | | | |___   _____ _ __ 
 `--. \ '_ \ / _ \| __| |  _| | | | | | | | | | | '_ ` _ \| '_ \/ __| __/ _ \ '__| | | | | \ \ / / _ \ '__|
/\__/ / |_) | (_) | |_| | | | |_| | | |/ /| |_| | | | | | | |_) \__ \ ||  __/ |    | |/ /| |\ V /  __/ |   
\____/| .__/ \___/ \__|_|_|  \__, | |___/  \__,_|_| |_| |_| .__/|___/\__\___|_|    |___/ |_| \_/ \___|_|   
      | |                     __/ |                       | |                                              
      |_|                    |___/                        |_|                                              
```

CSE 6242 Group Project | Spotify Dumpster Diver | Team 031

# Description

This is the "Spotify Dumpster Diver" project, a CSE 6242 Group Project by Team 031. This application uses features derived from the MillionSongDataset (MSD) and recommends “dumpster” finds, unpopular songs, based off of the user’s top Spotify songs. Additionally, the application provides an interactive web interface for users to explore and analyze various aspects of their Spotify listening habits and visualize the Dumpster Diver’s recommendations and compare them to the user’s songs and song’s Spotify would recommend.

The Spotify Dumpster Diver was built using Python with a Django frontend and uses packages such as Pandas and SciKitLearn for the backend. Additionally, the Dumpster Diver uses spotipy to interface with the Spotify Developer API.

# Installing Anaconda

* Install Anaconda from https://www.anaconda.com/download/

* NOTE: If on Windows Linux (WSL): https://gist.github.com/kauffmanes/5e74916617f9993bc3479f401dfec7da

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
<img width="1161" alt="image" src="https://user-images.githubusercontent.com/10931549/226196102-9bf8e028-a166-42e4-9358-34aaa27ac688.png">

* Top-level ```/diver``` >> This is where our "diver" app lives
* ```/diver/diver``` >> App settings and routes
* ```/diver/dumpster_diver``` >> This is where most people will spend their time; pages ("views"), assets (JS), etc.

## Working with Views
<img width="668" alt="image" src="https://user-images.githubusercontent.com/10931549/226192083-c74647f1-e9c7-4af1-9224-6de86899ef7f.png">

## Working with index.html

* In order to see a variable passed to the frontend via ```context```, use handlebars. I.e. if you pass the variable "tracks" into the context, use ```{{tracks}}``` in index.html to access the context variable
* For an exmaple of fancier templating methods, here's [how to do a for loop](https://www.geeksforgeeks.org/for-loop-django-template-tags/) in index.html

## config.yml

* Create Spotify Developer Account for API access:
    - Visit the Spotify Developer Dashboard at https://developer.spotify.com/dashboard/applications.
    - If you don't have a Spotify account yet, sign up for a free account by clicking the "Sign Up" button at the top-right corner of the page. If you already have a Spotify account, click the "Log In" button and enter your credentials.
    - After logging in, click the "Create an App" button in the Spotify Developer Dashboard
    - Fill out the "Create an App" form with the required information, such as the app name, description, and reasons for building the app. Agree to the terms and conditions, and then click "Create"
    - Once your app is created, you will be redirected to the app's management page. Here, you will find your "Client ID" and "Client Secret" 
    - Click on the "Edit Settings" button on the app management page. In the "Redirect URIs" field, add the following URI: http://localhost:8888/callback. Click "Save" at the bottom of the page.
* Create / Update your ```config.yml``` file in the `cse-6242-project\diver\spotify` directory in the following format:

```
spotify:
    id: "<id>"
    client_id: "<client_id>" 
    client_secret: "<client_secret>"
    redirect_uri: "http://localhost:8888/callback" <-- Make sure to add this to your Spotify Client!
```

* Note that ```config.yml``` has been added to ```.gitignore```, and thus won't be tracked (everyone has their own config.yml for your own Spotify connection, for now)

# Working with the Database [OBE]

## Start PostgreSQL Docker Container

To start the PostgreSQL database

```
cd docker
docker-compose up -d
```

## To Bring Down the Container

```
cd docker
docker-compose down
```
