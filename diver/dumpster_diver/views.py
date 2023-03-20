from spotify.spotify_wrapper import SpotifyWrapper
from django.shortcuts import render

# Create your views here.
def index(request):

    print('Yo this worked')
    
    ## Call your data processing functions here
    ## This is what gets called when a user hits the website index/root directory "/"

    test_var = [1,2,3]
    ## Then pass your processed data to the frontend via "context" below
    context = {
        ## Put data here that you want to pass to the frontend in key-value pair/dictionary form:
        ## 'key':variable,
        'test_list':test_var
    }
    
    return render(request, 'dumpster_diver/index.html', context)
