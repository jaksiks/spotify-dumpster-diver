from django.shortcuts import render

# Create your views here.
def index(request):

    ## Call your data processing functions here
    ## This is what gets called when a user hits the website index/root directory "/"


    ## Then pass your processed data to the frontend via "context" below
    context = {
        ## Put data here that you want to pass to the frontend in key-value pair/dictionary form:
        ## 'key':variable,
    }
    
    return render(request, 'dumpster_diver/index.html', context)
