from django.shortcuts import render

# Create your views here.
def index(request):
    context = {'a': 1}
    return render(request, 'classifier/index.html', context)