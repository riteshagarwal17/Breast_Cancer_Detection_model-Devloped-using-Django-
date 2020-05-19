from django.shortcuts import render
from django.http import HttpResponse
from sklearn.neighbors import KNeighborsClassifier
from django.conf import settings
from cancerapp.forms import *
from matplotlib import pyplot,pylab
from pylab import *
import PIL, PIL.Image
from io import StringIO, BytesIO
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve,accuracy_score
from sklearn import ensemble, linear_model, neighbors, svm, tree, neural_network
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures,StandardScaler,MinMaxScaler,Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm,model_selection, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.metrics import classification_report,confusion_matrix,log_loss,f1_score
import matplotlib.pyplot as plt
#from math import sqrt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# import api.forms


def index(request):
    temp = {}
    context = {'temp': temp}
    return render(request, 'index.html', context)


def getimage(request):
    # Construct the graph
    x = arange(0, 2 * pi, 0.01)
    s = cos(x) ** 2
    plot(x, s)

    xlabel('xlabel(X)')
    ylabel('ylabel(Y)')
    title('Simple Graph!')
    grid(True)

    # Store image in a string buffer
    buffer = BytesIO()
    canvas = pylab.get_current_fig_manager().canvas
    canvas.draw()
    pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    pilImage.save(buffer, "PNG")
    pylab.close()

    # Send buffer in a http response the the browser with the mime type image/png set
    return HttpResponse(buffer.getvalue(), content_type="image/png")

def getimageuniform(request):
    image = Image.open("media/Uniform_diistribution.png")
    image.show()
    value = 0
    context = {'model' : value}
    #context = {'temp': temp}
    return render(request, 'get_img.html', context)


def getimagecorrelation(request):
    image = Image.open("media/Correlation.png")
    buffer = BytesIO()
    image.save(buffer, 'png')
    canvas = pylab.get_current_fig_manager().canvas
    canvas.draw()
    pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    pilImage.save(buffer, "PNG")
    pylab.close()
    return HttpResponse(buffer.getvalue(), content_type="image/png")

def getimageimportance(request):
    nam = ['id', 'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape', 'marginal_adhesion',
           'single_epithelial_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
    data = pd.read_csv("breast_cancer.csv", names=nam)
    data.drop(data[data['bare_nuclei'] == '?'].index, inplace=True)
    data = data.drop('id', axis=1)
    data.drop_duplicates(keep='first', inplace=True)
    y = data["class"]
    x = data.drop('class', axis=1)
    model = ensemble.ExtraTreesClassifier()
    model.fit(x, y)
    feat_importances = pd.Series(model.feature_importances_, index=x.columns)
    feat_importances.nlargest(9).plot(kind='barh')
    pyplot.subplots_adjust(left= 0.25)
    buffer = BytesIO()
    canvas = pylab.get_current_fig_manager().canvas
    canvas.draw()
    pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    pilImage.save(buffer, "PNG")
    pylab.close()
    return HttpResponse(buffer.getvalue(), content_type="image/png")

def getimagebalance(request):
    base_image = Image.open("media\\Balancing_Data.png")
    #print(base_image.format)
    #content = {"base_image": base_image}
    #return render(request, 'get_img.html', content)
    base_image.show()
    #temp = {}
    value=1
    context = {'model': value}
    return render(request, 'get_img.html', context)

def prediction(request):
    print(request)
    print("hello")
    if request.method == 'POST':
        #url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancerapp-wisconsin/breast-cancerapp-wisconsin.data"
        nam = ['id', 'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape','marginal_adhesion', 'single_epithelial_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli','mitoses', 'class']
        data=pd.read_csv("breast_cancer.csv",names=nam)
        #data = pd.read_csv(url, names=nam)
        #print(data.head())
        data.drop(data[data['bare_nuclei'] == '?'].index, inplace=True)
        data = data.drop('id', axis=1)
        data.drop_duplicates(keep='first', inplace=True)
        y = data["class"]
        x = data.drop('class', axis=1)
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=20)
        #scaler = StandardScaler()
        #xtrain = scaler.fit_transform(xtrain)
        #xtest = scaler.transform(xtest)
        from sklearn.linear_model import LogisticRegression
        #RFC = LogisticRegression()
        RFC=RandomForestClassifier(n_estimators=80, max_features="auto", random_state=12)
        #print(xtrain)
        #print(ytrain)
        RFC.fit(xtrain, ytrain)  # traing to machine
        frm = KnnForm(request.POST)

        p1 = request.POST.get('p1')
        p2 = request.POST.get('p2')
        p3 = request.POST.get('p3')
        p4 = request.POST.get('p4')
        p5 = request.POST.get('p5')
        p6 = request.POST.get('p6')
        p7 = request.POST.get('p7')
        p8 = request.POST.get('p8')
        p9 = request.POST.get('p9')
        dataset = [[p1, p2, p3, p4, p5, p6, p7, p8, p9]]
        testdata = pd.DataFrame(dataset)
        scoreval = RFC.predict(testdata)[0]
        print(scoreval)
        if scoreval == 2:
            scoreval = "Benign"
        else:
            scoreval = "Malignant"
        # context = {'scoreval': scoreval, 'temp': temp}
        context = {'scoreval': scoreval, 'dataset': dataset}
        return render(request, 'index.html', context)
    else:
        frm = KnnForm()
        print("hello in views else")
        return render(request, 'index.html', {'scoreval': frm})

def getaccuracy(request):
    image = Image.open("media/Accuracy_plot.png")
    buffer = BytesIO()
    image.save(buffer, 'png')
    canvas = pylab.get_current_fig_manager().canvas
    canvas.draw()
    pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    pilImage.save(buffer, "PNG")
    pylab.close()
    return HttpResponse(buffer.getvalue(), content_type="image/png")

def getalgorithm_comparison(request):
    image = Image.open("media/Algorithm_comparison.png")
    buffer = BytesIO()
    image.save(buffer, 'png')
    canvas = pylab.get_current_fig_manager().canvas
    canvas.draw()
    pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    pilImage.save(buffer, "PNG")
    pylab.close()
    return HttpResponse(buffer.getvalue(), content_type="image/png")

def getAUC_Comparison(request):
    image = Image.open("media/AUC_Comparison_plot.png")
    buffer = BytesIO()
    image.save(buffer, 'png')
    canvas = pylab.get_current_fig_manager().canvas
    canvas.draw()
    pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    pilImage.save(buffer, "PNG")
    pylab.close()
    return HttpResponse(buffer.getvalue(), content_type="image/png")

def getmisclassification(request):
    image = Image.open("media/misclassification_plot.png")
    buffer = BytesIO()
    image.save(buffer, 'png')
    canvas = pylab.get_current_fig_manager().canvas
    canvas.draw()
    pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    pilImage.save(buffer, "PNG")
    pylab.close()
    return HttpResponse(buffer.getvalue(), content_type="image/png")

def getf1_score(request):
    image = Image.open("media/f1_score_plot.png")
    buffer = BytesIO()
    image.save(buffer, 'png')
    canvas = pylab.get_current_fig_manager().canvas
    canvas.draw()
    pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    pilImage.save(buffer, "PNG")
    pylab.close()
    return HttpResponse(buffer.getvalue(), content_type="image/png")

def getROC_curve(request):
    image = Image.open("media/ROC_curve_plot.png")
    buffer = BytesIO()
    image.save(buffer, 'png')
    canvas = pylab.get_current_fig_manager().canvas
    canvas.draw()
    pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    pilImage.save(buffer, "PNG")
    pylab.close()
    return HttpResponse(buffer.getvalue(), content_type="image/png")

def getsensitivity(request):
    image = Image.open("media/sensitivity_comparison_plot.png")
    buffer = BytesIO()
    image.save(buffer, 'png')
    canvas = pylab.get_current_fig_manager().canvas
    canvas.draw()
    pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    pilImage.save(buffer, "PNG")
    pylab.close()
    return HttpResponse(buffer.getvalue(), content_type="image/png")

def getspecificity(request):
    image = Image.open("media/specificity_plot.png")
    buffer = BytesIO()
    image.save(buffer, 'png')
    canvas = pylab.get_current_fig_manager().canvas
    canvas.draw()
    pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    pilImage.save(buffer, "PNG")
    pylab.close()
    return HttpResponse(buffer.getvalue(), content_type="image/png")

def gettest_accuracy(request):
    image = Image.open("media/test_accuracy_comparison.png")
    buffer = BytesIO()
    image.save(buffer, 'png')
    canvas = pylab.get_current_fig_manager().canvas
    canvas.draw()
    pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    pilImage.save(buffer, "PNG")
    pylab.close()
    return HttpResponse(buffer.getvalue(), content_type="image/png")

def gettrain_accuracy(request):
    image = Image.open("media/train_accuracy_plot.png")
    buffer = BytesIO()
    image.save(buffer, 'png')
    canvas = pylab.get_current_fig_manager().canvas
    canvas.draw()
    pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    pilImage.save(buffer, "PNG")
    pylab.close()
    return HttpResponse(buffer.getvalue(), content_type="image/png")

