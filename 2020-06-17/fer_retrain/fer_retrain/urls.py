"""fer_retrain URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from retrain_fer import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('save_data2db/', views.FerSaveData2DB.as_view(), name='save_data2db'),
    path('train/', views.TrainData.as_view(), name='train'),
    path('hard_train/', views.HardTrainData.as_view(), name='hard_train'),
    path('test/', views.TestData.as_view(), name='test'),
    path('all_data/', views.AllDatasinDB.as_view(), name='all_data'),
    path('delete_data/', views.DeleteDatasinDB.as_view(), name='delete_data'),
    path('update_data/', views.UpdateDatasinDB.as_view(), name='update_data'),
]
