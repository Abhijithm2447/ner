from django import forms
from retrain_fer.models import QueryDB, EntityDB

class QueryDBForm(forms.ModelForm):
    class Meta:
        model = QueryDB
        fields = '__all__'
class EntityDBForm(forms.ModelForm):
    class Meta:
        model = EntityDB
        fields = '__all__'
