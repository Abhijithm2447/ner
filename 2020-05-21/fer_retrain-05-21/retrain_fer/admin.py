from django.contrib import admin
from retrain_fer.models import QueryDB, EntityDB
# Register your models here.

admin.site.register(QueryDB)
admin.site.register(EntityDB)

