from django.db import models

class QueryDB(models.Model):    
    query = models.TextField( primary_key=True)    
class EntityDB(models.Model):
    id = models.AutoField(primary_key=True)
    # query = models.CharField(max_length=200, unique=True) 
    query = models.ForeignKey(QueryDB, on_delete=models.CASCADE)
    entity_name = models.CharField(max_length=64)
    start_pos = models.IntegerField()
    end_pos = models.IntegerField()
    is_trained = models.BooleanField(default=False)

    class Meta:
        # db_table = 'EntityDB'
        constraints = [
            models.UniqueConstraint(fields=['query', 'start_pos', 'end_pos'], name='unique appversion')
        ]

