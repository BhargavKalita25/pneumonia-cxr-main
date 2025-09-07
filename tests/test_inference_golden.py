from app.utils.metrics import compute_all_metrics

def test_metric_math():
    y_true=[0,0,1,1]; y_prob=[0.1,0.4,0.6,0.9]
    m=compute_all_metrics(y_true,y_prob)
    assert 0<=m["roc_auc"]<=1
    assert m["confusion_matrix"]==[[2,0],[0,2]]
