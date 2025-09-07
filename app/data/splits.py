def check_patient_leak(train_pids, val_pids, test_pids):
    return {
        "train∩val": len(set(train_pids)&set(val_pids)),
        "train∩test": len(set(train_pids)&set(test_pids)),
        "val∩test": len(set(val_pids)&set(test_pids)),
    }
