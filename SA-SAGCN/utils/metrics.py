import numpy as np

def masked_rmse_test(y_true, y_pred, null_val=np.nan):
    y_true = y_true.detach().numpy()
    y_pred = y_pred.detach().numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            # null_val=null_val
            mask = np.not_equal(y_true, null_val)
        mask = np.array(mask).astype('float32')
        mask /= np.mean(mask)
        mse = ((y_pred- y_true)**2)
        mse = np.nan_to_num(mask * mse)
        return np.sqrt(np.mean(mse))

def masked_mae_test(y_true, y_pred, null_val=np.nan):
    y_true = y_true.detach().numpy()
    y_pred = y_pred.detach().numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            # null_val=null_val
            mask = np.not_equal(y_true, null_val)
        mask = np.array(mask).astype('float32')
        mask /= np.mean(mask)
        mse = abs(y_pred- y_true)
        mse = np.nan_to_num(mask * mse)
        return np.sqrt(np.mean(mse))
