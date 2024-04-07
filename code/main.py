from preprocess import load_data, data_split


def main():
    '''Main function to load, preprocess, and train the data'''
    init_data = load_data()
    init_data.describe().to_csv('../outputs/descriptive_stats.csv')
    X_train, X_test, y_train, y_test = data_split(init_data)
    #linear_regressors(X_train, X_test, y_train, y_test)
    #coef_comp=pd.DataFrame({'var':X_train.columns, 'val_ols':trained_ols.params.tolist()})
    



if __name__ == '__main__':
    main()