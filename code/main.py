from json import load
from mimetypes import init
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from preprocess import load_data, data_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np




def visualise(df, y):
    plt.figure(figsize=(13, 8))
    ax = plt.axes()
    plt.scatter(df['unemployment'], df[y])
    m, b = np.polyfit(df['unemployment'], df[y], 1)
    plt.plot(df['unemployment'], m * df['unemployment'] + b, color='red')
    ax.set_facecolor("whitesmoke")
    plt.ylim(top=1.2)
    plt.ylabel(y, fontweight="bold")
    plt.xlabel("Unemployment", fontweight="bold")
    plt.title('Unemployment vs ' + y, fontweight="bold")
    plt.savefig('../outputs/trends_' + y +  '.jpeg', dpi=300)
    plt.show()


def linear_regressors(df):
    '''
    Evaluating initial coefficients assigned using OLS, Lasso, and Ridge
    '''
    outcomes = ['anger', 'love', 'sadness', 'emotions_id']
    for y in outcomes:
        X_train, X_test, y_train, y_test, cols = data_split(df,y)
        sm.add_constant(X_train)
        sm.add_constant(X_test)
        #print(X_train.head())
        trained_ols = sm.OLS(y_train, X_train).fit()
        ols_pred = trained_ols.predict(X_test)
        ols_mse = mean_squared_error(y_test, ols_pred)
    
        plt.figure(figsize=(13, 8))
        ax = plt.axes()
        plt.scatter(trained_ols.params.tolist(), X_train.columns)
        plt.errorbar(trained_ols.params.tolist(), X_train.columns, xerr=ols_mse, fmt="o", ecolor="red")
        plt.grid(True)
        ax.set_facecolor("whitesmoke")
        plt.ylabel('Coefficient', fontweight="bold")
        plt.title('Coefficient Plot - ' + y, fontweight="bold")
        plt.savefig('../outputs/coefficients_' + y +  '.jpeg', dpi=300)
        plt.show()
        visualise(df, y)
    
        print(trained_ols.summary())


def multilayer(df):
    '''Training the multilayer neural network'''
    #'anger', 'love', 'sadness', 
    outcomes = ['anger', 'love', 'sadness', 'emotions_id']
    for y in outcomes:
        X_train, X_test, y_train, y_test, cols = data_split(df,y)
        #print(X_train)
        scaler=StandardScaler() 
        scaler.fit(X_train) 
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        #Initializing a multilayer perceptron classifer with defined hidden layers, optimizer, and learning rate
        if y == 'emotions_id':
            MLP = MLPClassifier(
            random_state=1680,
                                activation='logistic', solver='adam', 
                                max_iter =500,
                                 learning_rate_init=0.01)
            MLP.fit(X_train,y_train) 
            print(accuracy_score(y_train,MLP.predict(X_train)))
            print("mlp test accuracy:")
            print(accuracy_score(y_test, MLP.predict(X_test)))
            cm = confusion_matrix(y_test, MLP.predict(X_test))
            
            # Plotting confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                        xticklabels=MLP.classes_, yticklabels=MLP.classes_)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix for {y}')
            plt.savefig('../outputs/emotion_cm.jpeg', dpi=300)
            plt.show()

            
            unemployment_index = cols.get_loc('unemployment')
            print(unemployment_index)
            # Plot Partial Dependence Plot for 'unemployment'
            pd_display = PartialDependenceDisplay.from_estimator(MLP, X_test, features=[unemployment_index])
            pd_display.plot()
            plt.title('Partial Dependence Plot for Unemployment - ' + y)
            plt.xlabel('Unemployment')
            plt.ylabel('Average Prediction of negative emotions')
            plt.tight_layout()
            plt.savefig('../outputs/emotion_id_mlp.jpeg', dpi=300)
            plt.show()
            
        
            

        else:
            MLP = MLPRegressor(
            random_state=1680,
                                activation='logistic', solver='adam', 
                                max_iter =500,
                                learning_rate_init=0.01)
            MLP.fit(X_train,y_train)
            #print(MLP.predict(X_test).shape)
            print(MLP.score(X_train,y_train))
            #print("mlp test accuracy:")

            print(MLP.score(X_test, y_test))

            unemployment_index = cols.get_loc('unemployment')
            print(unemployment_index)
            # Plot Partial Dependence Plot for 'unemployment'
            pd_display = PartialDependenceDisplay.from_estimator(MLP, X_test, features=[unemployment_index], kind='both')
            pd_display.plot()
            plt.title('Partial Dependence Plot for Unemployment - ' + y)
            plt.xlabel('Unemployment')
            plt.ylabel('Average Prediction of ' + y)
            plt.tight_layout()
            plt.savefig('../outputs/' + y + '_mlp.jpeg', dpi=300)
            plt.show()



       
def visualize_trend(init_data):
    df = init_data[init_data['emotions_id'] <= 1]
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    df['emotional'] = df['emotions_id'].apply(lambda x: 'Negative Emotion' if x == 1 else 'Positive Emotion')
    # Plot the data
    plt.figure(figsize=(10, 6))

    # Use Seaborn's lineplot to plot unemployment rate over time for each category of y
    sns.lineplot(x='date', y='unemployment', hue='emotional', data=df, palette='colorblind')
    plt.title('Unemployment Rate Over Time for Negative and Positive Emotions')

    plt.xlabel('Date')
    plt.ylabel('Unemployment Rate')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../outputs/unemployment_trends.jpeg')
    plt.show()




def main():
    '''Main function to load, preprocess, and train the data'''
    init_data = load_data()
    #init_data = init_data[(init_data['emotions_id'] <=1)]
    #print(len(init_data))
    #print(init_data.describe())
    init_data.describe().to_csv('../outputs/descriptive_stats.csv')
    #linear_regressors(init_data)

    #multilayer(init_data)

    #visualize_trend(init_data)
    



if __name__ == '__main__':
    main()

#lassoreg = make_pipeline(StandardScaler(with_mean=False), Lasso())

    #alphas=np.linspace(1e-6, 1, num=50)
    #params = {'lasso__alpha':alphas}
    #gslasso = GridSearchCV(lassoreg, params, n_jobs=-1, cv=10)
    #gslasso.fit(X_train, y_train)
    #lasso_alpha = list(gslasso.best_params_.values())[0]

    #ridgereg = make_pipeline(StandardScaler(with_mean=False), Ridge())
    #alphas=np.linspace(1e-6, 1, num=50)
    #ridgeparams = {'ridge__alpha':alphas * X_train.shape[0]}
    #gsridge = GridSearchCV(ridgereg, ridgeparams, n_jobs=-1, cv=10)
    #gsridge.fit(X_train, y_train)
    #ridge_alpha = list(gsridge.best_params_.values())[0] / X_train.shape[0]

    #lassoReg = make_pipeline(StandardScaler(with_mean=False), Lasso(alpha=lasso_alpha))
    #lassoReg.fit(X_train, y_train)
    #lasso_pred = lassoReg.predict(X_test)
    #lasso_mse = mean_squared_error(y_test, lasso_pred)

    #ridgeReg = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=ridge_alpha * X_train.shape[0]))
    #ridgeReg.fit(X_train, y_train)
    #ridge_pred = ridgeReg.predict(X_test)
    #ridge_mse = mean_squared_error(y_test, ridge_pred)
    #coef_comp=pd.DataFrame({'var':X_train.columns, 'val_ols':trained_ols.params.tolist(), 'val_lasso':lassoReg['lasso'].coef_, 'var_ridge':ridgeReg['ridge'].coef_})


    #mse = min(ols_mse, lasso_mse, ridge_mse)
    #if mse == ols_mse:
    #    coef = 'val_ols'
    #elif mse == lasso_mse:
    #    coef = 'val_lasso'
    #elif mse == ridge_mse:
    #    coef = 'val_ridge'
    #print(coef)