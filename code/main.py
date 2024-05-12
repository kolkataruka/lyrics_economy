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
from statsmodels.iolib.summary2 import summary_col

'''
File to be run for analysis. Only run after all data is generated. 
'''



def visualise(df, y):
    '''
    Visualizes the trend between an outcome variable and unemployment
    '''
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
    Evaluating initial coefficients for different dependent variables
    using OLS
    '''
    outcomes = ['anger', 'love', 'sadness', 'emotions_id']
    trained_models = []
    for y in outcomes:
        X_train, X_test, y_train, y_test, cols = data_split(df,y, True)
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
        trained_models.append(trained_ols)
        print(trained_ols.summary())
    summary = summary_col(trained_models, stars=True, float_format='%0.2f', model_names=outcomes, regressor_order=['unemployment', 'acoustic', 'dance', 'energy', 'instrumental', 'loudness', 'mode', 'tempo', 'valence', 'explicit'], drop_omitted=True).as_latex()
    with open('../outputs/ols_table.tex', 'w') as file:
        file.write(summary)


def multilayer(df):
    '''Training the multilayer neural network'''
    #'anger', 'love', 'sadness', 
    outcomes = ['anger', 'love', 'sadness', 'emotions_id']
    for y in outcomes:
        X_train, X_test, y_train, y_test, cols = data_split(df,y, False)
        #print(X_train)
        scaler=StandardScaler() 
        scaler.fit(X_train) 
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        #Initializing a multilayer perceptron classifer with defined hidden layers, optimizer, and learning rate
        if y == 'emotions_id':
            #Training an MLP Classifier only for emotions_id
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
            # Partial Dependence Plot for 'unemployment'
            pd_display = PartialDependenceDisplay.from_estimator(MLP, X_test, features=[unemployment_index])
            pd_display.plot()
            plt.title('Partial Dependence Plot for Unemployment - ' + y)
            plt.xlabel('Unemployment')
            plt.ylabel('Average Prediction of negative emotions')
            plt.tight_layout()
            plt.savefig('../outputs/emotion_id_mlp.jpeg', dpi=300)
            plt.show()

        else:
            #Training an MLP Regressor for anger, love, and sadness
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
            #  Partial Dependence Plot for 'unemployment'
            pd_display = PartialDependenceDisplay.from_estimator(MLP, X_test, features=[unemployment_index], kind='both')
            pd_display.plot()
            plt.title('Partial Dependence Plot for Unemployment - ' + y)
            plt.xlabel('Unemployment')
            plt.ylabel('Average Prediction of ' + y)
            plt.tight_layout()
            plt.savefig('../outputs/' + y + '_mlp.jpeg', dpi=300)
            plt.show()



       
def visualize_trend(init_data):
    '''Visualizes the trend between unemployment and emotions_id'''
    df = init_data[init_data['emotions_id'] <= 1]
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    df['Emotion Classification'] = df['emotions_id'].apply(lambda x: 'Negative Emotion' if x == 1 else 'Positive Emotion')
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

    init_data[['acoustic','dance','energy','instrumental','loudness','mode']].describe().to_latex('../outputs/descriptive_stats1.tex')
    init_data[['tempo', 'valence','emotions_id','anger','love','sadness','unemployment']].describe().to_latex('../outputs/descriptive_stats2.tex')
    linear_regressors(init_data)

    multilayer(init_data)

    visualize_trend(init_data)
    



if __name__ == '__main__':
    main()

