# Web
import streamlit as st 

# EDA
import pandas as pd 


# Data Viz 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns 

# ML Packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

class Web:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        st.title("Example Iris App")
        st.header("Example Web App with Streamlit ")
        st.markdown("""
            #### Description
            + This is a example Exploratory Data Analysis and Implementation Machine learning of the Iris Dataset 
              depicting the various species built with Streamlit.

            #### Purpose
            + To show a example EDA and ML of Iris using Streamlit framework. 
    	""")
        
    def ml (self, data) -> None:
        st.header("Exploratory Data Analysis")
        if data is not None:
            df = pd.read_csv(data)
            st.write(df.head())
            
            if st.checkbox("Show Shape"):
                st.write(df.shape)
            
            if st.checkbox("Show Columns"):
                all_columns = df.columns.to_list()
                st.write(all_columns)

            if st.checkbox("Show Null Data"):
                st.write(df.isnull().sum())
            
            if st.checkbox("Show Duplicate Data"):
                st.dataframe(df[df.duplicated()])
                
            if st.checkbox("Description Data"):
                st.write(df.describe())
            
            if st.checkbox("Show Value Counts"):
                st.write(df.iloc[:,-1].value_counts())
            
            if st.checkbox("Class Counts Bar Plot"):
                st.set_option('deprecation.showPyplotGlobalUse', False)
                plt.title('Class Count Plot')
                st.write(sns.countplot(x=df['species']))
                st.pyplot()
            
            if st.checkbox("Distribution Bar Plot"):
                st.set_option('deprecation.showPyplotGlobalUse', False)
                all_columns = df.columns.to_list()
                column_to_plot1 = st.selectbox("Select X Column",all_columns)
                column_to_plot2 = st.selectbox("Select Y Column",all_columns)
                plt.title('Distribution Species')
                st.write(sns.boxplot( y=column_to_plot2, x= column_to_plot1, data=df, orient='v'))
                st.pyplot()
            
            if st.checkbox("Compare Pair Plot"):
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.write(sns.pairplot(df,hue="species",height=4))
                st.pyplot()

            if st.checkbox("Correlation Plot(Seaborn)"):
                st.set_option('deprecation.showPyplotGlobalUse', False)
                fig, ax = plt.subplots(figsize=(10,10))
                st.write(sns.heatmap(df.corr(),annot=True, ax=ax))
                st.pyplot()

            st.header("Building ML Models")
            X = df.iloc[:,0:-1] 
            Y = df.iloc[:,-1]
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
            
            models = []
            models.append(('LR', Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression())])))
            models.append(('LDA', Pipeline([('scaler', StandardScaler()), ('lr', LinearDiscriminantAnalysis())])))
            models.append(('KNN', KNeighborsClassifier()))
            models.append(('CART', DecisionTreeClassifier()))
            models.append(('NB', GaussianNB()))
            models.append(('SVM', SVC()))
            
            model_names = []
            model_mean = []
            model_std = []
            all_models = []
            scoring = 'accuracy'
            for name, model in models:
                kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
                cv_results = model_selection.cross_val_score(model, X_test, y_test, cv=kfold, scoring=scoring)
                model_names.append(name)
                model_mean.append(cv_results.mean())
                model_std.append(cv_results.std())
                accuracy_results = {"model name":name,"model_accuracy":cv_results.mean(),"standard deviation":cv_results.std()}
                all_models.append(accuracy_results)
                
            if st.checkbox("Metrics As Table"):
                st.dataframe(pd.DataFrame(zip(model_names,model_mean,model_std),columns=["Algo","Mean of Accuracy","Std"]))
            
            if st.checkbox("Metrics As JSON"):
                st.json(all_models)

            clf = Pipeline([('scaler', StandardScaler()), ('lda', LinearDiscriminantAnalysis())])
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            st.subheader('Confusion Matrix')
            plot_confusion_matrix(clf, X_test, y_test)
            st.pyplot()

            st.subheader('Classification Report')
            report = classification_report(y_test, y_pred)
            st.text(report)

            st.subheader("Check Model")
            # urutan slider (start, end, default)
            sepal_length = st.slider('Sepal length', 4.3, 7.9, 5.4)
            sepal_width = st.slider('Sepal width', 2.0, 4.4, 3.4)
            petal_length = st.slider('Petal length', 1.0, 6.9, 1.3)
            petal_width = st.slider('Petal width', 0.1, 2.5, 0.2)
            dat = {'sepal_length': sepal_length,
                    'sepal_width': sepal_width,
                    'petal_length': petal_length,
                    'petal_width': petal_width}
            features = pd.DataFrame(dat, index=[0])
            d = features
            st.write(d)
            
            prediction = clf.predict(d)
            prediction_proba = clf.predict_proba(d)

            st.subheader('Result')
            st.write(prediction)

            st.subheader('Prediction Probability')
            st.write(prediction_proba)
    

    
    def main (self) -> None:
        dat = st.file_uploader("Upload Data", type=["csv", "txt"])
        self.ml(dat)

if __name__ == '__main__':
    app = Web()
    app.main()

