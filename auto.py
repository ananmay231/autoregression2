import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

def interpret_r_squared(r_squared):
    if r_squared > 0.75:
        return "The model explains a large portion of the variance in the dependent variable."
    elif r_squared > 0.5:
        return "The model explains a moderate portion of the variance in the dependent variable."
    else:
        return "The model explains a small portion of the variance in the dependent variable."

def interpret_f_statistic(f_statistic, p_value):
    if p_value < 0.05:
        return f"The model is statistically significant with an F-statistic of {f_statistic:.2f}."
    else:
        return "The model is not statistically significant."


def interpret_durbin_watson(durbin_watson):
    if 1.5 < durbin_watson < 2.5:
        return "There is little to no autocorrelation in the residuals."
    elif durbin_watson <= 1.5:
        return "There is positive autocorrelation in the residuals, which may indicate a trend in the data."
    else:
        return "There is negative autocorrelation in the residuals, which may indicate a cyclic pattern in the data."

# Define your functions here
def convert_and_drop(df):
    for column in df.columns:
        try:
            df[column] = pd.to_numeric(df[column])
        except ValueError:
            df.drop(column, axis=1, inplace=True)
    return df

def remove_outliers_iqr(df, x_var, y_var):
    Q1_x = df[x_var].quantile(0.25)
    Q3_x = df[x_var].quantile(0.75)
    IQR_x = Q3_x - Q1_x
    
    Q1_y = df[y_var].quantile(0.25)
    Q3_y = df[y_var].quantile(0.75)
    IQR_y = Q3_y - Q1_y
    
    lower_bound_x = Q1_x - 1.5 * IQR_x
    upper_bound_x = Q3_x + 1.5 * IQR_x
    lower_bound_y = Q1_y - 1.5 * IQR_y
    upper_bound_y = Q3_y + 1.5 * IQR_y
    
    df_clean = df[(df[x_var] >= lower_bound_x) & (df[x_var] <= upper_bound_x) &
                  (df[y_var] >= lower_bound_y) & (df[y_var] <= upper_bound_y)]
    return df_clean

# Streamlit UI for page navigation
page = st.sidebar.selectbox("Choose a page", ["Home", "EDA","Regression"])

if page == "Home":
    st.title("Regression Analysis")
    st.write("Please choose a page from the sidebar to begin.")

elif page == "EDA":
    st.title('CSV File Analysis - EDA')
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if df is not None:
            st.write(df)
            df = df.dropna()
            df = convert_and_drop(df)
            columns = df.columns.tolist()
            x_var = st.selectbox('Select the X variable:', columns)
            y_var = st.selectbox('Select the Y variable:', columns)
            
            if st.button('Show Scatter Plot'):
                fig, ax = plt.subplots()
                ax.scatter(df[x_var], df[y_var])
                ax.set_xlabel(x_var)
                ax.set_ylabel(y_var)
                ax.set_title(f'Scatter Plot of {x_var} vs {y_var}')
                st.pyplot(fig)
            
            if st.button('Correlations'):
                corr = convert_and_drop(df).corr()
                corrs = sns.heatmap(corr)
                st.write(corr)

            if st.button('Show Pairplots'):
                pairplot_fig = sns.pairplot(df)
                st.pyplot(pairplot_fig.fig)
            
            if st.button('Remove Outliers and Show Scatter Plot'):
                df_clean = remove_outliers_iqr(df, x_var, y_var)
                fig, ax = plt.subplots()
                ax.scatter(df_clean[x_var], df_clean[y_var])
                ax.set_xlabel(x_var)
                ax.set_ylabel(y_var)
                ax.set_title(f'Cleaned Scatter Plot of {x_var} vs {y_var}')
                st.pyplot(fig)
elif page == "Regression":
    st.title('CSV File Analysis - Regression')
    
    uploaded_file = st.file_uploader("Choose a CSV file for regression analysis", type="csv", key="regression")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        if df is not None:
            df = df.dropna()
            df = convert_and_drop(df)
            if not df.empty:
                st.write(df)
                
                # Select variables for regression
                columns = df.columns.tolist()
                x_var = st.selectbox('Select the independent variable (X):', columns, key="X_var")

                y_var = st.selectbox('Select the dependent variable (Y):', columns, key="Y_var")
                
                if st.button('Perform Regression'):
                    X = sm.add_constant(df[x_var])  # Adding a constant for the intercept
                    Y = df[y_var]
                    model = sm.OLS(Y, X).fit()
                    predictions = model.predict(X)
                    
                    # Display regression results
                    st.write(model.summary())
                    
                    # Plotting regression line over scatter plot
                    plt.figure(dpi=300)
                    fig, ax = plt.subplots()
                    ax.scatter(X[x_var], Y, color="blue", label="Original data")
                    ax.plot(X[x_var], predictions, color="red", label="Fitted line")
                    ax.set_xlabel(x_var)
                    ax.set_ylabel(y_var)
                    ax.legend()
                    st.pyplot(fig)

                    fig = plt.figure(dpi=300)
                    res = model.resid
                    stats.probplot(res, dist="norm", plot=plt)
                    plt.title("Normal Q-Q plot")
                    st.pyplot(fig)

                    # Residual Plot
                    plt.figure(dpi=300)
                    fig, ax = plt.subplots()
                    sns.residplot(x=predictions, y=Y, lowess=True, line_kws={'color': 'red', 'lw': 1})
                    ax.set_title('Residuals vs Fitted')
                    ax.set_xlabel('Fitted values')
                    ax.set_ylabel('Residuals')
                    st.pyplot(fig)

                    # Assuming 'model' is your fitted OLS model
                    r_squared = model.rsquared
                    f_statistic = model.fvalue
                    f_pvalue = model.f_pvalue
                    durbin_watson = sm.stats.stattools.durbin_watson(model.resid)

                    # Generate interpretations
                    r2_interpretation = interpret_r_squared(r_squared)
                    f_stat_interpretation = interpret_f_statistic(f_statistic, f_pvalue)
                    dw_interpretation = interpret_durbin_watson(durbin_watson)

                    # Display in Streamlit
                    st.write(r2_interpretation)
                    st.write(f_stat_interpretation)
                    st.write(dw_interpretation)

            else:
                st.write("No numeric data available for regression.")


    
