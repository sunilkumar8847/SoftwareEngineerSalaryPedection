import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]
le_devtype = data["le_devtype"]

def show_predic_page():
    st.title("Software Engineer Salary Prediction")
    st.write("""### We need some information to predict the salary""")

    countries = (
        "United States of America",
        "Germany",
        "United Kingdom of Great Britain and Northern Ireland",  
        "Canada",
        "India",                                                    
        "France",                                                   
        "Netherlands",                                              
        "Australia",                                                
        "Brazil",                                                   
        "Spain",                                                    
        "Poland",                                                   
        "Sweden",                                                    
        "Italy",                                                     
        "Switzerland",                                               
        "Denmark",                                                   
        "Norway",                                                    
        "Austria",                                                   
        "Israel",                                                    
        "Portugal",                                                  
        "Czech Republic",                                           
        "Belgium"
        "Finland",
    )
    
    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )
    
    job_role = (
        "Developer, full-stack",
        "Developer, back-end",
        "Developer, front-end",
        "Developer, desktop or enterprise applications",
        "Developer, mobile",
        "Engineering manager",
        "Developer, embedded applications or devices",
        "Data scientist or machine learning specialist",
        "DevOps specialist",
        "Engineer, data",
        "Research & Development role",
        "Cloud infrastructure engineer",
        "Senior Executive (C-Suite, VP, etc.)",
        "Academic researcher",
        "Developer, game or graphics",
        "Data or business analyst",
        "Developer, QA or test",
        "Engineer, site reliability",
        "System administrator",
        "Project manager",
        "Security professional",
        "Developer Experience",
        "Product manager",
        "Scientist",
        "Blockchain",
        "Hardware Engineer",
        "Database administrator",
        "Educator",
        "Developer Advocate",
        "Designer",
        "Marketing or sales professional"
    )
    
    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)
    jobRole = st.selectbox("Job Role", job_role)
    experience = st.slider("Years of experience", 0, 50, 3)
    
    ok = st.button("Calculates Salary")
    if ok:
        X = np.array([[country, education, experience, jobRole]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X[:, 3] = le_devtype.transform(X[:, 3])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")
        
        
# show_predic_page()