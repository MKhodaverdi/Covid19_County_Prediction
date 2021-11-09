# Covid19_County_Prediction

# Predicting increases in COVID-19 incidence to identify locations for targeted testing in West Virginia: A machine learning enhanced approach 
[doi.org/10.1371/journal.pone.0259538](https://doi.org/10.1371/journal.pone.0259538)

## Authors
[Brad Price](https://business.wvu.edu/faculty-and-staff/directory/profile?pid=273); [Maryam Khodaverdi](https://directory.hsc.wvu.edu/Profile/61365); [Adam Halasz](https://mathematics.wvu.edu/faculty-and-staff/faculty-directory/adam-halasz); [Brian Hendricks](https://directory.hsc.wvu.edu/Profile/52462); [Wesley Kimble](https://directory.hsc.wvu.edu/Profile/39623); [Gordon S. Smith](https://directory.hsc.wvu.edu/Profile/46172); [Sally L. Hodder](https://directory.hsc.wvu.edu/Profile/41751);

## Abstract
During the COVID-19 pandemic, West Virginia developed an aggressive SARS-CoV-2 testing strategy which included utilizing pop-up mobile testing in locations anticipated to have near-term increases in SARS-CoV-2 infections. This study describes and compares two methods for predicting near-term SARS-CoV-2 incidence in West Virginia counties. The first method, Rt Only, is solely based on producing forecasts for each county using the daily instantaneous reproductive numbers, Rt.  The second method, ML+Rt, is a machine learning approach that uses a Long Short-Term Memory network to predict the near-term number of cases for each county using epidemiological statistics such as Rt, county population information, and time series trends including information on major holidays, as well as leveraging statewide COVID-19 trends across counties and county population size. 

Both approaches used daily county-level SARS-CoV-2 incidence data provided by the West Virginia Department Health and Human Resources beginning April 2020. The methods are compared on the accuracy of near-term SARS-CoV-2 increases predictions by county over 17 weeks from January 1, 2021- April 30, 2021. Both methods performed well (correlation between forecasted number of cases and the actual number of cases week over week is 0.872 for the ML+Rt method and 0.867 for the Rt Only method) but differ in performance at various time points. Over the 17-week assessment period, the ML+Rt method outperforms the Rt Only method in identifying larger spikes. Results show that both methods perform adequately in both rural and non-rural predictions. Finally, a detailed discussion on practical issues regarding implementing forecasting models for public health action based on Rt is provided, and the potential for further development of machine learning methods that are enhanced by Rt.  
 

## Repository Usage

This repository is broken down into: 

[Code_workbook](https://github.com/MKhodaverdi/Covid19_County_Prediction/tree/main/Code_Workbook)

[Data_workbook](https://github.com/MKhodaverdi/Covid19_County_Prediction/tree/main/Data_Workbook)


## License
[MIT](https://choosealicense.com/licenses/mit/)
