import pandas

# STUDENT_PERFORMANCE DATASET
StudentPerformanceDF = pandas.read_csv('./data/raw/Student_Performance.csv')
extra_act_dummies = pandas.get_dummies(StudentPerformanceDF['Extracurricular Activities'], prefix='ExtraActivities')
StudentPerformanceDF = pandas.concat([extra_act_dummies, StudentPerformanceDF], axis=1)
StudentPerformanceDF.drop(columns=['Extracurricular Activities'], inplace=True)

StudentPerformanceDF.to_csv("./data/processed/Student_Performance_processed.csv", index=False)



