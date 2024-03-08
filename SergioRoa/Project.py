import pandas as plane
import numpy as np
from time import process_time
def data():
#REMOVED SATISFACTION(Satisfied = 1 | Disatsified = 0)
#REMOVED TYPE OF TRAVEL(Business = 0 | Personal = 1)
#REMOVED CLASS(Eco = 0 | Eco Plus = 1 | Business = 2)
    start = process_time()
    names = ['Age','Seat comfort','Type of Travel','Class','Baggage handling','Online boarding','Inflight entertainment','Food and drink','Checkin service','Departure Delay in Minutes', 'Arrival Delay in Minutes','Ease of Online booking','satisfaction']
#Imports csv to program(the portion that states r'FILE PATH' must be changed)
    df = plane.read_csv(r'C:\Users\Sergio\Desktop\SergioRoa\train.csv', usecols=names)
#Sets X to be the attributes of the table and y to be the target class satisfaction
    X=df.drop(['satisfaction'],axis=1)
    y=df['satisfaction']

    plane.set_option('display.max_columns',13)

#Splits the data into Train(making the model) and Test(testing the model)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
#Data Preprocessing 
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
#Sets the N-neighbors that will be checked
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 10)
    
    classifier.fit(X_train, y_train)
    
#The test data set is used to make predicitons to test against the Model
    y_pred = classifier.predict(X_test)
#The metrics that will be measured
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    end = process_time()
#Print
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report")
    print(classification_report(y_test, y_pred))
    print("Accuracy")
    print(accuracy_score(y_test,y_pred))
    print("Run-Time")
    print(end - start)
#Keeps the program running until the user decides to exit
    while True:
        userAge = input("Age: ")
        userSeatComfort = input("Seat comfort(0-5): ")
        userTypeofTravel = input("Reason for travel(0 = Business| 1 = Personal): ")
        userClass = input("Class(0=Eco|1=EcoPlus|2=Business): ")
        userBaggage = input("Baggage handling(0-5): ")
        userOnlineBoarding = input("Online boarding(0-5): ")
        userInflightEntertainment = input("Inflight entertainment(0-5): ")
        userFoodandDrinks = input("Food and drink(0-5): ")
        userCheckin = input("Checkin service(0-5): ")
        userDeparture = input("Departure Delay in Minutes(in minutes): ")
        userArrival = input("Arrival Delay in Minutes(in minutes): ")
        userBooking = input("Ease of Online booking(0-5): ")

        userArray = np.array([[userAge,userTypeofTravel,userClass,userBooking,userFoodandDrinks,userOnlineBoarding,userSeatComfort,userInflightEntertainment,userBaggage,userCheckin,userDeparture,userArrival]],dtype=np.float64)
        #userArray = scaler.transform(userArray)        
        userPredict = classifier.predict(userArray)
        if userPredict==1:
            print("You are likely to be satisfied")
        elif userPredict == 0:
            print("You are likely to be dissatisfied or neutral")

        userExit = input("Type 'exit' to close the program or 'retry' to run another simulation: ")
        if userExit == 'exit':
            break
        elif userExit == 'retry':
            data()
data()


