def generate_results(results:dict, train_times:dict)->None:
    with open("./results/results.txt",'w') as file:
        file.write("THE RESULTS\n")
        
        for model in results:
            file.write("The accuracy of " + model + " model: " + str(results[model]) + " | train_time: " + str(train_times[model]) + "\n")


        