import FNA.create_dataset as dt 
import FNA.generate_model as gm 
import os

os.getcwd()
os.listdir(os.getcwd())

def main():
    #Initiate Data Set Object 
    create_dataset = dt.CREATE_DATASET()

    #Obtain and Extract Text file or CSV file
    input_data = create_dataset.__read_csv__('input/wdbc.data.txt')

    #Print Dataset

    print(input_data)

    #Obtain Training and Testing DataSet
    input_x, features_train, features_test, label_train, label_test = create_dataset.__obtain_data__('input/wdbc.data.txt', number_features=30, number_labels=2)

    #Initiate Generate Model Object
    classifer = gm.GENERATE_MODEL() 

        #Model Generation and Metric Evaluation
    metrics1, _ = classifer.__evaluate_DNN__(input_x, features_train, features_test, label_train,  label_test, batch_size=100)

    print(metrics1)

if __name__ == "__main__":
    main()