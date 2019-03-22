import Mammography.create_dataset as dt
import Mammography.generate_model as gm 

def main():
    #Initiate Data Set Object 
    create_dataset = dt.CREATE_DATASET()

    #Obtain TFREcords Data Set


    #Obtain Training and Testing DataSet


    #Initiate Generate Model Object
    classifer = gm.GENERATE_MODEL()

    #Model Generation and Metric Evaluation
    metrics1, _ = classifer.__generate_model_CNN__()
    metrics2, _ = classifer.__generate_model_RNN__()

    metrics = [metrics1, metrics2]
    
    print(metrics)

if __name__ == "__main__":
    main()
    