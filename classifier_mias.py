import Mammography.create_dataset as dt 
import Mammography.generate_model as gm 

def main():
    #Initiate Data Set Object
    create_dataset = dt.CREATE_DATASET()

    #Obtain Image Data Set 

    
    #Obtain Training and Testing Dataset


    #Initiate Generate Model Object 
    classifier = gm.GENERATE_MODEL()

    metrics1, _ = classifier.__generate_model_CNN__()
    metrics2, _ = classifier.__generate_model_RNN__()

    metrics = [metrics1, metrics2]

    print(metrics)

if __name__ == "__main__":
    main()