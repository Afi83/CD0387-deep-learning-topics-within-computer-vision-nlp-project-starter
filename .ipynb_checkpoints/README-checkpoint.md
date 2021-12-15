# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Run through the train_and_deploy.ipynb notebook to download and upload the dogImages to S3 bucket 
for model training and evaluations. Running code cells sequentially should first set up hyperparameters 
and run that for best model performance in hyperparamters space selected. Following that uses the selected hyperparameters and try to use those for more profiling and debugging features, and finally try to deploy the final model as an inference API endpoint. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
It uses dogImages dataset provided which contains 3 folders of train/val/test of 133 dog breeds which is used for multiclass prediction of dog breeds. 

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
Resnet18 pretrained model was used to fine tune my dog breed prediction model. This model was chosen because it was relatively newer but still not too big that might have increased the trainning needed for good results. 3 different hyperparamters were chosen to do the model tuning based on dog breeds data. 
learning rate was used as a ContinousParamter in range of 0.001 to 0.1 because this parameter has great importance on if the model succeed in traning or fails due to non-convergence. Small range makes the model run slow but usually could be better peroforming. Number of epochs was used as IntegerParameter to see if longer training improves the performance. The last used parametr was batch-size of different Categories. This is another important parameter to tune because selection of this as too large value might results in poor generalization even though makes the model train faster. So above 3 parameters were chosen to give the best possible hyperparameters in 10 different runs of the model using Hyperparameter tunner job.

Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
