import json
import boto3
import time
def lambda_handler(event, context):
    sm_client=boto3.client("sagemaker")
    # The name of the model created in the Pipeline CreateModelStep
    model_name = event["model_name"]
    endpoint_config_name = event["endpoint_config_name"]
    endpoint_name = event["endpoint_name"]
    role = event["role"] 


    create_endpoint_config_response = sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "InstanceType": "ml.g5.2xlarge",
                "InitialVariantWeight": 1,
                "InitialInstanceCount": 1,
                "ModelName": model_name,
                "VariantName": "AllTraffic",
            }
        ],
        
    )
    time.sleep(120)
    create_endpoint_response = sm_client.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name)
    return {
        "statusCode": 200,
        "body": json.dumps("Created Endpoint"),
        "other_key": "example_value",
    }
