
"""
This Lambda function creates a model card for the fine-tuned model
"""

import json
import boto3

def _create_model_card(file,event):
    sm_client = boto3.client("sagemaker")
    file['model_overview']['model_name']=event['model_name']
    file['model_overview']['model_id']=event["model_arn"]
    file['model_overview']['model_artifact']=[event['model_artifact']]
    file['model_overview']['problem_type']=""
    file['model_overview']['algorithm_type']="NeuralNetwork"
    file['model_overview']['model_description']=""
    file['model_overview']['model_creator']=""
    file['model_overview']['model_owner']=""
    file['model_overview']['inference_environment']['container_image']=[event['model_image']]
    file['business_details']['business_problem']=""
    file['business_details']['business_stakeholders']=""
    file['business_details']['line_of_business']=""
    file['intended_uses']['intended_uses']=""
    file['intended_uses']['explanations_for_risk_rating']=""
    file['intended_uses']['factors_affecting_model_efficiency']="Data Quality"
    file['intended_uses']['risk_rating']="Low"
    file['training_details']['training_job_details']['training_arn']=event['training_job_arn']
    file['training_details']['training_job_details']['training_datasets']=event["input_data"]
    file['training_details']['training_job_details']['training_environment']['container_image']=[event['training_image_arn']]
    file['training_details']['training_job_details']['hyper_parameters']=event["hyper_param"]
    file['training_details']['training_job_details']['training_metrics']=event["metrics"]
    file['evaluation_details']=[{
 'datasets': [event["llm_metric_output"]],
 'name': event["llm_metric_name"],
 'metric_groups': [{'name': event["llm_metric_name"],
   'metric_data': event["stereotype"]}],
 'evaluation_observation': 'NA'}]


    file=json.loads(str(file).replace("'",'"'))    
    model_card_name=event['model_card']
    
    model_card_list=sm_client.list_model_cards(
        NameContains=model_card_name,    
        SortBy='CreationTime',
        SortOrder='Descending'
    )
    if [x['ModelCardName'] for x in model_card_list['ModelCardSummaries'] if x['ModelCardName'] == model_card_name]:
        sm_client.update_model_card(
            ModelCardName=model_card_name,
            Content=json.dumps(file),   
            ModelCardStatus='PendingReview'
        )
    else:
        
        sm_client.create_model_card(
            ModelCardName=model_card_name,    
            Content=json.dumps(file),
            ModelCardStatus='PendingReview',    
        )
    
    

def lambda_handler(event, context):
    """ """
    sm_client = boto3.client("sagemaker")
    print(event)
    
    training_job_details=sm_client.describe_training_job(TrainingJobName=event["training_job_name"])
    event["hyper_param"]={key: value for key, value in training_job_details['HyperParameters'].items() if "sagemaker" not in key}
    event["hyper_param"]=[{"name": key, "value": value.strip('"')} for key, value in event["hyper_param"].items()]
    event["input_data"]=[f"{x['ChannelName']} -> {x['DataSource']['S3DataSource']['S3Uri']}" for x in training_job_details['InputDataConfig']]
    
    trial_c_list=sm_client.list_trial_components(
        SortBy='CreationTime',
        SortOrder='Descending',
    )['TrialComponentSummaries']
    trial_c_name=[x['TrialComponentName'] for x in trial_c_list if event['training_job_name'] in x['TrialComponentName']]
    if not trial_c_name:
        time.sleep(15)
        trial_c_name=[x['TrialComponentName'] for x in trial_c_list if event['training_job_name'] in x['TrialComponentName']]
    else: trial_c_name=trial_c_name[0]
    metrics=sm_client.describe_trial_component(
        TrialComponentName=trial_c_name)['Metrics']
    event["metrics"]=[{"name": item['MetricName'], "value": item['Min']} for item in metrics]
    eval_job_output=sm_client.describe_training_job(
            TrainingJobName=event['eval_job_name']
        )['OutputDataConfig']['S3OutputPath']+event['eval_job_name']+"/output/output.tar.gz"
    s3client=boto3.client("s3")
    import io
    import tarfile
    bucket=eval_job_output.split("//")[-1].split('/',1)[0]
    key=eval_job_output.split("//")[-1].split('/',1)[-1]
    
    s3_object = s3client.get_object(Bucket=bucket, Key=key)
    wholefile = s3_object['Body'].read()
    fileobj = io.BytesIO(wholefile)
    tarf = tarfile.open(fileobj=fileobj)
    names = tarf.getnames()
    json_file_content = tarf.extractfile(names[0]).read()
    json_file_content=json.loads(json_file_content.decode('utf-8'))[0]['category_scores']
    
    stereotypes_metrics=[{"name": item['name'], "type":"number","value": item['scores'][0]['value']} for item in json_file_content]
    event["stereotype"]=stereotypes_metrics
    event["llm_metric_output"]=eval_job_output
    event["llm_metric_name"]=json_file_content[0]['scores'][0]['name']
    
    model_card_template={'model_overview': {'model_name': '',
  'model_id': '',
  'model_artifact': [],
  'model_version': 1,
  'problem_type': '',
  'algorithm_type': '',
  'model_description': '',
  'model_creator': '',
  'model_owner': '',
  'inference_environment': {'container_image': []}},
 'business_details': {'business_problem': '',
  'business_stakeholders': '',
  'line_of_business': ''},
 'intended_uses': {'intended_uses': '',
  'explanations_for_risk_rating': '',
  'factors_affecting_model_efficiency': '',
  'risk_rating': ''},
 'training_details': {'objective_function': {'function': {'function': 'Maximize',
    'facet': 'Accuracy'}},
  'training_job_details': {'training_arn': '',
   'training_datasets': [],
   'training_environment': {'container_image': []},
   'hyper_parameters': [],
   
   'user_provided_hyper_parameters': []}},
    'evaluation_details': [],     
                        }
    
    model_card_template=json.loads(str(model_card_template).replace("'",'"'))
    _create_model_card(model_card_template,event)

    return {
        "statusCode": 200,
        "body": json.dumps("Created Model Card!"),
        "other_key": "example_value",
    }
