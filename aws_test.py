import boto3

s3 = boto3.client("s3")

s3.put_object(
    Bucket="resume-rag-storage-preena-2026",
    Key="test.txt",
    Body="Hello from Smart Hire RAG system"
)

print("Upload successful!")