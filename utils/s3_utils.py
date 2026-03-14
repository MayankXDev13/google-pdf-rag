import boto3
from botocore.exceptions import ClientError
from config import (
    S3_BUCKET_NAME,
    S3_REGION,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    INDEX_PREFIX,
)
from io import BytesIO


def get_s3_client():
    """Get S3 client"""
    return boto3.client(
        "s3",
        region_name=S3_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def get_s3_key(filename: str) -> str:
    """Get S3 path(key)"""
    return f"{INDEX_PREFIX}/{filename}"


def upload_file(file_data: bytes, filename: str) -> str:
    """Upload the file to S3"""
    s3_client = get_s3_client()
    s3_key = get_s3_key(filename)

    s3_client.upload_fileobj(
        BytesIO(file_data),
        S3_BUCKET_NAME,
        s3_key,
        ExtraArgs={"ContentType": "application/pdf"},
    )

    return s3_key


def download_file(filename: str) -> bytes:
    """Download the file from S3"""
    s3_client = get_s3_client()
    s3_key = get_s3_key(filename)

    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        return response["Body"].read()
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            print(f"Warning: S3 file not found: {filename}")
            raise FileNotFoundError(f"File not found in S3: {filename}")
        raise


def delete_from_s3(filename: str) -> bool:
    """Delete the file from S3"""
    s3_client = get_s3_client()
    s3_key = get_s3_key(filename)

    try:
        s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        print(f"Deleted object from S3: {s3_key}")
        return True
    except ClientError as e:
        print(f"Error deleting file from S3: {e}")
        return False


def file_exists(filename: str) -> bool:
    """Check if the file exists in S3"""
    s3_client = get_s3_client()
    s3_key = get_s3_key(filename)

    try:
        s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        return True
    except ClientError:
        return False


def list_files() -> list[str]:
    """List all files in the S3 bucket with the index prefix"""
    s3_client = get_s3_client()

    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET_NAME, Prefix=INDEX_PREFIX + "/"
        )
        files = []
        if "Contents" in response:
            for obj in response["Contents"]:
                key = obj["Key"]
                filename = key.replace(INDEX_PREFIX + "/", "", 1)
                files.append(filename)
        return files
    except ClientError as e:
        print(f"Error listing files from S3: {e}")
        return []
