import hashlib
import boto3
from botocore.exceptions import ClientError
from config import (
    S3_BUCKET_NAME,
    S3_REGION,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    INDEX_PREFIX,
)
from logger import logger


def get_s3_client():
    return boto3.client(
        "s3",
        region_name=S3_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def get_s3_key(filename: str) -> str:
    return f"{INDEX_PREFIX}/{filename}"


def upload_file(file_data: bytes, filename: str) -> str:
    s3_client = get_s3_client()
    s3_key = get_s3_key(filename)

    from io import BytesIO

    s3_client.upload_fileobj(
        BytesIO(file_data),
        S3_BUCKET_NAME,
        s3_key,
        ExtraArgs={"ContentType": "application/pdf"},
    )

    return s3_key


def download_file(filename: str) -> bytes:
    s3_client = get_s3_client()
    s3_key = get_s3_key(filename)

    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        return response["Body"].read()
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            logger.warning("S3 file not found: %s", filename)
            raise FileNotFoundError(f"File not found in S3: {filename}")
        raise


def delete_file(filename: str) -> bool:
    s3_client = get_s3_client()
    s3_key = get_s3_key(filename)

    try:
        s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        logger.info("Deleted object from S3: %s", s3_key)
        return True
    except ClientError as e:
        logger.exception("Error deleting file from S3: %s", e)
        return False


def file_exists(filename: str) -> bool:
    s3_client = get_s3_client()
    s3_key = get_s3_key(filename)

    try:
        s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        return True
    except ClientError:
        return False


def list_files() -> list[str]:
    s3_client = get_s3_client()
    prefix = f"{INDEX_PREFIX}/"

    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=prefix)
        if "Contents" not in response:
            return []

        files = []
        for obj in response["Contents"]:
            key = obj["Key"]
            if key.startswith(prefix):
                filename = key[len(prefix) :]
                if filename:
                    files.append(filename)
        return files
    except ClientError as e:
        logger.exception("Error listing files from S3: %s", e)
        return []


def compute_file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()
