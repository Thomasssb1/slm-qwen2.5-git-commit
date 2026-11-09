provider "aws" {
  region = var.aws_region
}

resource "aws_s3_bucket" "transfer" {
  bucket        = lower(var.run_label)
  force_destroy = true
  tags          = { Run = var.run_label }
}

resource "aws_s3_bucket_versioning" "transfer" {
  bucket = aws_s3_bucket.transfer.id
  versioning_configuration {
    status = "Disabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "transfer" {
  bucket     = aws_s3_bucket.transfer.id
  depends_on = [aws_s3_bucket_versioning.transfer]

  rule {
    id     = "expire-run-artifacts"
    status = "Enabled"
    filter {
      prefix = ""
    }
    expiration {
      days = 7
    }
  }
}

resource "aws_s3_bucket_public_access_block" "transfer" {
  bucket                  = aws_s3_bucket.transfer.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "transfer" {
  bucket = aws_s3_bucket.transfer.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

output "bucket" {
  value = aws_s3_bucket.transfer.id
}
