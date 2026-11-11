provider "aws" {
  region                      = "eu-west-2"
  access_key                  = "local-test"
  secret_key                  = "local-test"
  skip_credentials_validation = true
  skip_requesting_account_id  = true
  skip_metadata_api_check     = true
}

provider "vastai" {
  api_key = "unused-local-plan-test"
}

variables {
  image  = "ghcr.io/example/cuda@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  launch = true
  run_config = {
    run_id        = "test"
    sample_sha256 = "sample-checksum"
    sample_url    = "https://example.invalid/sample"
    result_url    = "https://example.invalid/result"
    hold_seconds  = 3000
  }
  run_label        = "qwen-cuda-local-plan-test"
  max_hourly_price = 0.40
  rental_seconds   = 2700
}

run "bounded_container_rental" {
  command = plan
  variables {
    gpu_names = [" RTX 3090 ", "RTX PRO 4000", "RTX 3090"]
  }

  assert {
    condition     = !vastai_instance.smoke[0].vm && vastai_instance.smoke[0].runtype == "args" && vastai_instance.smoke[0].cancel_unavail
    error_message = "Launch our image directly in entrypoint mode."
  }

  assert {
    condition     = vastai_instance.smoke[0].search_offer.dph_total.lte == 0.40 && vastai_instance.smoke[0].search_offer.inet_down_cost.lte == 0.01 && vastai_instance.smoke[0].search_offer.inet_up_cost.lte == 0.01
    error_message = "The hourly and transfer price ceilings must constrain offer selection."
  }

  assert {
    condition     = vastai_instance.smoke[0].disk == 80 && vastai_instance.smoke[0].search_offer.allocated_storage == 80 && vastai_instance.smoke[0].search_offer.disk_space.gte == 80
    error_message = "The quote and offer filter must include the disk space being rented."
  }

  assert {
    condition     = vastai_instance.smoke[0].search_offer.num_gpus.eq == 1 && vastai_instance.smoke[0].search_offer.gpu_ram.gte == 24000 && vastai_instance.smoke[0].search_offer.cuda_max_good.gte == 13.2 && vastai_instance.smoke[0].search_offer.cpu_arch.eq == "amd64"
    error_message = "The selected GPU must match the built CUDA image."
  }

  assert {
    condition     = vastai_instance.smoke[0].search_offer.gpu_name.in == tolist(["RTX 3090", "RTX PRO 4000"]) && vastai_instance.smoke[0].search_offer.gpu_name.eq == null
    error_message = "Search every allowed model together, trimming whitespace and removing duplicates."
  }

  assert {
    condition     = vastai_instance.smoke[0].search_offer.order == tolist([tolist(["dph_total", "asc"])])
    error_message = "Rank matching offers by total hourly price, independent of GPU list order."
  }

  assert {
    condition     = vastai_instance.smoke[0].search_offer.vms_enabled == null && vastai_instance.smoke[0].search_offer.verified.eq && vastai_instance.smoke[0].search_offer.reliability.gte == 0.98 && vastai_instance.smoke[0].search_offer.duration.gte == 3000
    error_message = "Select a verified host available for the full rental window, without requiring VM support."
  }
}

run "default_gpu_allowlist" {
  command = plan
  assert {
    condition     = vastai_instance.smoke[0].search_offer.gpu_name.in == tolist(["RTX 3090", "RTX PRO 4000"])
    error_message = "Defaults must allow both the RTX 3090 and RTX PRO 4000."
  }
}

run "single_gpu_allowlist" {
  command = plan
  variables {
    gpu_names = ["RTX 4090"]
  }
  assert {
    condition     = vastai_instance.smoke[0].search_offer.gpu_name.in == tolist(["RTX 4090"])
    error_message = "A custom one-model allowlist must replace the defaults."
  }
}

run "reject_empty_gpu_allowlist" {
  command = plan
  variables {
    gpu_names = []
  }
  expect_failures = [var.gpu_names]
}

run "reject_blank_gpu_name" {
  command = plan
  variables {
    gpu_names = ["RTX 3090", "  "]
  }
  expect_failures = [var.gpu_names]
}

run "reject_null_gpu_name" {
  command = plan
  variables {
    gpu_names = ["RTX 3090", null]
  }
  expect_failures = [var.gpu_names]
}

run "reject_excessive_price" {
  command = plan
  variables {
    max_hourly_price = 11
  }
  expect_failures = [var.max_hourly_price]
}

run "reject_nonpositive_price" {
  command = plan
  variables {
    max_hourly_price = 0
  }
  expect_failures = [var.max_hourly_price]
}

run "storage_before_launch" {
  command = plan
  variables { launch = false }
  assert {
    condition     = length(vastai_instance.smoke) == 0
    error_message = "Storage preparation must not rent a GPU."
  }
  assert {
    condition     = aws_s3_bucket.transfer.bucket == "qwen-cuda-local-plan-test" && aws_s3_bucket.transfer.force_destroy
    error_message = "Use a run-specific bucket and remove its transfer objects on teardown."
  }
  assert {
    condition     = aws_s3_bucket_versioning.transfer.versioning_configuration[0].status == "Disabled"
    error_message = "The temporary transfer bucket must remain unversioned."
  }
  assert {
    condition     = one(aws_s3_bucket_lifecycle_configuration.transfer.rule).status == "Enabled" && one(aws_s3_bucket_lifecycle_configuration.transfer.rule).expiration[0].days == 7 && one(aws_s3_bucket_lifecycle_configuration.transfer.rule).filter[0].prefix == ""
    error_message = "Expire all leftover transfer objects after seven days."
  }
  assert {
    condition     = aws_s3_bucket_public_access_block.transfer.block_public_acls && aws_s3_bucket_public_access_block.transfer.block_public_policy && aws_s3_bucket_public_access_block.transfer.ignore_public_acls && aws_s3_bucket_public_access_block.transfer.restrict_public_buckets
    error_message = "The transfer bucket must block all public access."
  }
  assert {
    condition     = one(aws_s3_bucket_server_side_encryption_configuration.transfer.rule).apply_server_side_encryption_by_default[0].sse_algorithm == "AES256"
    error_message = "Encrypt the temporary sample and report."
  }
}

run "launch_uses_published_image_and_run_config" {
  command = plan
  assert {
    condition     = vastai_instance.smoke[0].image == var.image && vastai_instance.smoke[0].args[0] == "/opt/qwen-commit/startup.py" && jsondecode(vastai_instance.smoke[0].args[1]) == var.run_config
    error_message = "Run the image wrapper with the scoped container run configuration."
  }
}

run "reject_launch_without_run_config" {
  command = plan
  variables { run_config = null }
  expect_failures = [vastai_instance.smoke[0]]
}

run "model_load_without_dataset" {
  command = plan
  variables {
    run_config = {
      run_id       = "model-test"
      check        = "model"
      result_url   = "https://example.invalid/result"
      hold_seconds = 3000
    }
  }
  assert {
    condition     = jsondecode(vastai_instance.smoke[0].args[1]).check == "model" && jsondecode(vastai_instance.smoke[0].args[1]).sample_url == null
    error_message = "Pass model mode through to the container without requiring a training sample."
  }
}

run "reject_unknown_check" {
  command = plan
  variables {
    run_config = {
      run_id       = "bad-test"
      check        = "unknown"
      result_url   = "https://example.invalid/result"
      hold_seconds = 3000
    }
  }
  expect_failures = [var.run_config]
}

run "inference_without_dataset" {
  command = plan
  variables {
    run_config = {
      run_id       = "inference-test"
      check        = "inference"
      result_url   = "https://example.invalid/result"
      hold_seconds = 3000
    }
  }
  assert {
    condition     = jsondecode(vastai_instance.smoke[0].args[1]).check == "inference" && jsondecode(vastai_instance.smoke[0].args[1]).sample_url == null
    error_message = "Run synthetic inference without transferring a private dataset."
  }
}

run "reject_cuda_without_sample" {
  command = plan
  variables {
    run_config = {
      run_id       = "bad-test"
      result_url   = "https://example.invalid/result"
      hold_seconds = 3000
    }
  }
  expect_failures = [var.run_config]
}

run "validation_with_input" {
  command = plan
  variables {
    run_config = {
      run_id        = "validation-test"
      check         = "validation"
      sample_sha256 = "input-checksum"
      sample_url    = "https://example.invalid/validation-input.parquet"
      result_url    = "https://example.invalid/result"
      hold_seconds  = 3000
    }
  }
  assert {
    condition     = jsondecode(vastai_instance.smoke[0].args[1]).check == "validation" && jsondecode(vastai_instance.smoke[0].args[1]).sample_url == var.run_config.sample_url
    error_message = "Pass validation mode and its scoped input transfer through to the container."
  }
}

run "reject_validation_without_input" {
  command = plan
  variables {
    run_config = {
      run_id       = "bad-validation-test"
      check        = "validation"
      result_url   = "https://example.invalid/result"
      hold_seconds = 3000
    }
  }
  expect_failures = [var.run_config]
}
