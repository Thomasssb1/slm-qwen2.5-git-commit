# Plan only: no Vast API requests or billable resources.
provider "vastai" {
  api_key = "unused-local-plan-test"
}

variables {
  ssh_public_key   = "ssh-ed25519 local-plan-test"
  run_label        = "qwen-cuda-local-plan-test"
  max_hourly_price = 0.40
  rental_seconds   = 2700
}

run "bounded_vm_rental" {
  command = plan
  variables {
    gpu_names = [" RTX 3090 ", "RTX PRO 4000", "RTX 3090"]
  }

  assert {
    condition     = vastai_instance.smoke.vm && vastai_instance.smoke.runtype == "ssh" && vastai_instance.smoke.cancel_unavail
    error_message = "The smoke test needs an immediately available SSH VM for nested Docker."
  }

  assert {
    condition     = vastai_instance.smoke.search_offer.dph_total.lte == 0.40 && vastai_instance.smoke.search_offer.inet_down_cost.lte == 0.01 && vastai_instance.smoke.search_offer.inet_up_cost.lte == 0.01
    error_message = "The hourly and transfer price ceilings must constrain offer selection."
  }

  assert {
    condition     = vastai_instance.smoke.disk == 80 && vastai_instance.smoke.search_offer.allocated_storage == 80 && vastai_instance.smoke.search_offer.disk_space.gte == 80
    error_message = "The quote and offer filter must include the disk space being rented."
  }

  assert {
    condition     = vastai_instance.smoke.search_offer.num_gpus.eq == 1 && vastai_instance.smoke.search_offer.gpu_ram.gte == 24000 && vastai_instance.smoke.search_offer.cuda_max_good.gte == 13.2 && vastai_instance.smoke.search_offer.cpu_arch.eq == "amd64"
    error_message = "The selected GPU must match the built CUDA image."
  }

  assert {
    condition     = vastai_instance.smoke.search_offer.gpu_name.in == tolist(["RTX 3090", "RTX PRO 4000"]) && vastai_instance.smoke.search_offer.gpu_name.eq == null
    error_message = "Search every allowed model together, trimming whitespace and removing duplicates."
  }

  assert {
    condition     = vastai_instance.smoke.search_offer.order == tolist([tolist(["dph_total", "asc"])])
    error_message = "Rank matching offers by total hourly price, independent of GPU list order."
  }

  assert {
    condition     = vastai_instance.smoke.search_offer.vms_enabled.eq && vastai_instance.smoke.search_offer.verified.eq && vastai_instance.smoke.search_offer.reliability.gte == 0.98 && vastai_instance.smoke.search_offer.duration.gte == 3000
    error_message = "Select a verified VM host available for the full rental window."
  }
}

run "default_gpu_allowlist" {
  command = plan
  assert {
    condition     = vastai_instance.smoke.search_offer.gpu_name.in == tolist(["RTX 3090", "RTX PRO 4000"])
    error_message = "Defaults must allow both the RTX 3090 and RTX PRO 4000."
  }
}

run "single_gpu_allowlist" {
  command = plan
  variables {
    gpu_names = ["RTX 4090"]
  }
  assert {
    condition     = vastai_instance.smoke.search_offer.gpu_name.in == tolist(["RTX 4090"])
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
