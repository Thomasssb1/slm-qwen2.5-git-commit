# Plan only: no Vast API requests or billable resources.
provider "vastai" {
  api_key = "unused-local-plan-test"
}

variables {
  ssh_public_key   = "ssh-ed25519 local-plan-test"
  run_label        = "qwen-cuda-local-plan-test"
  gpu_name         = "RTX 3090"
  max_hourly_price = 0.40
  rental_seconds   = 2700
}

run "bounded_vm_rental" {
  command = plan

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
    condition     = vastai_instance.smoke.search_offer.num_gpus.eq == 1 && vastai_instance.smoke.search_offer.gpu_name.eq == "RTX 3090" && vastai_instance.smoke.search_offer.gpu_ram.gte == 24000 && vastai_instance.smoke.search_offer.cuda_max_good.gte == 13.2 && vastai_instance.smoke.search_offer.cpu_arch.eq == "amd64"
    error_message = "The selected GPU must match the built CUDA image."
  }

  assert {
    condition     = vastai_instance.smoke.search_offer.vms_enabled.eq && vastai_instance.smoke.search_offer.verified.eq && vastai_instance.smoke.search_offer.reliability.gte == 0.98 && vastai_instance.smoke.search_offer.duration.gte == 3000
    error_message = "Select a verified VM host available for the full rental window."
  }
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
