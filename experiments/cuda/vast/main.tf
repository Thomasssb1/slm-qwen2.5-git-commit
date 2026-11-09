terraform {
  required_version = ">= 1.6, < 2.0"

  backend "local" {}

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "6.72.0"
    }
    vastai = {
      source  = "berops/vastai"
      version = "0.1.1"
    }
  }
}

provider "vastai" {
  api_url = "https://console.vast.ai"
}

resource "vastai_instance" "smoke" {
  count      = var.launch ? 1 : 0
  depends_on = [aws_s3_bucket_public_access_block.transfer]

  label          = var.run_label
  image          = var.image
  image_login    = var.image_login
  vm             = false
  runtype        = "args"
  args           = ["/opt/qwen-commit/startup.py", jsonencode(var.job)]
  cancel_unavail = true
  disk           = 80

  lifecycle {
    precondition {
      condition     = var.job != null
      error_message = "Provide the transfer job before launching."
    }
  }

  search_offer {
    type              = "ondemand"
    gpu_name          = { in = distinct([for name in var.gpu_names : trimspace(name)]) }
    num_gpus          = { eq = 1 }
    gpu_ram           = { gte = 24000 }
    cpu_arch          = { eq = "amd64" }
    cuda_max_good     = { gte = 13.2 }
    disk_space        = { gte = 80 }
    allocated_storage = 80
    duration          = { gte = var.rental_seconds + 300 }
    reliability       = { gte = 0.98 }
    verified          = { eq = true }
    inet_down         = { gte = 100 }
    inet_down_cost    = { lte = 0.01 }
    inet_up_cost      = { lte = 0.01 }
    dph_total         = { lte = var.max_hourly_price }
    order             = [["dph_total", "asc"]]
  }
}

output "instance" {
  value = var.launch ? {
    id           = vastai_instance.smoke[0].id
    hourly_price = vastai_instance.smoke[0].dph_total
    gpu_name     = vastai_instance.smoke[0].gpu_name
  } : null
}
