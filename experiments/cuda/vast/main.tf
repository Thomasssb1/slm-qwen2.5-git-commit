terraform {
  required_version = ">= 1.6, < 2.0"

  required_providers {
    vastai = {
      source  = "berops/vastai"
      version = "0.1.1"
    }
  }
}

provider "vastai" {
  api_url = "https://console.vast.ai"
}

resource "vastai_ssh_key" "run" {
  public_key = var.ssh_public_key
}

resource "vastai_instance" "smoke" {
  depends_on = [vastai_ssh_key.run]

  label          = var.run_label
  image          = "docker.io/vastai/kvm:ubuntu_terminal"
  vm             = true
  runtype        = "ssh"
  cancel_unavail = true
  disk           = 80

  search_offer {
    type              = "ondemand"
    gpu_name          = { eq = var.gpu_name }
    num_gpus          = { eq = 1 }
    gpu_ram           = { gte = 24000 }
    cpu_arch          = { eq = "amd64" }
    cuda_max_good     = { gte = 13.2 }
    disk_space        = { gte = 80 }
    allocated_storage = 80
    duration          = { gte = var.rental_seconds + 300 }
    reliability       = { gte = 0.98 }
    verified          = { eq = true }
    vms_enabled       = { eq = true }
    inet_down         = { gte = 100 }
    inet_down_cost    = { lte = 0.01 }
    inet_up_cost      = { lte = 0.01 }
    dph_total         = { lte = var.max_hourly_price }
    order             = [["dph_total", "asc"]]
  }
}

output "instance" {
  value = {
    id           = vastai_instance.smoke.id
    ssh_host     = vastai_instance.smoke.ssh_host
    ssh_port     = vastai_instance.smoke.ssh_port
    hourly_price = vastai_instance.smoke.dph_total
    gpu_name     = vastai_instance.smoke.gpu_name
  }
}
