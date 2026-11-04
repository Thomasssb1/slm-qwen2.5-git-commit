variable "ssh_public_key" {
  type = string
}

variable "run_label" {
  type = string
}

variable "gpu_name" {
  type    = string
  default = "RTX 3090"
}

variable "max_hourly_price" {
  type    = number
  default = 0.50

  validation {
    condition     = var.max_hourly_price > 0 && var.max_hourly_price <= 10
    error_message = "Use a positive hourly limit no greater than $10."
  }
}

variable "rental_seconds" {
  type = number
}
