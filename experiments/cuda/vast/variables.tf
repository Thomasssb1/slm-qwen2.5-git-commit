variable "ssh_public_key" {
  type = string
}

variable "run_label" {
  type = string
}

variable "gpu_names" {
  description = "Allowed Vast GPU model names. Offers are ranked by price, not list order."
  type        = list(string)
  default     = ["RTX 3090", "RTX PRO 4000"]
  nullable    = false

  validation {
    condition     = length(var.gpu_names) > 0 && alltrue([for name in var.gpu_names : try(trimspace(name) != "", false)])
    error_message = "Provide at least one nonblank GPU model name."
  }
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
