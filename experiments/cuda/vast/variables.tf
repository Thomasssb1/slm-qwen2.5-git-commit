variable "aws_region" {
  type    = string
  default = "eu-west-2"
}

variable "launch" {
  description = "Create storage first; launch after the runner prepares the transfer job."
  type        = bool
  default     = false
}

variable "image" {
  description = "Published image pinned by registry digest."
  type        = string
}

variable "image_login" {
  description = "Optional docker login arguments for a private registry."
  type        = string
  sensitive   = true
  default     = null
}

variable "job" {
  type = object({
    run_id        = string
    check         = optional(string, "cuda")
    sample_sha256 = optional(string)
    sample_url    = optional(string)
    result_url    = string
    hold_seconds  = number
  })
  sensitive = true
  default   = null

  validation {
    condition     = var.job == null ? true : contains(["cuda", "model", "inference"], var.job.check) && (var.job.check != "cuda" || (var.job.sample_url != null && var.job.sample_sha256 != null))
    error_message = "Choose cuda with a sample URL and checksum, or model/inference without a dataset."
  }
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
