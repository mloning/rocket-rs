# Makefile for easier setup

.PHONY: help build test

.DEFAULT_GOAL := help

help:
	# see http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

build: ## Build Rust and Python packages
	maturin develop
	poetry install

test: ## Run Python tests
	pytest py/
