.PHONY: all generate_masking_configurations mask_data compute_distribution reconstruct evaluate_pud model

generate_masking_configurations:
	@mkdir -p $(CONFIG_DIR)
	@echo "Generating masking configurations for dataset $(ORIGINAL_DATA) with target variable $(CLASS_LABEL)"
	@python3 masking_configuration/generate_configs.py \
		--original_dataset "$(ORIGINAL_DATA)" \
		--class_label "$(CLASS_LABEL)" \
		--total_configurations $(TOTAL_CONFIGURATIONS) \
		--masking_configurations $(CONFIG_DIR)
	@echo "Masking configurations generated in $(CONFIG_DIR)"

mask_data:
	@echo "Applying masking configuration $(MASKING_CONFIG) to create masked dataset"
	@mkdir -p $(dir $(MASKED_DATA))
	@python3 mask/mask.py \
		--masking_configuration "$(MASKING_CONFIG)" \
		--masked_dataset "$(MASKED_DATA)"
	@echo "Masked data saved to $(MASKED_DATA)"

compute_distribution:
	@echo "Computing distributions for dataset $(DATASET)"
	@python3 marginal/marginal.py \
		--dataset "$(DATASET)" \
		--marginals "$(MARGINALS)" \
		--joint-distribution "$(JOINT_DISTRIBUTION)" \
		--class-label "$(CLASS_LABEL)"
	@echo "Marginals saved to $(MARGINALS)"
	@echo "Joint distributions saved to $(JOINT_DISTRIBUTION)"

reconstruct:
	@echo "Reconstructing data using $(ALGORITHM) algorithm"
	@if [ "$(ALGORITHM)" = "AEGIS_with_1D" ]; then \
		python3 reconstruction/aegis_with_1D.py \
			--masked-joint-distribution "$(MASKED_JOINT_DIST)" \
			--original-marginal "$(ORIGINAL_MARGINAL)" \
			--reconstructed-joint-distribution "$(RECONSTRUCTED_DIST)" \
			$(if $(MAX_ITER),--max-iter $(MAX_ITER),) \
			$(if $(TOL),--tol $(TOL),); \
	elif [ "$(ALGORITHM)" = "AEGIS_without_1D" ]; then \
		python3 reconstruction/aegis_without_1D.py \
			--masked-joint-distribution "$(MASKED_JOINT_DIST)" \
			--reconstructed-joint-distribution "$(RECONSTRUCTED_DIST)" \
			$(if $(MAX_ITER),--max-iter $(MAX_ITER),) \
			$(if $(TOL),--tol $(TOL),) \
			$(if $(ALPHA),--alpha $(ALPHA),); \
	elif [ "$(ALGORITHM)" = "Sampling" ]; then \
		python3 reconstruction/sampling.py \
			--masked-joint-distribution "$(MASKED_JOINT_DIST)" \
			--reconstructed-joint-distribution "$(RECONSTRUCTED_DIST)" \
			$(if $(SEED),--seed $(SEED),); \
	else \
		echo "Error: Invalid algorithm. Choose AEGIS_with_1D, AEGIS_without_1D, or Sampling"; \
		exit 1; \
	fi
	@echo "Reconstructed data saved to $(RECONSTRUCTED_DIST)"

evaluate_pud:
	@echo "Evaluating PUD using $(METRIC) metric"
	@if [ "$(METRIC)" = "chi2" ]; then \
		python3 metrics/chi2.py \
			--masked "$(MASKED_JOINT_DIST)" \
			--reconstructed "$(RECONSTRUCTED_JOINT_DIST)"; \
	elif [ "$(METRIC)" = "mi" ]; then \
		python3 metrics/mi.py \
			--masked "$(MASKED_JOINT_DIST)" \
			--reconstructed "$(RECONSTRUCTED_JOINT_DIST)"; \
	elif [ "$(METRIC)" = "tvd" ]; then \
		python3 metrics/tvd.py \
			--masked "$(MASKED_JOINT_DIST)" \
			--reconstructed "$(RECONSTRUCTED_JOINT_DIST)"; \
	elif [ "$(METRIC)" = "g3" ]; then \
		python3 metrics/g3.py \
			--masked "$(MASKED_JOINT_DIST)" \
			--reconstructed "$(RECONSTRUCTED_JOINT_DIST)"; \
	else \		echo "Error: Invalid metric. Choose one of: chi2, mi, tvd, g3, kl"; \
		exit 1; \
	fi

model:
	@echo "Running model on dataset $(DATA)"
	@if [ ! -f "$(MODEL)" ]; then \
		echo "Error: Model file $(MODEL) does not exist"; \
		exit 1; \
	fi
	@if [ ! -f "$(DATA)" ]; then \
		echo "Error: Data file $(DATA) does not exist"; \
		exit 1; \
	fi
	@python3 $(MODEL) \
		--data "$(DATA)" \
		--target "$(TARGET)"

