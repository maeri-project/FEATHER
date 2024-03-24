clean:
	rm -rf results

gemm_dse:
	export PYTHONPATH="./scripts_gemm:$PYTHONPATH"
	python3 ./scripts_gemm/search_layoutloop_all_design.py results

conv_dse:
	export PYTHONPATH="./scripts:$PYTHONPATH"
	python3 ./scripts/search_layoutloop_all_design.py results