# CAAC 项目 Makefile

.PHONY: help install test run-experiments update-docs clean

help:  ## 显示帮助信息
	@echo "可用的命令:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install:  ## 安装依赖
	pip install -r requirements.txt

test:  ## 运行测试
	python -m pytest tests/ -v

run-experiments:  ## 运行所有实验
	python run_experiments.py

update-docs:  ## 更新实验文档（根据最新的实验结果）
	python update_experiments_doc.py

clean:  ## 清理临时文件
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf *.egg-info/

# 组合命令
full-experiment:  ## 运行完整实验流程（实验+更新文档）
	$(MAKE) run-experiments
	$(MAKE) update-docs
	@echo "实验运行完成，文档已更新！" 