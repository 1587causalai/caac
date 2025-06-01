#!/usr/bin/env python3
"""
CAAC参数化测试示例脚本
展示如何使用不同的参数组合进行各种实验
"""

import subprocess
import sys
import os

def run_command(command, description):
    """运行命令并显示描述"""
    print(f"\n🧪 {description}")
    print("=" * 60)
    print(f"命令: {command}")
    print("=" * 60)
    
    # 询问用户是否执行
    response = input("是否执行此命令? (y/n/q): ").lower().strip()
    if response == 'q':
        print("退出演示")
        return False
    elif response == 'y':
        try:
            subprocess.run(command, shell=True, check=True)
            print("✅ 命令执行完成")
        except subprocess.CalledProcessError as e:
            print(f"❌ 命令执行失败: {e}")
        except KeyboardInterrupt:
            print("\n⏸️ 用户中断执行")
    else:
        print("⏭️ 跳过此命令")
    
    return True

def show_examples():
    """展示各种参数化示例"""
    print("🎛️ CAAC参数化测试示例演示")
    print("=" * 80)
    print("这个脚本将展示如何使用不同的参数组合来运行CAAC鲁棒性测试")
    print("你可以选择运行任何感兴趣的示例")
    print("=" * 80)
    
    # 示例列表
    examples = [
        {
            "description": "1. 基础使用 - 默认参数快速测试",
            "command": "python run_quick_robustness_test.py"
        },
        {
            "description": "2. 自定义噪声水平 - 只测试轻度噪声",
            "command": "python run_quick_robustness_test.py --noise-levels 0.0 0.05 0.10"
        },
        {
            "description": "3. 极简测试 - 最快验证（2分钟）",
            "command": "python run_quick_robustness_test.py --noise-levels 0.0 0.20 --epochs 30 --datasets breast_cancer"
        },
        {
            "description": "4. 深度网络实验 - 更复杂的网络结构",
            "command": "python run_quick_robustness_test.py --representation-dim 256 --feature-hidden-dims 128 64 32 --abduction-hidden-dims 256 128 64"
        },
        {
            "description": "5. 超参数调优 - 不同学习率和批量大小",
            "command": "python run_quick_robustness_test.py --learning-rate 0.0001 --batch-size 128 --epochs 75"
        },
        {
            "description": "6. 精细噪声分析 - 更多噪声水平",
            "command": "python run_standard_robustness_test.py --noise-levels 0.0 0.02 0.05 0.08 0.10 0.15 0.20"
        },
        {
            "description": "7. 快速标准测试 - 减少训练时间",
            "command": "python run_standard_robustness_test.py --epochs 75 --early-stopping-patience 8"
        },
        {
            "description": "8. 大规模网络测试 - 测试网络容量影响",
            "command": "python run_standard_robustness_test.py --representation-dim 512 --feature-hidden-dims 256 128 64"
        },
        {
            "description": "9. 极端噪声测试 - 测试高噪声环境",
            "command": "python run_quick_robustness_test.py --noise-levels 0.0 0.20 0.40 0.60"
        },
        {
            "description": "10. 自定义数据集组合 - 只测试特定数据集",
            "command": "python run_standard_robustness_test.py --datasets breast_cancer optical_digits digits"
        }
    ]
    
    # 显示所有示例
    print("\n📋 可用示例:")
    for i, example in enumerate(examples, 1):
        print(f"{i:2d}. {example['description'].split(' - ')[1]}")
    
    print("\n选择要演示的示例:")
    print("输入数字选择示例，输入 'all' 查看所有命令，输入 'q' 退出")
    
    while True:
        choice = input("\n请选择 (1-10/all/q): ").strip().lower()
        
        if choice == 'q':
            print("退出演示")
            break
        elif choice == 'all':
            print("\n📖 所有示例命令:")
            print("=" * 80)
            for i, example in enumerate(examples, 1):
                print(f"\n{i:2d}. {example['description']}")
                print(f"    命令: {example['command']}")
            print("=" * 80)
        elif choice.isdigit() and 1 <= int(choice) <= len(examples):
            idx = int(choice) - 1
            if not run_command(examples[idx]["command"], examples[idx]["description"]):
                break
        else:
            print("无效选择，请输入 1-10、'all' 或 'q'")

def show_parameter_help():
    """显示参数说明"""
    print("\n📖 参数详细说明:")
    print("=" * 60)
    
    params = {
        "--noise-levels": {
            "说明": "噪声水平列表，值在 [0.0, 1.0] 范围内",
            "示例": "--noise-levels 0.0 0.05 0.10 0.15 0.20",
            "默认": "快速测试: [0.0, 0.10, 0.20], 标准测试: [0.0, 0.05, 0.10, 0.15, 0.20]"
        },
        "--representation-dim": {
            "说明": "表征维度，影响模型容量",
            "示例": "--representation-dim 256",
            "默认": "128"
        },
        "--feature-hidden-dims": {
            "说明": "特征网络隐藏层维度列表",
            "示例": "--feature-hidden-dims 128 64 32",
            "默认": "[64]"
        },
        "--abduction-hidden-dims": {
            "说明": "推断网络隐藏层维度列表",
            "示例": "--abduction-hidden-dims 256 128 64",
            "默认": "[128, 64]"
        },
        "--batch-size": {
            "说明": "批量大小，影响训练稳定性和速度",
            "示例": "--batch-size 128",
            "默认": "64"
        },
        "--epochs": {
            "说明": "最大训练轮数",
            "示例": "--epochs 200",
            "默认": "快速测试: 100, 标准测试: 150"
        },
        "--learning-rate": {
            "说明": "学习率",
            "示例": "--learning-rate 0.0001",
            "默认": "0.001"
        },
        "--early-stopping-patience": {
            "说明": "早停耐心值",
            "示例": "--early-stopping-patience 20",
            "默认": "快速测试: 10, 标准测试: 15"
        },
        "--datasets": {
            "说明": "选择要测试的数据集",
            "示例": "--datasets breast_cancer optical_digits digits",
            "默认": "快速测试: 4个小数据集, 标准测试: 6个数据集"
        }
    }
    
    for param, info in params.items():
        print(f"\n{param}")
        print(f"  说明: {info['说明']}")
        print(f"  示例: {info['示例']}")
        print(f"  默认: {info['默认']}")

def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        show_parameter_help()
        return
    
    print("选择操作:")
    print("1. 运行参数化示例演示")
    print("2. 查看参数详细说明")
    print("3. 退出")
    
    choice = input("\n请选择 (1-3): ").strip()
    
    if choice == '1':
        show_examples()
    elif choice == '2':
        show_parameter_help()
    elif choice == '3':
        print("退出")
    else:
        print("无效选择")

if __name__ == "__main__":
    main() 