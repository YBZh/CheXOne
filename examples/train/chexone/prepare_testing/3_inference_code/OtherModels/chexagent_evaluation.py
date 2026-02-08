import json
import os
import gc
from collections import defaultdict
from typing import List, Dict, Any, Optional
import torch
from rich import print
from tqdm import tqdm
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from multiprocessing import Pool, Manager
import time
import psutil

try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    VLLM_AVAILABLE = True
except ImportError:
    print("Warning: vLLM not available. Please install vllm: pip install vllm")
    VLLM_AVAILABLE = False

from chestxagent_model.chexagent import CheXagent


# 全局变量存储每个进程的模型实例
_process_model = None

def init_worker(device, dtype):
    """初始化worker进程，只加载一次模型"""
    global _process_model
    try:
        # 将dtype字符串转换为torch.dtype
        if dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
            
        print(f"Worker process {os.getpid()}: Loading CheXagent model...")
        # import ipdb; ipdb.set_trace()
        _process_model = CheXagent(device=device, dtype=torch_dtype)
        print(f"Worker process {os.getpid()}: Model loaded successfully")
    except Exception as e:
        print(f"Worker process {os.getpid()}: Failed to load model: {e}")
        _process_model = None

def process_single_sample(sample):
    """处理单个样本的worker函数"""
    global _process_model
    try:
        if _process_model is None:
            # 保留input中所有的key value，并额外保存response（错误信息）
            result = {**sample, "response": "ERROR: Model not loaded"}
            return result
            
        images = sample.get("images")
        messages = sample.get("messages")
        prompt = messages[0].get("content")
        
        # # 处理images为null的情况
        # if images is None:
        #     # 对于没有图像的任务，直接返回一个默认响应或跳过
        #     result = {**sample, "response": "No image provided for this task"}
        #     return result
        
        response = _process_model.generate(images, prompt, do_sample=False)
        # 保留input中所有的key value，并额外保存response
        result = {**sample, "response": response}
        
        # 定期清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result
    except Exception as e:
        print(f"Error processing sample: {e}")
        # 保留input中所有的key value，并额外保存response（错误信息）
        result = {**sample, "response": f"ERROR: {str(e)}"}
        return result


class OptimizedMultiProcessCheXagentProcessor:
    """优化的多进程CheXagent处理器"""
    
    def __init__(self, 
                 batch_size: int = 8, 
                 device: str = "cuda", 
                 dtype: str = "bfloat16",
                 num_processes: int = None,
                 save_interval: int = 1000,
                 output_path: str = None,
                 memory_cleanup_interval: int = 100):
        
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.save_interval = save_interval
        self.output_path = output_path
        self.memory_cleanup_interval = memory_cleanup_interval
        self.processed_count = 0
        self.total_processed = 0
        self.all_results = []  # 存储所有结果
        
        # 设置进程数，默认为GPU数量
        if num_processes is None:
            self.num_processes = torch.cuda.device_count() if torch.cuda.is_available() else 1
        else:
            self.num_processes = num_processes
            
        print(f"=> Initializing Optimized MultiProcess CheXagent processor")
        print(f"=> Device: {device}")
        print(f"=> Number of processes: {self.num_processes}")
        print(f"=> Batch size: {batch_size}")
        print(f"=> Memory cleanup interval: {memory_cleanup_interval}")
        
        # 创建进程池，每个进程只初始化一次模型
        print(f"=> Creating process pool with {self.num_processes} workers...")
        self.pool = Pool(processes=self.num_processes, 
                        initializer=init_worker, 
                        initargs=(self.device, self.dtype))
        print(f"=> Process pool created successfully")
    
    def create_batches(self, data: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """将数据分割成批次"""
        batches = []
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            batches.append(batch)
        return batches
    
    def process_batch_multiprocess(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用多进程处理一个批次"""
        # 使用已经创建的进程池处理
        results = self.pool.map(process_single_sample, batch)
        return results
    
    def save_incremental_results(self, batch_results: List[Dict[str, Any]]):
        """优化的增量保存结果 - 累积所有结果并定期保存"""
        if self.output_path is None:
            return
            
        # 将批次结果添加到总结果中
        self.all_results.extend(batch_results)
        self.processed_count += len(batch_results)
        self.total_processed += len(batch_results)
        
        # 每处理完save_interval个样本就保存一次
        if self.processed_count >= self.save_interval:
            print(f"=> Saving incremental results: {self.total_processed} samples processed")
            
            # 保存所有累积的结果
            with open(self.output_path, 'w') as f:
                json.dump(self.all_results, f, ensure_ascii=False, indent=2)
            
            print(f"=> Incremental results saved to {self.output_path}")
            # 重置计数器
            self.processed_count = 0
    
    def cleanup_memory(self):
        """定期清理内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # 打印内存使用情况
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print(f"=> Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    
    def process_all(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理所有数据"""
        print(f"=> Processing {len(data)} samples with {self.num_processes} processes")
        print(f"=> Batch size: {self.batch_size}")
        if self.output_path:
            print(f"=> Incremental saving enabled: every {self.save_interval} samples to {self.output_path}")
        
        # 创建批次
        batches = self.create_batches(data)
        print(f"=> Created {len(batches)} batches")
        
        all_results = []
        
        # 处理每个批次
        for i, batch in enumerate(tqdm(batches, desc="Processing batches")):
            print(f"=> Processing batch {i+1}/{len(batches)} (size: {len(batch)})")
            batch_results = self.process_batch_multiprocess(batch)
            
            # 增量保存
            self.save_incremental_results(batch_results)
            
            # 定期清理内存
            if (i + 1) % self.memory_cleanup_interval == 0:
                print(f"=> Performing memory cleanup at batch {i+1}")
                self.cleanup_memory()
        
        # 保存最终结果
        if self.output_path:
            print(f"=> Saving final results: {self.total_processed} samples")
            with open(self.output_path, 'w') as f:
                json.dump(self.all_results, f, ensure_ascii=False, indent=2)
            print(f"=> Final results saved to {self.output_path}")
        
        return self.all_results
    
    def close(self):
        """关闭进程池"""
        if hasattr(self, 'pool') and self.pool is not None:
            print(f"=> Closing process pool...")
            self.pool.close()
            self.pool.join()
            print(f"=> Process pool closed successfully")
    
    def __del__(self):
        """析构函数，确保进程池被正确关闭"""
        self.close()


class OptimizedVLLMCheXagentProcessor:
    """优化的vLLM CheXagent处理器"""
    
    def __init__(self, 
                 batch_size: int = 8, 
                 device: str = "cuda", 
                 dtype: str = "bfloat16",
                 tensor_parallel_size: int = 1,
                 max_model_len: int = 4096,
                 gpu_memory_utilization: float = 0.9,
                 save_interval: int = 1000,
                 output_path: str = None,
                 memory_cleanup_interval: int = 50):
        
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.save_interval = save_interval
        self.output_path = output_path
        self.memory_cleanup_interval = memory_cleanup_interval
        self.processed_count = 0
        self.total_processed = 0
        self.all_results = []  # 存储所有结果
        
        if not VLLM_AVAILABLE:
            print("vLLM not available, falling back to standard processing")
            self.use_vllm = False
            self.model = CheXagent(device=device, dtype=torch.bfloat16)
            return
        
        self.checkpoint = "StanfordAIMI/CheXagent-2-3b"
        
        print(f"=> Attempting to load CheXagent model with vLLM...")
        print(f"=> Checkpoint: {self.checkpoint}")
        print(f"=> Batch size: {batch_size}")
        print(f"=> Tensor parallel size: {tensor_parallel_size}")
        print(f"=> Memory cleanup interval: {memory_cleanup_interval}")
        
        # Try to initialize vLLM engine
        try:
            self.llm = LLM(
                model=self.checkpoint,
                tensor_parallel_size=tensor_parallel_size,
                dtype=dtype,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True
            )
            
            # 设置采样参数
            self.sampling_params = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=512,
                stop=None
            )
            
            self.use_vllm = True
            print(f"=> vLLM model loaded successfully")
            
        except Exception as e:
            print(f"=> vLLM initialization failed: {str(e)}")
            if "not supported" in str(e) or "CheXagentForCausalLM" in str(e):
                print("=> CheXagent model architecture is not supported by vLLM")
                print("=> Falling back to standard processing with transformers")
            else:
                print(f"=> Unknown vLLM error: {str(e)}")
                print("=> Falling back to standard processing")
            
            self.use_vllm = False
            self.model = CheXagent(device=device, dtype=torch.bfloat16)
            print(f"=> Standard model loaded successfully")
    
    def create_batches(self, data: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """将数据分割成批次"""
        batches = []
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            batches.append(batch)
        return batches
    
    def prepare_prompts_for_batch(self, batch: List[Dict[str, Any]]) -> List[str]:
        """为批次准备提示词"""
        prompts = []
        for sample in batch:
            messages = sample.get("messages")
            prompt = messages[0].get("content")
            prompts.append(prompt)
        return prompts
    
    def process_batch_vllm(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用vLLM处理一个批次"""
        if not self.use_vllm:
            return self.process_batch_standard(batch)
        
        try:
            # 准备提示词
            prompts = self.prepare_prompts_for_batch(batch)
            
            # 使用vLLM生成
            outputs = self.llm.generate(prompts, self.sampling_params)
            
            # 处理结果
            batch_results = []
            for i, (sample, output) in enumerate(zip(batch, outputs)):
                response = output.outputs[0].text.strip()
                # 保留input中所有的key value，并额外保存response
                result = {**sample, "response": response}
                batch_results.append(result)
            
            return batch_results
            
        except Exception as e:
            print(f"Error in vLLM batch processing: {e}")
            print("Falling back to standard processing for this batch")
            return self.process_batch_standard(batch)
    
    def process_batch_standard(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """标准处理方式（备用）"""
        batch_results = []
        
        for sample in batch:
            try:
                images = sample.get("images")
                messages = sample.get("messages")
                prompt = messages[0].get("content")
                
                # # 处理images为null的情况
                # if images is None:
                #     result = {**sample, "response": "No image provided for this task"}
                #     batch_results.append(result)
                #     continue
                
                response = self.model.generate(images, prompt, do_sample=False)
                # 保留input中所有的key value，并额外保存response
                result = {**sample, "response": response}
                batch_results.append(result)
                
            except Exception as e:
                print(f"Error processing sample: {e}")
                # 保留input中所有的key value，并额外保存response（错误信息）
                result = {**sample, "response": f"ERROR: {str(e)}"}
                batch_results.append(result)
        
        return batch_results
    
    def save_incremental_results(self, batch_results: List[Dict[str, Any]]):
        """优化的增量保存结果 - 累积所有结果并定期保存"""
        if self.output_path is None:
            return
            
        # 将批次结果添加到总结果中
        self.all_results.extend(batch_results)
        self.processed_count += len(batch_results)
        self.total_processed += len(batch_results)
        
        # 每处理完save_interval个样本就保存一次
        if self.processed_count >= self.save_interval:
            print(f"=> Saving incremental results: {self.total_processed} samples processed")
            
            # 保存所有累积的结果
            with open(self.output_path, 'w') as f:
                json.dump(self.all_results, f, ensure_ascii=False, indent=2)
            
            print(f"=> Incremental results saved to {self.output_path}")
            self.processed_count = 0
    
    def cleanup_memory(self):
        """定期清理内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # 打印内存使用情况
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print(f"=> Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
        
        # 打印GPU内存使用情况
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
                cached = torch.cuda.memory_reserved(i) / 1024 / 1024
                print(f"=> GPU {i} memory: {allocated:.2f} MB allocated, {cached:.2f} MB cached")
    
    def process_all(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理所有数据"""
        print(f"=> Processing {len(data)} samples with batch size {self.batch_size}")
        print(f"=> Using {'vLLM' if self.use_vllm else 'standard'} processing")
        if self.output_path:
            print(f"=> Incremental saving enabled: every {self.save_interval} samples to {self.output_path}")
        
        # 创建批次
        batches = self.create_batches(data)
        print(f"=> Created {len(batches)} batches")
        
        # 处理每个批次
        for i, batch in enumerate(tqdm(batches, desc="Processing batches")):
            print(f"=> Processing batch {i+1}/{len(batches)} (size: {len(batch)})")
            batch_results = self.process_batch_vllm(batch)
            
            # 增量保存
            self.save_incremental_results(batch_results)
            
            # 定期清理内存
            if (i + 1) % self.memory_cleanup_interval == 0:
                print(f"=> Performing memory cleanup at batch {i+1}")
                self.cleanup_memory()
        
        # 保存最终结果
        if self.output_path:
            print(f"=> Saving final results: {self.total_processed} samples")
            with open(self.output_path, 'w') as f:
                json.dump(self.all_results, f, ensure_ascii=False, indent=2)
            print(f"=> Final results saved to {self.output_path}")
        
        return self.all_results
# ./data/ReXVQA/swiftformated-test.json
# examples/train/vlm_medical/chexbench/test_jsons/temporal_vqa_chestimgenome.json
# ./data/chexpert-public/chexpert_vqa_disease_classification_reasoning_frontal_v3.json
def main():
    parser = argparse.ArgumentParser(description="Optimized vLLM-accelerated batch processing for CheXagent evaluation")
    parser.add_argument("--data_path", type=str, 
                       default="./data/mimic-cxr/yulequan/vqa_disease_classification_reasoning_test_41821.json",
                       help="Path to the test data JSON file")
    parser.add_argument("--save_dir", type=str, default="./chexbench_eval_updated/chexagent/",
                       help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for processing")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--output_name", type=str, default="mimic_vqa_disease_classification_reasoning_test_41821.json",
                       help="Output filename")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Tensor parallel size for vLLM")
    parser.add_argument("--max_model_len", type=int, default=1024,
                       help="Maximum model length")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                       help="GPU memory utilization ratio")
    parser.add_argument("--multiprocess_mode", action="store_true",
                       help="Use multiprocess processing mode")
    parser.add_argument("--num_processes", type=int, default=8,
                       help="Number of processes for multiprocess mode (default: number of GPUs)")
    parser.add_argument("--save_interval", type=int, default=100,
                       help="Save results every N samples (default: 1000)")
    parser.add_argument("--memory_cleanup_interval", type=int, default=50,
                       help="Perform memory cleanup every N batches (default: 50)")
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置输出路径
    output_path = os.path.join(args.save_dir, args.output_name)
    
    # 加载数据
    print(f"=> Loading data from {args.data_path}")
    with open(args.data_path, 'r') as f:
        bench = json.load(f)
    print(f"=> Loaded {len(bench)} samples")
    
    if args.multiprocess_mode:
        # 多进程处理
        print("=> Using optimized multiprocess processing mode")
        processor = OptimizedMultiProcessCheXagentProcessor(
            batch_size=args.batch_size,
            device=args.device,
            num_processes=args.num_processes,
            save_interval=args.save_interval,
            output_path=output_path,
            memory_cleanup_interval=args.memory_cleanup_interval
        )
        try:
            results = processor.process_all(bench)
        finally:
            # 确保进程池被正确关闭
            processor.close()
    else:
        # 同步处理
        processor = OptimizedVLLMCheXagentProcessor(
            batch_size=args.batch_size,
            device=args.device,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            save_interval=args.save_interval,
            output_path=output_path,
            memory_cleanup_interval=args.memory_cleanup_interval
        )
        
        results = processor.process_all(bench)
    
    print(f"=> Processing completed! Results saved to {output_path}")


if __name__ == '__main__':
    # 多进程保护
    mp.set_start_method('spawn', force=True)
    main()

# 使用示例:
# python examples/train/vlm_medical/chexbench/axis_all_chexagent_vllm_optimized.py --data_path /home/zhangyabin/project/ms-swift/examples/train/vlm_medical/chexbench/test_jsons/axis_all_normal.json --output_name axis_all_chexagent_vllm_optimized.json --multiprocess_mode --batch_size 2 --num_processes 8 --save_interval 1000 --memory_cleanup_interval 100
# python examples/train/vlm_medical/chexbench/axis_all_chexagent_vllm_optimized.py --data_path /home/zhangyabin/project/ms-swift/data/Findings_Generation_ReXGradient_160K_test.json --output_name findings_generation_rexgradient_160k_test_chexagent_vllm_optimized.json --multiprocess_mode --batch_size 2 --num_processes 8 --save_interval 1000 --memory_cleanup_interval 100
# python examples/train/vlm_medical/chexbench/axis_all_chexagent_vllm_optimized.py --data_path ./data/ReXVQA/swiftformated-test.json --output_name rexvqa.json --multiprocess_mode --batch_size 2 --num_processes 8 --save_interval 100 --memory_cleanup_interval 100

# python examples/train/vlm_medical/chexbench/axis_all_chexagent_vllm_optimized.py --data_path /home/zhangyabin/project/ms-swift/./data/chexpert-public/chexpert_vqa_disease_classification_reasoning_frontal_v3.json --output_name axis_all_chexagent_vllm_optimized.json --multiprocess_mode --batch_size 2 --num_processes 8 --save_interval 1000 --memory_cleanup_interval 100