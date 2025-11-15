"""
Memory Usage Measurement Script for Multimodal Arousal Detection

This script measures actual memory usage during:
1. Model initialization
2. Forward pass
3. Backward pass
4. Full training step

Usage:
    python measure_memory.py --batch_size 4 --base_ch 32
"""

import torch
import numpy as np
import psutil
import os
import argparse
from models.DeepSleepFinal import DeepSleepMultimodal


def format_bytes(bytes_val):
    """Convert bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def get_gpu_memory():
    """Get current GPU memory usage"""
    if not torch.cuda.is_available():
        return None

    return {
        'allocated': torch.cuda.memory_allocated(0),
        'reserved': torch.cuda.memory_reserved(0),
        'max_allocated': torch.cuda.max_memory_allocated(0)
    }


def get_cpu_memory():
    """Get current CPU memory usage"""
    process = psutil.Process(os.getpid())
    return {
        'rss': process.memory_info().rss,
        'vms': process.memory_info().vms,
        'percent': process.memory_percent()
    }


def print_memory_stats(stage, gpu_mem, cpu_mem):
    """Print formatted memory statistics"""
    print(f"\n{'='*60}")
    print(f"Memory Usage: {stage}")
    print(f"{'='*60}")

    if gpu_mem:
        print(f"GPU Memory:")
        print(f"  Allocated:     {format_bytes(gpu_mem['allocated'])}")
        print(f"  Reserved:      {format_bytes(gpu_mem['reserved'])}")
        print(f"  Max Allocated: {format_bytes(gpu_mem['max_allocated'])}")
    else:
        print(f"GPU: Not available (using CPU)")

    print(f"\nCPU Memory:")
    print(f"  RSS (Resident): {format_bytes(cpu_mem['rss'])}")
    print(f"  VMS (Virtual):  {format_bytes(cpu_mem['vms'])}")
    print(f"  Percent:        {cpu_mem['percent']:.2f}%")
    print(f"{'='*60}")


def measure_model_memory(n_channels=9, base_ch=32, use_attention=True, device='cuda'):
    """Measure memory usage of model initialization"""
    print("\n🔍 Measuring Model Initialization Memory...")

    # Before model creation
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    gpu_before = get_gpu_memory()
    cpu_before = get_cpu_memory()

    # Create model
    model = DeepSleepMultimodal(
        n_channels=n_channels,
        base_ch=base_ch,
        use_attention=use_attention
    )

    if device == 'cuda' and torch.cuda.is_available():
        model = model.to('cuda:0')

    # After model creation
    gpu_after = get_gpu_memory()
    cpu_after = get_cpu_memory()

    # Calculate differences
    if gpu_after and gpu_before:
        gpu_diff = gpu_after['allocated'] - gpu_before['allocated']
        print(f"✅ Model created on GPU: {format_bytes(gpu_diff)}")
    else:
        cpu_diff = cpu_after['rss'] - cpu_before['rss']
        print(f"✅ Model created on CPU: {format_bytes(cpu_diff)}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size = total_params * 4  # float32 = 4 bytes

    print(f"\nModel Statistics:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Parameter size:       {format_bytes(param_size)}")

    return model


def measure_forward_memory(model, batch_size=4, device='cuda'):
    """Measure memory usage during forward pass"""
    print(f"\n🔍 Measuring Forward Pass Memory (batch_size={batch_size})...")

    # Create dummy inputs
    x_time_combined = torch.randn(batch_size, 9, 4, 3000)
    x_spec = torch.randn(batch_size, 9, 51, 119)
    x_stat = torch.randn(batch_size, 9, 6, 59)

    if device == 'cuda' and torch.cuda.is_available():
        x_time_combined = x_time_combined.cuda()
        x_spec = x_spec.cuda()
        x_stat = x_stat.cuda()

    # Input size
    input_size = (
        x_time_combined.numel() * 4 +
        x_spec.numel() * 4 +
        x_stat.numel() * 4
    )
    print(f"  Input data size: {format_bytes(input_size)}")

    # Before forward
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    gpu_before = get_gpu_memory()
    cpu_before = get_cpu_memory()

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x_time_combined, x_spec, x_stat, comp=True)

    # After forward
    gpu_after = get_gpu_memory()
    cpu_after = get_cpu_memory()

    print_memory_stats(f"After Forward Pass (batch_size={batch_size})", gpu_after, cpu_after)

    print(f"\n  Output shape: {output.shape}")
    output_size = output.numel() * 4
    print(f"  Output size: {format_bytes(output_size)}")

    return output


def measure_backward_memory(model, batch_size=4, device='cuda'):
    """Measure memory usage during backward pass"""
    print(f"\n🔍 Measuring Backward Pass Memory (batch_size={batch_size})...")

    # Create dummy inputs
    x_time_combined = torch.randn(batch_size, 9, 4, 3000)
    x_spec = torch.randn(batch_size, 9, 51, 119)
    x_stat = torch.randn(batch_size, 9, 6, 59)
    y = torch.randn(batch_size, 3000)

    if device == 'cuda' and torch.cuda.is_available():
        x_time_combined = x_time_combined.cuda()
        x_spec = x_spec.cuda()
        x_stat = x_stat.cuda()
        y = y.cuda()

    # Before backward
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model.train()

    # Forward
    output = model(x_time_combined, x_spec, x_stat, comp=False)

    # Resize output to match y
    if output.shape[2] != y.shape[1]:
        output = torch.nn.functional.interpolate(
            output, size=y.shape[1], mode='linear', align_corners=False
        )
    output = output.squeeze(1)

    # Simple loss
    loss = torch.nn.functional.mse_loss(output, y)

    gpu_before_backward = get_gpu_memory()

    # Backward
    loss.backward()

    # After backward
    gpu_after = get_gpu_memory()
    cpu_after = get_cpu_memory()

    print_memory_stats(f"After Backward Pass (batch_size={batch_size})", gpu_after, cpu_after)

    if gpu_after and gpu_before_backward:
        grad_mem = gpu_after['allocated'] - gpu_before_backward['allocated']
        print(f"\n  Gradient memory: {format_bytes(grad_mem)}")


def measure_optimizer_memory(model, device='cuda'):
    """Measure memory usage with optimizer"""
    print(f"\n🔍 Measuring Optimizer Memory...")

    # Before optimizer
    gpu_before = get_gpu_memory()

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # After optimizer
    gpu_after = get_gpu_memory()
    cpu_after = get_cpu_memory()

    print_memory_stats("After Optimizer Creation", gpu_after, cpu_after)

    if gpu_after and gpu_before:
        opt_mem = gpu_after['allocated'] - gpu_before['allocated']
        print(f"\n  Optimizer state memory: {format_bytes(opt_mem)}")

    return optimizer


def measure_full_training_step(model, optimizer, batch_size=4, device='cuda'):
    """Measure memory during a full training iteration"""
    print(f"\n🔍 Measuring Full Training Step (batch_size={batch_size})...")

    # Create dummy batch
    x_time_combined = torch.randn(batch_size, 9, 4, 3000)
    x_spec = torch.randn(batch_size, 9, 51, 119)
    x_stat = torch.randn(batch_size, 9, 6, 59)
    y = torch.randn(batch_size, 3000)

    if device == 'cuda' and torch.cuda.is_available():
        x_time_combined = x_time_combined.cuda()
        x_spec = x_spec.cuda()
        x_stat = x_stat.cuda()
        y = y.cuda()

    # Reset peak stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model.train()

    # Training step
    optimizer.zero_grad()

    # Forward
    output = model(x_time_combined, x_spec, x_stat, comp=False)

    # Resize
    if output.shape[2] != y.shape[1]:
        output = torch.nn.functional.interpolate(
            output, size=y.shape[1], mode='linear', align_corners=False
        )
    output = output.squeeze(1)

    # Loss
    loss = torch.nn.functional.mse_loss(output, y)

    # Backward
    loss.backward()

    # Optimizer step
    optimizer.step()

    # Measure peak
    gpu_mem = get_gpu_memory()
    cpu_mem = get_cpu_memory()

    print_memory_stats(f"Peak During Training Step (batch_size={batch_size})", gpu_mem, cpu_mem)

    print(f"\n  Loss: {loss.item():.6f}")


def measure_chunk_preprocessing():
    """Estimate memory for preprocessing a 60-second chunk"""
    print(f"\n🔍 Estimating Preprocessing Memory (60-second chunk)...")

    # Simulated data sizes
    n_channels = 9
    fs = 50
    chunk_sec = 60
    samples = fs * chunk_sec  # 3000

    # Time domain
    x_time_raw = n_channels * samples * 4  # float32
    envelope = n_channels * samples * 4
    derivs = n_channels * samples * 4 * 2  # first + second
    x_time_combined = n_channels * 4 * samples * 4

    # Frequency domain
    freq_bins = 51
    time_bins = 119
    x_spec = n_channels * freq_bins * time_bins * 4

    # Statistical
    n_features = 6
    windows = 59
    x_stat = n_channels * n_features * windows * 4

    # Labels
    y_time = samples * 4
    y_spec = time_bins * 4

    # Total
    total = (x_time_raw + envelope + derivs + x_time_combined +
             x_spec + x_stat + y_time + y_spec)

    print(f"\nEstimated Memory per 60-second Chunk:")
    print(f"  x_time_raw:      {format_bytes(x_time_raw)}")
    print(f"  envelope:        {format_bytes(envelope)}")
    print(f"  derivatives:     {format_bytes(derivs)}")
    print(f"  x_time_combined: {format_bytes(x_time_combined)}")
    print(f"  x_spec:          {format_bytes(x_spec)}")
    print(f"  x_stat:          {format_bytes(x_stat)}")
    print(f"  y_time:          {format_bytes(y_time)}")
    print(f"  y_spec:          {format_bytes(y_spec)}")
    print(f"  {'─'*40}")
    print(f"  Total:           {format_bytes(total)}")

    # For 8-hour file
    hours = 8
    n_chunks = hours * 60
    total_file = total * n_chunks
    print(f"\nFor 8-hour file ({n_chunks} chunks):")
    print(f"  Total storage:   {format_bytes(total_file)}")


def main():
    parser = argparse.ArgumentParser(description='Measure memory usage of multimodal arousal detection')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for measurement')
    parser.add_argument('--base_ch', type=int, default=32, help='Base channels for model')
    parser.add_argument('--use_attention', action='store_true', default=True, help='Use attention fusion')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("🧠 Multimodal Arousal Detection - Memory Measurement")
    print("="*60)

    # System info
    print(f"\nSystem Information:")
    print(f"  Python: {psutil.python_version()}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory
        print(f"  GPU Memory: {format_bytes(total_mem)}")

    cpu_info = psutil.virtual_memory()
    print(f"  Total RAM: {format_bytes(cpu_info.total)}")
    print(f"  Available RAM: {format_bytes(cpu_info.available)}")

    # Preprocessing memory
    measure_chunk_preprocessing()

    # Model memory
    device = args.device if torch.cuda.is_available() else 'cpu'
    model = measure_model_memory(
        n_channels=9,
        base_ch=args.base_ch,
        use_attention=args.use_attention,
        device=device
    )

    # Forward pass
    measure_forward_memory(model, batch_size=args.batch_size, device=device)

    # Backward pass
    measure_backward_memory(model, batch_size=args.batch_size, device=device)

    # Optimizer
    optimizer = measure_optimizer_memory(model, device=device)

    # Full training step
    measure_full_training_step(model, optimizer, batch_size=args.batch_size, device=device)

    # Summary
    print("\n" + "="*60)
    print("📊 Summary")
    print("="*60)

    gpu_mem = get_gpu_memory()
    cpu_mem = get_cpu_memory()

    if gpu_mem:
        print(f"\nTotal GPU Memory Used: {format_bytes(gpu_mem['allocated'])}")
        print(f"Peak GPU Memory: {format_bytes(gpu_mem['max_allocated'])}")

    print(f"\nTotal CPU Memory Used: {format_bytes(cpu_mem['rss'])}")

    print("\n💡 Recommendations:")

    if gpu_mem:
        peak_mb = gpu_mem['max_allocated'] / (1024**2)

        if peak_mb < 2000:
            print(f"  ✅ Memory usage is low ({peak_mb:.0f} MB)")
            print(f"  ✅ Can increase batch_size or base_ch for better performance")
        elif peak_mb < 4000:
            print(f"  ✅ Memory usage is moderate ({peak_mb:.0f} MB)")
            print(f"  ✅ Current settings are good for most GPUs")
        elif peak_mb < 6000:
            print(f"  ⚠️  Memory usage is high ({peak_mb:.0f} MB)")
            print(f"  ⚠️  Consider using GPU with 8GB+ VRAM")
        else:
            print(f"  ❌ Memory usage is very high ({peak_mb:.0f} MB)")
            print(f"  ❌ Reduce batch_size or base_ch")
            print(f"  ❌ Suggested: --batch_size 2 --base_ch 16")

    print("\n" + "="*60)
    print("✨ Measurement Complete!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
