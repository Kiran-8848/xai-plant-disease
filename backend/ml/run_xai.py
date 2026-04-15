import torch
import json
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

from backend.ml.models.resnet_model import load_checkpoint, get_device
from backend.ml.utils.data_utils import get_inference_transforms, denormalize, build_dataloaders
from backend.ml.xai.gradcam import GradCAM
from backend.ml.xai.lime_explainer import LIMEExplainer
from backend.ml.xai.shap_explainer import SHAPExplainer, build_background_samples
from backend.ml.evaluation.metrics import (
    FaithfulnessEvaluator,
    RobustnessEvaluator,
    XAIComparator
)

def main():
    # ── Setup ──────────────────────────────────────────────
    device = get_device()

    model, meta = load_checkpoint(
        'backend/ml/checkpoints/best_model.pth',
        device
    )

    with open('backend/ml/checkpoints/class_mapping.json') as f:
        mapping      = json.load(f)
        idx_to_class = {int(k): v for k, v in mapping['idx_to_class'].items()}

    transform = get_inference_transforms(224)

    # ── Pick test images (one per class) ───────────────────
    data_root = Path('./data/PlantVillage')
    test_images = []
    for class_dir in sorted(data_root.iterdir()):
        if class_dir.is_dir():
            imgs = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.JPG')) + list(class_dir.glob('*.png'))
            if imgs:
                test_images.append(imgs[0])
            if len(test_images) >= 6:
                break

    print(f"Using {len(test_images)} test images")

    # Create output dirs
    Path('outputs/gradcam').mkdir(parents=True, exist_ok=True)
    Path('outputs/lime').mkdir(parents=True, exist_ok=True)
    Path('outputs/shap').mkdir(parents=True, exist_ok=True)
    Path('outputs/evaluation').mkdir(parents=True, exist_ok=True)

    # ════════════════════════════════════════════════════════
    # EXPERIMENT 2 — Generate XAI Explanations
    # ════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("EXPERIMENT 2 — Generating XAI Explanations")
    print("="*60)

    gcam = GradCAM(model, model.get_last_conv_layer())

    gcam_results = []
    for img_path in test_images:
        pil    = Image.open(img_path).convert('RGB')
        tensor = transform(pil).unsqueeze(0).to(device)
        hm, pred, conf = gcam.generate(tensor)
        original = denormalize(tensor.cpu()[0])
        overlay  = GradCAM.overlay_heatmap(original, hm)
        gcam_results.append({
            'path':     img_path,
            'original': original,
            'overlay':  overlay,
            'heatmap':  hm,
            'pred':     pred,
            'conf':     conf,
            'label':    idx_to_class.get(pred, str(pred))
        })
        print(f"  GradCAM: {img_path.parent.name[:30]} → {idx_to_class.get(pred,'?')[:25]} ({conf:.3f})")

    gcam.remove_hooks()

    # ── Plot GradCAM grid ──────────────────────────────────
    n   = len(gcam_results)
    fig, axes = plt.subplots(2, n, figsize=(n*3, 7))
    fig.suptitle('Grad-CAM Explanations', fontsize=14, y=1.01)
    for i, r in enumerate(gcam_results):
        axes[0, i].imshow(r['original'])
        axes[0, i].set_title(r['label'][:20], fontsize=8)
        axes[0, i].axis('off')
        axes[1, i].imshow(r['overlay'])
        axes[1, i].set_title(f"conf={r['conf']:.2f}", fontsize=8)
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Original', fontsize=10)
    axes[1, 0].set_ylabel('Grad-CAM', fontsize=10)
    plt.tight_layout()
    plt.savefig('outputs/gradcam/gradcam_grid.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved → outputs/gradcam/gradcam_grid.png")

    # ════════════════════════════════════════════════════════
    # EXPERIMENT 3 — Faithfulness Evaluation
    # ════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("EXPERIMENT 3 — Faithfulness (Deletion Test)")
    print("="*60)

    faith_eval = FaithfulnessEvaluator(
        model, device,
        percentages=[0.05, 0.10, 0.20, 0.30, 0.50],
        masking_method='blur'
    )

    faith_gcam_all = []

    for r in gcam_results[:3]:
        tensor = transform(
            Image.open(r['path']).convert('RGB')
        ).unsqueeze(0).to(device)

        target = int(torch.argmax(model(tensor)).item())
        result = faith_eval.evaluate(tensor, r['heatmap'], target)
        faith_gcam_all.append(result)

        print(f"  {r['path'].parent.name[:30]}")
        print(f"    Baseline conf: {result['baseline_conf']:.3f}")
        print(f"    AUC:           {result['auc']:.4f}")
        print(f"    Drop at 50%:   {result['drops'][-1]:.3f}")

    # ── Plot faithfulness curves ───────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = ['#f0a832', '#4d9ef5', '#f06060']

    for i, (r, result) in enumerate(zip(gcam_results[:3], faith_gcam_all)):
        label = r['path'].parent.name[:25]
        ax.plot(
            result['percentages'],
            result['confidences'],
            label=f"{label} (AUC={result['auc']:.3f})",
            color=colors[i], linewidth=2.5, marker='o', markersize=5
        )

    ax.set_xlabel('Fraction of pixels removed', fontsize=11)
    ax.set_ylabel('Model confidence', fontsize=11)
    ax.set_title('Faithfulness — Deletion Test (Grad-CAM)', fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig('outputs/evaluation/faithfulness_gradcam.png', dpi=150)
    plt.show()
    print("  Saved → outputs/evaluation/faithfulness_gradcam.png")

    # ════════════════════════════════════════════════════════
    # EXPERIMENT 4 — LIME Explanations + Faithfulness
    # ════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("EXPERIMENT 4 — LIME Explanations")
    print("="*60)

    lime_explainer = LIMEExplainer(
        model, device,
        image_size  = 224,
        num_samples = 500,
        num_features= 10,
    )

    lime_results   = []
    faith_lime_all = []

    for r in gcam_results[:3]:
        pil    = Image.open(r['path']).convert('RGB')
        tensor = transform(pil).unsqueeze(0).to(device)
        target = int(torch.argmax(model(tensor)).item())

        lime_r = lime_explainer.explain(pil, target_class=target)
        lime_results.append(lime_r)

        faith_r = faith_eval.evaluate(tensor, lime_r['heatmap'], target)
        faith_lime_all.append(faith_r)

        print(f"  {r['path'].parent.name[:30]}")
        print(f"    Confidence: {lime_r['confidence']:.3f}")
        print(f"    Time:       {lime_r['computation_time']:.1f}s")
        print(f"    Faith AUC:  {faith_r['auc']:.4f}")

    # ── Plot LIME explanations ─────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    fig.suptitle('LIME Explanations', fontsize=14)

    for i, (r, lime_r) in enumerate(zip(gcam_results[:3], lime_results)):
        from skimage.segmentation import mark_boundaries
        overlay = LIMEExplainer.overlay_segments(
            lime_r['original_image'],
            lime_r['mask'],
            lime_r['explanation'].segments
        )
        axes[0, i].imshow(lime_r['original_image'])
        axes[0, i].set_title(r['path'].parent.name[:20], fontsize=8)
        axes[0, i].axis('off')
        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f"conf={lime_r['confidence']:.2f}", fontsize=8)
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('Original', fontsize=10)
    axes[1, 0].set_ylabel('LIME', fontsize=10)
    plt.tight_layout()
    plt.savefig('outputs/lime/lime_grid.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved → outputs/lime/lime_grid.png")

    # ════════════════════════════════════════════════════════
    # EXPERIMENT 5 — SHAP Explanations + Faithfulness
    # ════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("EXPERIMENT 5 — SHAP Explanations")
    print("="*60)

    # Build background samples
    _, train_loader, _, _ = build_dataloaders(
        data_root   = './data/PlantVillage',
        batch_size  = 32,
        num_workers = 0,
    )
    background = build_background_samples(train_loader, n=50, device=device)

    shap_explainer = SHAPExplainer(model, background, device)

    shap_results   = []
    faith_shap_all = []

    for r in gcam_results[:3]:
        pil    = Image.open(r['path']).convert('RGB')
        tensor = transform(pil).unsqueeze(0).to(device)
        target = int(torch.argmax(model(tensor)).item())

        shap_r  = shap_explainer.explain(tensor, target_class=target)
        shap_results.append(shap_r)

        faith_r = faith_eval.evaluate(tensor, shap_r['pos_heatmap'], target)
        faith_shap_all.append(faith_r)

        print(f"  {r['path'].parent.name[:30]}")
        print(f"    Confidence: {shap_r['confidence']:.3f}")
        print(f"    Time:       {shap_r['computation_time']:.1f}s")
        print(f"    Faith AUC:  {faith_r['auc']:.4f}")

    # ── Plot SHAP explanations ─────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    fig.suptitle('SHAP Explanations', fontsize=14)

    for i, (r, shap_r) in enumerate(zip(gcam_results[:3], shap_results)):
        original = denormalize(
            transform(Image.open(r['path']).convert('RGB')).unsqueeze(0)[0]
        )
        vis = SHAPExplainer.visualize(original, shap_r['pos_heatmap'], shap_r['neg_heatmap'])
        axes[0, i].imshow(original)
        axes[0, i].set_title(r['path'].parent.name[:20], fontsize=8)
        axes[0, i].axis('off')
        axes[1, i].imshow(vis)
        axes[1, i].set_title(f"conf={shap_r['confidence']:.2f}", fontsize=8)
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('Original', fontsize=10)
    axes[1, 0].set_ylabel('SHAP', fontsize=10)
    plt.tight_layout()
    plt.savefig('outputs/shap/shap_grid.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved → outputs/shap/shap_grid.png")

    # ════════════════════════════════════════════════════════
    # EXPERIMENT 6 — Full Comparison + Robustness
    # ════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("EXPERIMENT 6 — Full XAI Comparison")
    print("="*60)

    # Use first image for comparison
    r      = gcam_results[0]
    pil    = Image.open(r['path']).convert('RGB')
    tensor = transform(pil).unsqueeze(0).to(device)
    target = int(torch.argmax(model(tensor)).item())

    # Define explain functions
    def explain_gradcam(t):
        g = GradCAM(model, model.get_last_conv_layer())
        hm, _, _ = g.generate(t)
        g.remove_hooks()
        return hm

    def explain_lime(t):
        pil_img = Image.fromarray(denormalize(t.cpu()[0]))
        res     = lime_explainer.explain(pil_img)
        return res['heatmap']

    def explain_shap(t):
        res = shap_explainer.explain(t)
        return res['pos_heatmap']

    # Faithfulness for all 3 methods on same image
    methods = {
        'GradCAM': (explain_gradcam, gcam_results[0]['heatmap']),
        'LIME':    (explain_lime,    lime_results[0]['heatmap']),
        'SHAP':    (explain_shap,    shap_results[0]['pos_heatmap']),
    }

    METHOD_COLORS = {
        'GradCAM': '#f0a832',
        'LIME':    '#4d9ef5',
        'SHAP':    '#f06060',
    }

    faith_compare = {}
    for name, (fn, hm) in methods.items():
        fr = faith_eval.evaluate(tensor, hm, target)
        faith_compare[name] = fr
        print(f"  {name:10s} → AUC={fr['auc']:.4f}  Drop@50%={fr['drops'][-1]:.3f}")

    # ── Robustness test ────────────────────────────────────
    print("\n  Running robustness tests...")
    robust_eval = RobustnessEvaluator(
        noise_levels     = [0.01, 0.05, 0.10, 0.20],
        num_repetitions  = 3,
    )

    robust_compare = {}
    for name, (fn, _) in methods.items():
        rb = robust_eval.evaluate(tensor, fn, 'noise')
        robust_compare[name] = rb
        print(f"  {name:10s} → mean_SSIM={rb['mean_ssim']:.4f}  mean_ρ={rb['mean_spearman']:.4f}")

    # ── Final comparison plot ──────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('XAI Method Comparison — Experiment 6', fontsize=14)

    # Plot 1: Faithfulness curves
    ax = axes[0]
    for name, fr in faith_compare.items():
        ax.plot(
            fr['percentages'], fr['confidences'],
            label=f"{name} (AUC={fr['auc']:.3f})",
            color=METHOD_COLORS[name], linewidth=2.5
        )
    ax.set_title('Faithfulness Curves')
    ax.set_xlabel('Pixels deleted')
    ax.set_ylabel('Confidence')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Plot 2: Robustness SSIM
    ax = axes[1]
    for name, rb in robust_compare.items():
        ax.plot(
            rb['levels'], rb['ssim'],
            label=name, color=METHOD_COLORS[name],
            linewidth=2.5, marker='o', markersize=5
        )
    ax.set_title('Robustness (SSIM)')
    ax.set_xlabel('Noise level')
    ax.set_ylabel('SSIM ↑')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 3: Computation time
    ax    = axes[2]
    names = list(methods.keys())
    # Approximate times based on your runs
    times = [0.05, float(lime_results[0]['computation_time']), float(shap_results[0]['computation_time'])]
    bars  = ax.bar(names, times, color=[METHOD_COLORS[n] for n in names], alpha=0.85, width=0.5)
    ax.set_title('Computation Time')
    ax.set_ylabel('Seconds (log)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, t in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() * 1.2,
            f'{t:.2f}s', ha='center', fontsize=10
        )

    plt.tight_layout()
    plt.savefig('outputs/evaluation/xai_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved → outputs/evaluation/xai_comparison.png")

    # ════════════════════════════════════════════════════════
    # FINAL SUMMARY TABLE
    # ════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"{'Method':<12} {'Faith.AUC':>10} {'Drop@50%':>10} {'SSIM':>8} {'Spearman':>10}")
    print("-" * 55)
    for name in ['GradCAM', 'LIME', 'SHAP']:
        fr = faith_compare[name]
        rb = robust_compare[name]
        print(
            f"{name:<12} "
            f"{fr['auc']:>10.4f} "
            f"{fr['drops'][-1]:>10.4f} "
            f"{rb['mean_ssim']:>8.4f} "
            f"{rb['mean_spearman']:>10.4f}"
        )

    # Save results to JSON
    import datetime
    summary = {
        'timestamp':     datetime.datetime.now().isoformat(),
        'test_accuracy': 0.9693,
        'results': {
            name: {
                'faithfulness_auc':  faith_compare[name]['auc'],
                'drop_at_50pct':     faith_compare[name]['drops'][-1],
                'mean_ssim':         robust_compare[name]['mean_ssim'],
                'mean_spearman':     robust_compare[name]['mean_spearman'],
            }
            for name in ['GradCAM', 'LIME', 'SHAP']
        }
    }
    with open('outputs/evaluation/final_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nAll results saved → outputs/evaluation/final_results.json")
    print("\nDone! All 6 experiments complete ✅")


if __name__ == '__main__':
    main()