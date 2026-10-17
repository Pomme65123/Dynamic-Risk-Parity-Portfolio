import matplotlib.pyplot as plt

def plot_performance(weights, strategy_cum, benchmark_cum):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    axes[0].plot(strategy_cum.index, strategy_cum, label='ML Performance', color='#1f77b4', linewidth=2)
    axes[0].plot(benchmark_cum.index, benchmark_cum, label='Naive Performance', color='#7f7f7f', linestyle='--')
    
    axes[0].set_title('Strategy vs. Benchmark: Cumulative Growth', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Portfolio Value ($1 Initial)')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='upper left')
    axes[1].stackplot(weights.index, weights.T, labels=weights.columns, alpha=0.85)
    
    axes[1].set_title('Dynamic Asset Allocation (Risk Parity Weights)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Capital Allocation (%)')
    axes[1].set_ylim(0, 1)
    axes[1].margins(x=0)
    
    axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    
    plt.tight_layout()
    plt.savefig('portfolio_performance.png', dpi=300, bbox_inches='tight')
    print("=> Visualization saved successfully as 'portfolio_performance.png'")