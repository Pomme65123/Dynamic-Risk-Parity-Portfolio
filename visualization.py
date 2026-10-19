import matplotlib.pyplot as plt

def plot_performance(weights, strategy_cum, benchmark_cum):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})

    ax1.plot(strategy_cum.index, strategy_cum, label='ML Performance', color='#1f77b4', linewidth=2)
    ax1.plot(benchmark_cum.index, benchmark_cum, label='Benchmark', color='#7f7f7f', linestyle='--')
    ax1.set_title('Strategy vs. Benchmark: Cumulative Growth', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($1 Initial)')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left')
    ax1.margins(x=0)

    ax2.stackplot(weights.index, weights.T, labels=weights.columns, alpha=0.8)
    ax2.set_title('Dynamic Asset Allocation', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Weight Allocation')
    ax2.set_xlabel('Date')
    ax2.set_ylim(0, 1)
    ax2.margins(x=0)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig('portfolio_performance.png', dpi=300, bbox_inches='tight')
    print("Performance charts saved to 'portfolio_performance.png'")