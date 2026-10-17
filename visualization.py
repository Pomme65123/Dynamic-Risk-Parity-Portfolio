import matplotlib.pyplot as plt

def plot_performance(weights, strategy_cum, benchmark_cum):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(strategy_cum.index, strategy_cum, label='ML Performance', color='#1f77b4', linewidth=2)
    ax.plot(benchmark_cum.index, benchmark_cum, label='Benchmark', color='#7f7f7f', linestyle='--')
    
    ax.set_title('Strategy vs. Benchmark: Cumulative Growth', fontsize=14, fontweight='bold')
    ax.set_ylabel('Portfolio Value ($1 Initial)')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left')
    ax.margins(x=0)
    
    plt.tight_layout()
    plt.savefig('portfolio_performance.png', dpi=300, bbox_inches='tight')