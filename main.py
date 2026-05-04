import warnings
warnings.filterwarnings("ignore")

from portfolio_config import LABELS
from data.data import fetch_data
from garch.garch import rolling_garch_and_corr
from signals.signals import compute_signals, print_signal_table
from backtest.backtest import run_backtest, print_performance
from plots.plots import make_plots


def main():
    _, log_ret   = fetch_data()
    garch_res    = rolling_garch_and_corr(log_ret, LABELS)
    sig_df       = compute_signals(garch_res, LABELS)

    print_signal_table(sig_df, LABELS)

    strat_ret, bench_ret = run_backtest(sig_df, log_ret, LABELS)
    print_performance(sig_df, log_ret, strat_ret, bench_ret, LABELS)

    make_plots(sig_df, strat_ret, bench_ret, LABELS)


if __name__ == "__main__":
    main()
