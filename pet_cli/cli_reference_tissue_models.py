import argparse
import numpy as np
from typing import Union
from .reference_tissue_models import RTMAnalysis


def add_common_args(parser):
    parser.add_argument('-e', '--ref-tac-path', required=True, type=str,
                        help='Absolute path for reference TAC')
    parser.add_argument('-i', '--roi-tac-path', required=True, type=str,
                        help='Absolute path for ROI TAC')
    parser.add_argument('-o', '--output-directory', required=True, type=str,
                        help='Absolute path for the output directory')
    parser.add_argument('-f', '--output-filename-prefix', required=True, type=str,
                        help='Prefix for the output filename of the result')
    parser.add_argument('--print', required=False, action='store_true',
                        help="Whether to print the fitting results to screen.")


def parse_args():
    parser = argparse.ArgumentParser(prog='pet-cli-rtms',
                                     description='Command line interface for running RTMAnalysis.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="method", help="Analysis method")

    parser_srtm = subparsers.add_parser('srtm', help='Perform SRTM analysis')
    parser_frtm = subparsers.add_parser('frtm', help='Perform SRTM analysis')
    parser_mrtm = subparsers.add_parser('mrtm',
                                        help='Perform Ichise\'s MRTM (2003) analysis')
    parser_mrtm_2 = subparsers.add_parser('mrtm2',
                                          help='Perform Ichise\'s MRTM2 (2003) analysis')
    parser_mrtm_original = subparsers.add_parser('mrtm-original',
                                                 help='Perform Ichise\'s original MRTM (1996) analysis')
    parser_list = [parser_srtm, parser_frtm, parser_mrtm, parser_mrtm_2, parser_mrtm_original]
    
    for a_parser in parser_list[:]:
        add_common_args(a_parser)
        if 'mrtm' in a_parser.prog:
            a_parser.add_argument('-t', '--t-thresh-in-mins', required=True, type=float,
                                  help='Threshold time in minutes for the MRTM analyses.')
        else:
            a_parser.add_argument("-g", "--initial-guesses", required=False, nargs='+', type=float,
                                      help="Initial guesses for each fitting parameter.")
            a_parser.add_argument("-l", "--lower-bounds", required=False, nargs='+', type=float,
                                      help="Lower bounds for each fitting parameter.")
            a_parser.add_argument("-u", "--upper-bounds", required=False, nargs='+', type=float,
                                      help="Upper bounds for each fitting parameter.")
        if a_parser.prog.endswith('2'):
            parser_mrtm_2.add_argument('-k', '--k2-prime', required=True, type=float,
                                       help='k2_prime value for the reduced RTM analysis.')

    return parser.parse_args()


def _generate_bounds(initial: Union[list, None],
                     lower: Union[list, None],
                     upper: Union[list, None]) -> Union[np.ndarray, None]:
    r"""
    Generates the bounds for the fitting parameters.

    This function takes lists of initial fitting parameters, lower bounds, and upper bounds.
    All lists must have the same length. If no initial parameters are provided, the function returns None.

    Args:
        initial (list, optional): List of initial guesses for fitting parameters. If None, no bounds are generated.
        lower (list, optional): List of lower bounds for fitting parameters.
        upper (list, optional): List of upper bounds for fitting parameters.

    Returns:
        (np.ndarray, optional): If initial is not None, then we return a numpy array of shape [n, 3],
        where n is the number of parameters, where column 0 has the initial guesses, column 1 has lower bounds, and
        column 2 has upper bounds. If initial is None, function will return None.

    Raises:
        ValueError: If initial is not None and the length of initial, lower, and upper are not the same.
    """
    if initial is not None:
        if upper is None or lower is None:
            raise ValueError("The lower and upper bounds must also be set.")
        if (len(initial) != len(lower)) or (len(initial) != len(upper)) or (len(upper) != len(lower)):
            raise ValueError("The number of initial guesses, lower bounds and upper bounds must be the same.")
        return np.asarray(np.asarray([initial, lower, upper]).T)
    else:
        return None


def _pretty_print_results(vals: dict, errs: dict) -> None:
    title_str = f"{'Param':<5} {'FitVal':<6}    {'StdErr':<6} ({'%Err':>6})|"
    print("-" * len(title_str))
    print(title_str)
    print("-" * len(title_str))
    for param_name in vals:
        val = vals[param_name]
        err = errs[param_name]
        print(f"{param_name:<5} {val:<6.4f} +- {err:<6.4f} ({err / val * 100:>5.2f}%)|")
    print("-" * len(title_str))


def main():
    args = parse_args()
    
    analysis = RTMAnalysis(ref_tac_path=args.ref_tac_path,
                           roi_tac_path=args.roi_tac_path,
                           output_directory=args.output_directory,
                           output_filename_prefix=args.output_filename_prefix,
                           method=args.method)
    
    if args.method.startswith('mrtm'):
        analysis.run_analysis(t_thresh_in_mins=args.t_thresh_in_mins,
                              k2_prime=getattr(args, 'k2_prime', None))
    else:
        bounds = _generate_bounds(initial=args.initial_guesses, lower=args.lower_bounds, upper=args.upper_bounds)
        analysis.run_analysis(bounds=bounds)
    
    analysis.save_analysis()

    if args.print:
        if args.method.startswith('mrtm'):
            print(f"BP      : {analysis.analysis_props['BP']:<.7f}")
            print(f"k2_prime: {analysis.analysis_props['k2Prime']:<.7f}")
        else:
            _pretty_print_results(vals=analysis.analysis_props["FitValues"],
                                  errs=analysis.analysis_props["FitStdErr"])
    

if __name__ == '__main__':
    main()