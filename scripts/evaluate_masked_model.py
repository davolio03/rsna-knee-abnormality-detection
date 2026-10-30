"""Image-free comparison against training-only soft-label constant baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rsna_knee.validation.masked_baseline import evaluate_masked_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=Path('data/private/full-cohort-prefilter/run01/full-cohort-soft-label-manifest.json'))
    parser.add_argument('--folds',type=Path,default=Path('data/private/full-cohort-prefilter/run01/full-cohort-grouped-folds.json'))
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--report-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    private=(Path(__file__).resolve().parents[1]/'data/private').resolve()
    try:
        destination=args.output.absolute()
        destination.relative_to(private)
        if destination.exists() or any(path.is_symlink() for path in (destination,*destination.parents)):
            raise ValueError('output_must_be_new')
        result=evaluate_masked_report(args.manifest,args.folds,args.report,args.report_sha256)
        destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open('x',encoding='utf-8') as stream:
            stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    except Exception:
        parser.error('masked_model_evaluation_failed')
    print(json.dumps({'status':'complete','model':result['model'],'constant':result['constant'],
                      'bce_targets_better_than_constant':result['bce_targets_better_than_constant']},sort_keys=True))


if __name__=='__main__': main()
